# The shell line an `ExecutionPlan` wraps a command in on a host once ssh'd in: `cd` into the
# workspace root, stack per-user install dirs onto `PATH`, load modules, then env/container.

import shlex
from typing import TYPE_CHECKING

from tenacity import retry as tenacity_retry
from tenacity import retry_if_exception_type, stop_after_attempt, wait_fixed

from ..core.project import Project
from .transport import BoundedSshMachine, HostUnreachable, SshTransport

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..context.plan import ExecutionPlan

# Per-user install dirs prepended to PATH first, so an already-installed `mainboard` is found
# on a fresh host before any env is even activated.
_USER_BINS = ("$HOME/.local/bin", "$HOME/.pixi/bin", "$HOME/.cargo/bin")

# A connect-time transport blip is the same transient fault a wait loop rides out, so it is
# retried here too; a host-key failure is not transient and is never retried.
_CONNECT_ATTEMPTS = 4
_CONNECT_BACKOFF = 2.0


def activation(root: str, env: str = "default") -> str:
    """The activation script a provisioned workspace under `root` carries for `env`.

    root: the workspace root on the machine the command runs on.
    env: the environment the script activates.
    """
    return f"{root}/{Project().activation(env)}"


def wrap(
    plan: ExecutionPlan,
    root: str,
    *,
    command: str,
    containerize: Callable[[list[str]], list[str]] | None = None,
    activate: bool = True,
) -> str:
    """The activated shell line for `command`, staged as `cd`, `PATH`, modules, then env/container.

    The env stage runs `command` straight after sourcing the plan's env prefix (falling back to a
    PATH prepend when the prefix was never built), unless the plan is containerized, in which case
    `containerize` wraps `command` in the container runtime's own argv instead.

    plan: the resolved execution context (profile, env, container).
    root: the workspace root on the host commands run from.
    command: the bare command to run once activated.
    containerize: builds the container runtime argv around an inner `["bash", "-c", command]`
        argv; required when `plan.containerized`, so the integrator (not this module) owns how a
        base image is actually invoked.
    activate: stage the environment (or the container) before running `command`. False keeps
        only `cd`, `PATH` and modules, the footing an onboarding stands on while the host has
        no environment to activate yet.
    """
    steps = [f"cd {shlex.quote(root)}", f"export PATH={':'.join(_USER_BINS)}:$PATH"]
    if plan.profile.modules:
        steps.append("module purge")
        steps += [
            f"module load {name}/{version}" if version else f"module load {name}"
            for name, version in plan.profile.modules.items()
        ]
    if not activate:
        steps.append(command)
    elif plan.containerized:
        if containerize is None:
            raise LookupError(
                f"plan for host {plan.host!r} is containerized but no container argv "
                "builder was given"
            )
        steps.append(shlex.join(containerize(["bash", "-c", command])))
    else:
        steps.append(activation_stage(plan.prefix(root), root, plan.env))
        steps.append(command)
    return " && ".join(steps)


def activation_stage(
    prefix: str, root: str = "", env: str = "default", *, strict: bool = False
) -> str:
    """The shell stage that activates a provisioned workspace before a command runs.

    The one activation both a wrapped line and a rendered job script use, so an interactive run
    and a queued job never disagree about which interpreter they got. It sources the environment's
    own generated activation when there is one and otherwise falls back to that environment
    prefix's `bin/`. Only the default environment ever falls through to a script another
    environment wrote, since sourcing one named environment's activation for another silently
    hands the command the wrong interpreter, which is exactly what `--env serving` must never do.
    `strict` refuses the silent fall-through when nothing exists at all, which is what a
    dispatched job wants: a queued job that quietly ran the host's system python costs a whole
    scheduler round trip to discover, while an interactive command may legitimately need nothing
    activated at all.

    REMOVE AT CHEFE ARCHIVE: the `.chefe/activate.sh` branch is transitional, for hosts chefe
    provisioned, and chefe only ever wrote one for the default environment. Once every host has
    been set up through `Board.install` and chefe is archived, that branch goes and the stage
    keeps the generated activation and the prefix.

    prefix: the environment prefix on the machine the command runs on.
    root: the workspace root there; empty when the caller knows only the prefix, which leaves
        the workspace activation out of the chain entirely.
    env: the environment being activated, naming which generated script to source.
    strict: fail loudly when nothing can be activated, instead of running the command anyway.
    """
    scripts: list[str] = []
    if root:
        scripts.append(activation(root, env))
        if env == "default":
            scripts.append(f"{root}/.chefe/activate.sh")
    prepend = f"export PATH={shlex.quote(prefix)}/bin:$PATH"
    branches = [(f"[ -f {shlex.quote(path)} ]", f"source {shlex.quote(path)}") for path in scripts]
    if strict:
        branches.append((f"[ -d {shlex.quote(prefix)}/bin ]", prepend))
        message = (
            f"mainboard: no environment at {prefix} on this host; "
            "set the host up before dispatching"
        )
        closing = f"echo {shlex.quote(message)} >&2; exit 1"
    else:
        closing = prepend
    if not branches:
        return closing
    chain = "; ".join(
        f"{'if' if index == 0 else 'elif'} {test}; then {action}"
        for index, (test, action) in enumerate(branches)
    )
    return f"{chain}; else {closing}; fi"


def argv(
    plan: ExecutionPlan,
    root: str,
    *,
    command: str,
    login: bool = True,
    containerize: Callable[[list[str]], list[str]] | None = None,
) -> list[str]:
    """`wrap` under a `bash` login (`-lc`) or plain (`-c`) shell, ready for plumbum/subprocess."""
    flag = "-lc" if login else "-c"
    return ["bash", flag, wrap(plan, root, command=command, containerize=containerize)]


def connection(host: str, ssh: SshTransport | None = None) -> BoundedSshMachine:
    """Open an ssh connection to `host` with the per-user install dirs on PATH.

    First warms the host's ssh `ControlMaster` (from `~/.ssh/config`) with a throwaway one-shot
    `ssh`: if the persistent master has expired, that slow relogin happens on a robust one-shot
    channel, so plumbum's persistent session then rides a live master instead of dying mid-
    handshake during the reconnect. We do not set `ControlMaster`/`ControlPath` ourselves, the
    user's config owns the multiplexing; overriding it would open a second, unauthenticated
    master.

    That same warm-up doubles as a host-key check: a failed verification (the host or its
    ProxyJump rotated its key, or the entry is missing) raises a clear, actionable error here
    instead of an opaque plumbum traceback. A transport-level warm-up failure (a refused session
    under MaxSessions, a dropped link) is retried a few times before giving up, the same
    transient-fault footing a wait loop already stands on.

    host: the ssh alias to connect to.
    ssh: the bounded SSH policy; the default policy when omitted.
    """
    policy = ssh or SshTransport()
    retrying = tenacity_retry(
        retry=retry_if_exception_type(HostUnreachable),
        stop=stop_after_attempt(_CONNECT_ATTEMPTS),
        wait=wait_fixed(_CONNECT_BACKOFF),
        reraise=True,
    )
    return retrying(_open)(host, policy)


def _open(host: str, ssh: SshTransport) -> BoundedSshMachine:
    """One attempt to open the connection: warm the master, key-check, then build the session.

    Raises `HostUnreachable` on a transient transport fault (so the caller retries) and
    `ConnectionError` on a host-key failure (which no retry can fix).
    """
    ssh.warm(host)
    remote = ssh.machine(host)
    for bindir in reversed(_USER_BINS):
        remote.env.path.insert(0, remote.cwd / bindir.removeprefix("$HOME/"))
    return remote
