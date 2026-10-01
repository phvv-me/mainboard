# The shell line an `ExecutionPlan` wraps a command in on a host once ssh'd in: `cd` into the
# workspace root, stack per-user install dirs onto `PATH`, load modules, then env/container.

import shlex
from typing import TYPE_CHECKING

from tenacity import retry as tenacity_retry
from tenacity import retry_if_exception_type, stop_after_attempt, wait_fixed

from ..core.project import Project
from ..engines.compile.generated.activation import module_specs
from .transport import BoundedSshMachine, HostUnreachable, SshTransport

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..context.plan import ExecutionPlan

# Per-user install dirs prepended to PATH first, so an already-installed `mainboard` is found
# on a fresh host before any env is even activated. pixi's global tools live in either home: its
# own `~/.pixi`, or the dotfiles' one per architecture (`~/.pixi/<uname -m>`, so an HPC home
# shared by x86-64 and aarch64 nodes works), which is where `pixi global install` writes there.
USER_BINS = (
    "$HOME/.local/bin",
    "$HOME/.pixi/$(uname -m)/bin",
    "$HOME/.pixi/bin",
    "$HOME/.cargo/bin",
)

# A connect-time transport blip is the transient fault a wait loop rides out, so it is retried.
_CONNECT_ATTEMPTS = 4
_CONNECT_BACKOFF = 2.0

# The share of a login node's per-user limit a guarded line may let the user reach, the bound
# the owner set (2026-10-01), and how many seconds apart the guard reads the user's memory.
_LOGIN_SHARE = 0.75
_LOGIN_POLL = 2


def guarded(line: str, plan: ExecutionPlan) -> str:
    """`line`, stopped with everything it started once all of this user's processes on the
    login node hold `_LOGIN_SHARE` of the host's `login_memory_gb`; unchanged where it has none.

    A site enforcing a per-user limit kills the largest tasks itself, and on miyabi-g that ended
    the one login its owner unlocks every 72 hours, mid-install (2026-10-01); stopping short
    turns that into an ordinary failure that names the memory. The line runs in a session of its
    own, so the stop reaches its whole process tree and nothing of the shell it rides.
    """
    if not plan.profile.login_memory_gb:
        return line
    budget = plan.profile.login_memory_gb * _LOGIN_SHARE
    limit = int(budget * 2**20)  # KiB, the unit ps reports resident memory in
    held = """$(ps -u "$(id -u)" -o rss= | awk '{s += $1} END {print s + 0}')"""
    stop = (
        f"echo \"stopped: this user held $used KiB on {plan.host}'s login node, past "
        f'{budget:g} GB, {_LOGIN_SHARE:.0%} of its limit" >&2; kill -- -$pid; wait $pid; exit 137'
    )
    return (
        f"setsid bash -c {shlex.quote(line)} & pid=$!; "
        f"while kill -0 $pid 2>/dev/null; do used={held}; "
        f'if [ "$used" -gt {limit} ]; then {stop}; fi; sleep {_LOGIN_POLL}; done; wait $pid'
    )


def activation(root: str, *, env: str = "default") -> str:
    """The activation script a provisioned workspace under `root` carries for `env`, under the
    state directory this workspace keeps, which a host's mirror of it keeps too."""
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

    The env stage activates the plan's environment (refusing when it was never provisioned), or,
    for a containerized plan, has `containerize` wrap `command` in the runtime's own argv.

    containerize: builds the container runtime argv around an inner `["bash", "-c", command]`;
        required when `plan.containerized`, so the integrator owns how a base image is invoked.
    activate: False keeps only `cd`, `PATH` and modules, the footing an onboarding stands on
        while the host has no environment to activate yet.
    """
    steps = [f"cd {shlex.quote(root)}", f"export PATH={':'.join(USER_BINS)}:$PATH"]
    if plan.profile.modules:
        steps += [
            "module purge",
            *(f"module load {shlex.quote(spec)}" for spec in module_specs(plan.profile.modules)),
        ]
    if not activate:
        return " && ".join([*steps, command])
    # Host overrides follow activation, just as in a submitted job, so a workspace GPU mask
    # never replaces a host's reserved card.
    exported = " && ".join(
        [f"export {key}={shlex.quote(value)}" for key, value in plan.exports.items()] + [command]
    )
    if not plan.containerized:
        return " && ".join([*steps, activation_stage(plan, root), exported])
    if containerize is None:
        raise LookupError(
            f"plan for host {plan.host!r} is containerized but no container argv builder was given"
        )
    return " && ".join([*steps, shlex.join(containerize(["bash", "-c", exported]))])


def activation_stage(plan: ExecutionPlan, root: str) -> str:
    """The shell stage that activates `plan`'s environment before a wrapped command runs.

    It sources the environment's generated activation, else puts its prefix's `bin/` on PATH.
    With neither, the default environment runs on a bare PATH, since an interactive command may
    need nothing activated; a named one refuses, naming the command that provisions it, because
    naming it states which interpreter the user wants, and falling through is how a command asking
    for `vserve` silently runs the system python.

    A dispatched job never comes through here: its runner enters the environment itself and
    refuses the default one just the same, since a queued run on the host's system python costs a
    whole scheduler round trip to find out.
    """
    prefix = shlex.quote(plan.prefix(root))
    script = shlex.quote(activation(root, env=plan.env))
    prepend = f"export PATH={prefix}/bin:$PATH"
    closing = (
        prepend
        if plan.env == "default"
        else f"echo {shlex.quote(missing(plan, plan.prefix(root)))} >&2; exit 1"
    )
    return (
        f"if [ -f {script} ]; then source {script}; "
        f"elif [ -d {prefix}/bin ]; then {prepend}; else {closing}; fi"
    )


def absent(prefix: str, env: str) -> str:
    """The refusal a job whose addressed prefix is missing or unfinished prints, naming the
    host's own tool by the script every release installs."""
    tool = Project().package
    return (
        f"{tool} found no completed environment with the expected identity at {prefix}. It is "
        "addressed by the content of the manifest and lock this job was dispatched with, so "
        f"`{tool} provide {env}` rebuilds exactly it."
    )


def missing(plan: ExecutionPlan, prefix: str) -> str:
    """The refusal a machine with nothing to activate at `prefix` prints, naming the fix."""
    tool = Project().name
    where, fix = (
        (prefix, f"install {plan.env}")
        if plan.host == "local"
        else (f"{prefix} on {plan.host}", f"setup {plan.host} --env {plan.env}")
    )
    return (
        f"{tool} found no {plan.env} environment at {where}. Run `{tool} {fix}` to provision it."
    )


def connection(host: str, ssh: SshTransport | None = None) -> BoundedSshMachine:
    """Open an ssh connection to `host` with the per-user install dirs on PATH.

    A throwaway one-shot `ssh` first warms the host's `ControlMaster` from `~/.ssh/config`, so an
    expired master relogs on a robust channel and plumbum's persistent session rides a live one
    instead of dying mid-handshake. We never open a master here: the user's config owns
    multiplexing, and the only master this tool opens is the one `mb host unlock` logs into by
    hand, which every ssh then rides in proxy mode.

    The warm-up doubles as a host-key check: a failed verification (a rotated key on the host or
    its ProxyJump, a missing entry) raises a clear `ConnectionError` instead of an opaque plumbum
    traceback, never retried. A transport fault (a session refused under MaxSessions, a dropped
    link) raises `HostUnreachable` and is retried a few times.

    ssh: the bounded SSH policy; the default policy when omitted.
    """
    retrying = tenacity_retry(
        retry=retry_if_exception_type(HostUnreachable),
        stop=stop_after_attempt(_CONNECT_ATTEMPTS),
        wait=wait_fixed(_CONNECT_BACKOFF),
        reraise=True,
    )
    return retrying(_open)(host, ssh or SshTransport())


def _open(host: str, ssh: SshTransport) -> BoundedSshMachine:
    """One attempt: warm the master, key-check, then build the session."""
    ssh.warm(host)
    remote = ssh.machine(host)
    arch = remote["uname"]("-m").strip()
    for bindir in reversed(USER_BINS):
        remote.env.path.insert(
            0, remote.cwd / bindir.removeprefix("$HOME/").replace("$(uname -m)", arch)
        )
    return remote
