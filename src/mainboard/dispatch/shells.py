# One host's command shell: how a line is staged and run there, and how the few probes onboarding
# and the board need are spelled. Every host answers `bash -lc` over plumbum's persistent session.

import shlex
from importlib.metadata import metadata
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Self

from ..core.errors import MissionError
from ..core.project import Project
from ..core.shell import foreground
from ..engines.compile.backend import POSIX_INSTALLER
from .schedulers.base import failure_reason
from .ssh import client
from .targets import placed
from .transport import BoundedSshMachine
from .wrapping import USER_BINS, activation, connection, guarded, wrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context.plan import ExecutionPlan
    from .transport import Machine, SshTransport

# uv's official installer, the one way uv reaches a host that has none.
_UV_INSTALLER = "curl -LsSf https://astral.sh/uv/install.sh | sh"


class Dialect:
    """How a login-`bash` host, Linux or macOS, cluster or rental, spells the lines dispatch
    needs, with no connection in hand."""

    noop = "true"
    # The probe and the line that fetch uv onto a host that has none.
    uv_bootstrap = ("command -v curl", _UV_INSTALLER)
    pixi_installer = POSIX_INSTALLER
    # The interpreters the tool runs on, as its own metadata requires them.
    requires_python = metadata(Project().package)["Requires-Python"]

    def python(self, prefix: str, *, host: str) -> str:
        """The command that starts mainboard's own Python on `host`, under any login shell.

        The workspace environment at `prefix` when the host has it, else the uv-managed CPython
        the tool is installed onto, which onboarding puts there before anything ships. Never the
        machine's own `python3`, whose version and packages nobody here chose. The arguments
        written after the command reach the interpreter as they are.
        """
        found = f"{placed(prefix, home='$HOME')}/bin/python"
        wanted = self.requires_python
        refusal = (
            f"$0: {host} has neither {found} nor a uv-managed CPython {wanted}; "
            f"run {Project().name} host setup {host}"
        )
        script = (
            f'p="{found}"; [ -x "$p" ] || '
            f'p=$(PATH={":".join(USER_BINS)}:$PATH; uv python find --managed-python "{wanted}") '
            f'|| {{ echo "{refusal}" >&2; exit 127; }}; exec "$p" "$@"'
        )
        return f"sh -c {shlex.quote(script)} {Project().package}"

    def stage(self, plan: ExecutionPlan, root: str, *, command: str, activate: bool) -> str:
        """The line that runs `command` from the workspace, its environment entered when asked."""
        return wrap(plan, root, command=command, activate=activate)

    def session(self, host: str, line: str) -> list[str]:
        """The argv that hands this terminal to `line` on `host`.

        `-t` forces the pty the far side needs, and the staged line is quoted whole because ssh
        joins its argv back into one string for the remote login shell to parse.
        """
        return [str(client()), "-t", host, f"bash -lc {shlex.quote(line)}"]

    def one_shot(self, ssh: SshTransport, host: str, line: str) -> tuple[str, ...]:
        """The argv of one ssh process that runs the staged `line` on `host` and returns."""
        return (*ssh.command(host), f"bash -lc {shlex.quote(line)}")

    def has(self, tool: str) -> str:
        """The probe exiting zero when `tool` is on the host's PATH."""
        return f"command -v {tool}"

    def is_directory(self, path: str) -> str:
        return f"[ -d {shlex.quote(path)} ]"

    def is_file(self, path: str) -> str:
        return f"test -f {shlex.quote(path)}"

    def proof(self, plan: ExecutionPlan, root: str) -> str:
        """The activation script whose presence proves `plan`'s environment was provisioned."""
        return activation(root, env=plan.env)

    def invocation(self, argv: Sequence[str]) -> str:
        """`argv` as one shell command, every word passed on as it is."""
        return shlex.join(argv)


class HostShell:
    """A host's shell staged by an execution plan, the one way a remote command is run, over
    plumbum's persistent ssh session `remote`.

    Two footings: a bare command gets `cd`, the per-user install dirs on `PATH` and the host's
    modules, all an unprovisioned machine can offer, while an activated one also enters the
    environment, which is what proves the environment an install just built actually runs.
    """

    dialect = Dialect()

    def __init__(self, remote: Machine, plan: ExecutionPlan, root: str) -> None:
        self.remote = remote
        self.plan = plan
        self.root = root

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        if isinstance(self.remote, BoundedSshMachine):
            self.remote.close()

    def ok(self, command: str, *, activate: bool = False) -> bool:
        """Whether `command` exits zero on the host, its output discarded."""
        retcode, _, _ = self.execute(self.stage(command, activate=activate))
        return retcode == 0

    def run(self, command: str, *, activate: bool = False) -> str:
        """`command`'s stdout on the host, raising a `MissionError` naming why it failed."""
        retcode, out, err = self.execute(self.stage(command, activate=activate))
        if retcode:
            reason = failure_reason(err or out, retcode)
            raise MissionError(f"`{command}` failed on {self.plan.host!r}: {reason}")
        return out

    def stage(self, command: str, *, activate: bool) -> str:
        return self.dialect.stage(self.plan, self.root, command=command, activate=activate)

    def place(self) -> None:
        """Make the workspace root, so a machine that has none yet can stand in it."""
        retcode, _, err = self.execute(f"mkdir -p {shlex.quote(self.root)}")
        if retcode:
            reason = failure_reason(err, retcode)
            raise MissionError(f"cannot make {self.root} on {self.plan.host!r}: {reason}")

    @property
    def proof(self) -> str:
        """The path that proves the plan's environment was provisioned here."""
        return self.dialect.proof(self.plan, self.root)

    @property
    def provisioned(self) -> str:
        """The probe exiting zero once the plan's environment exists here."""
        return self.dialect.is_file(self.proof)

    def execute(self, line: str) -> tuple[int, str, str]:
        """Run one already staged `line` and answer its exit status, stdout and stderr."""
        return self.remote["bash"][["-lc", guarded(line, self.plan)]].run(retcode=None)

    def foreground(self, command: str, *, activate: bool = True) -> int:
        """Run `command` with this terminal's stdio and answer its exit status."""
        return foreground(self.remote["bash"]["-lc", self.stage(command, activate=activate)])

    def write(self, path: str, text: str) -> None:
        """Write `text` to `path` on the host, private to the user, its directory made."""
        parent = shlex.quote(str(PurePosixPath(path).parent))
        written = f"umask 077; mkdir -p {parent}; cat > {shlex.quote(path)}"
        (self.remote["bash"]["-c", written] << text)()


def open_shell(plan: ExecutionPlan, root: str, *, ssh: SshTransport | None = None) -> HostShell:
    """`plan`'s host's login-bash session, connected; `ssh` defaults to the bounded policy."""
    return HostShell(connection(plan.host, ssh), plan, root)
