# One host's command shell: how a line is staged and run there, and how the few probes onboarding
# and the board need are spelled. A POSIX host answers `bash -lc` over plumbum's persistent
# session. A Windows host has no bash and no `/bin/sh` for plumbum to open a session with, so it
# answers PowerShell one-shots over the bounded transport instead, each script base64-encoded
# the way `-EncodedCommand` takes it so nothing is ever quoted for cmd.exe.

import base64
import html
import re
import shlex
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=ssh argv built from typed fields, not untrusted input since=2026-09-11
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Self

from ..core.errors import MissionError
from ..core.host import platform_family
from ..core.project import Project
from ..core.shell import foreground
from ..engines.compile.backend import POSIX_INSTALLER, WINDOWS_INSTALLER
from .schedulers.base import failure_reason
from .transport import BoundedSshMachine, SshTransport
from .wrapping import activation, connection, wrap

if TYPE_CHECKING:
    from ..context.plan import ExecutionPlan
    from ..manifest.schema.host import HostProfile
    from .transport import Machine

_TOOL = Project().name

# uv's official installer, used only when a host has neither uv nor pip to install the tool with.
_UV_INSTALLER = "curl -LsSf https://astral.sh/uv/install.sh | sh"

# Per-user install dirs a PowerShell stage puts ahead of PATH, uv's and pixi's own targets.
_WINDOWS_BINS = ("$HOME\\.local\\bin", "$HOME\\.pixi\\bin", "$HOME\\.cargo\\bin")

# The PowerShell argv a Windows host runs a script through, the script encoded after it.
POWERSHELL = (
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy",
    "Bypass",
    "-EncodedCommand",
)


def is_windows(profile: HostProfile) -> bool:
    """Whether `profile` declares (or was resolved to) a Windows platform."""
    return platform_family(profile.platform) == "win"


def encoded(script: str) -> str:
    """`script` as PowerShell's `-EncodedCommand` carries it: base64 over UTF-16LE."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def quoted(value: str) -> str:
    """`value` as a PowerShell single-quoted literal, the one form that expands nothing."""
    return "'" + value.replace("'", "''") + "'"


def plain_errors(stderr: str) -> str:
    """`stderr` with PowerShell's CLIXML wrapping undone, the error records' text alone.

    A powershell.exe whose stderr is a pipe rather than a console serializes its error stream
    as CLIXML, one `<S S="Error">` element per line with newlines escaped, which is what a
    failure would otherwise be reported through verbatim.
    """
    if not stderr.lstrip().startswith("#< CLIXML"):
        return stderr
    records = re.findall(r'<S S="Error">(.*?)</S>', stderr, flags=re.DOTALL)
    text = "".join(records).replace("_x000D_", "").replace("_x000A_", "\n")
    return html.unescape(text).strip()


def _exit_on(test: str) -> str:
    """A PowerShell line exiting zero when `test` holds, the shape a shell probe answers in."""
    return f"if ({test}) {{ exit 0 }} else {{ exit 1 }}"


class Dialect(ABC):
    """How one family of hosts spells the lines dispatch needs, with no connection in hand."""

    @abstractmethod
    def stage(self, plan: ExecutionPlan, root: str, *, command: str, activate: bool) -> str:
        """The line that runs `command` from the workspace, its environment entered when asked."""

    @abstractmethod
    def session(self, host: str, line: str) -> list[str]:
        """The argv that hands this terminal to `line` on `host`."""

    @abstractmethod
    def has(self, tool: str) -> str:
        """The probe exiting zero when `tool` is on the host's PATH."""

    @abstractmethod
    def is_directory(self, path: str) -> str:
        """The probe exiting zero when `path` is a directory on the host."""

    @abstractmethod
    def is_file(self, path: str) -> str:
        """The probe exiting zero when `path` is a file on the host."""

    @abstractmethod
    def chain(self, *commands: str) -> str:
        """`commands` run in order, each only after the previous one succeeded."""

    @property
    @abstractmethod
    def noop(self) -> str:
        """The command that succeeds doing nothing."""

    @property
    @abstractmethod
    def uv_bootstrap(self) -> tuple[str, str]:
        """The probe and the line that fetch uv onto a host that has none."""

    @property
    @abstractmethod
    def pip(self) -> tuple[str, str]:
        """The probe for a usable pip and the user-site install line it completes."""

    @property
    @abstractmethod
    def pixi_installer(self) -> str:
        """pixi's official installer at the fleet's pinned version."""

    @abstractmethod
    def proof(self, plan: ExecutionPlan, root: str) -> str:
        """The path whose presence proves `plan`'s environment was provisioned under `root`."""

    @abstractmethod
    def provisioned(self, plan: ExecutionPlan, root: str) -> str:
        """The probe exiting zero once `proof` exists."""

    @abstractmethod
    def activation_record(self, plan: ExecutionPlan, root: str) -> str:
        """The activation script a `HostSetup` records, empty where a host sources none."""


class Posix(Dialect):
    """The login-`bash` family: Linux and macOS hosts, clusters and rentals alike."""

    def stage(self, plan: ExecutionPlan, root: str, *, command: str, activate: bool) -> str:
        return wrap(plan, root, command=command, activate=activate)

    def session(self, host: str, line: str) -> list[str]:
        # `-t` forces the pty the far side needs, and the staged line is quoted whole because
        # ssh joins its argv back into one string for the remote login shell to parse.
        return ["ssh", "-t", host, f"bash -lc {shlex.quote(line)}"]

    def has(self, tool: str) -> str:
        return f"command -v {tool}"

    def is_directory(self, path: str) -> str:
        return f"[ -d {shlex.quote(path)} ]"

    def is_file(self, path: str) -> str:
        return f"test -f {shlex.quote(path)}"

    def chain(self, *commands: str) -> str:
        return " && ".join(commands)

    @property
    def noop(self) -> str:
        return "true"

    @property
    def uv_bootstrap(self) -> tuple[str, str]:
        return "command -v curl", _UV_INSTALLER

    @property
    def pip(self) -> tuple[str, str]:
        return "python3 -m pip --version", "python3 -m pip install --user --break-system-packages"

    @property
    def pixi_installer(self) -> str:
        return POSIX_INSTALLER

    def proof(self, plan: ExecutionPlan, root: str) -> str:
        return activation(root, env=plan.env)

    def provisioned(self, plan: ExecutionPlan, root: str) -> str:
        return self.is_file(self.proof(plan, root))

    def activation_record(self, plan: ExecutionPlan, root: str) -> str:
        return self.proof(plan, root)


class Windows(Dialect):
    """The PowerShell family: a Windows host whose ssh login shell is cmd.exe.

    An activated command is handed to the host's own tool, `mainboard run --env <env> -- ...`,
    because a Windows workspace activates through the activation pixi cached when it was
    provisioned and nothing else on the machine can source it.
    """

    def stage(self, plan: ExecutionPlan, root: str, *, command: str, activate: bool) -> str:
        # Progress records would otherwise reach stderr as CLIXML noise, and a script made only of
        # cmdlets leaves no exit code behind, so the code is cleared before the command sets it.
        steps = [
            "$ProgressPreference = 'SilentlyContinue'",
            "$LASTEXITCODE = 0",
            f"Set-Location -LiteralPath {quoted(root)} -ErrorAction Stop",
            f'$env:Path = "{";".join(_WINDOWS_BINS)};" + $env:Path',
        ]
        if activate:
            steps += [f"$env:{key} = {quoted(value)}" for key, value in plan.exports.items()]
            # Windows PowerShell's legacy native-argument binder strips embedded quotes.
            # Give CreateProcess the CRT-encoded argument vector directly instead. Data such
            # as Python source, empty strings, percent signs and trailing slashes stays data.
            arguments = subprocess.list2cmdline(
                ["run", "--env", plan.env, "--", *shlex.split(command)]
            )
            command = "; ".join(
                [
                    "$start = New-Object System.Diagnostics.ProcessStartInfo",
                    f"$start.FileName = {quoted(_TOOL)}",
                    f"$start.WorkingDirectory = {quoted(root)}",
                    f"$start.Arguments = {quoted(arguments)}",
                    "$start.UseShellExecute = $false",
                    "$process = [System.Diagnostics.Process]::Start($start)",
                    "$process.WaitForExit()",
                    "$LASTEXITCODE = $process.ExitCode",
                ]
            )
        return "; ".join([*steps, command, "exit $LASTEXITCODE"])

    def session(self, host: str, line: str) -> list[str]:
        return ["ssh", "-t", host, *POWERSHELL[:1], "-NoProfile", "-EncodedCommand", encoded(line)]

    def has(self, tool: str) -> str:
        return _exit_on(f"Get-Command {tool} -ErrorAction SilentlyContinue")

    def is_directory(self, path: str) -> str:
        return _exit_on(f"Test-Path -LiteralPath {quoted(path)} -PathType Container")

    def is_file(self, path: str) -> str:
        return _exit_on(f"Test-Path -LiteralPath {quoted(path)} -PathType Leaf")

    def chain(self, *commands: str) -> str:
        return "; ".join(commands)

    @property
    def noop(self) -> str:
        return "exit 0"

    @property
    def uv_bootstrap(self) -> tuple[str, str]:
        # PowerShell fetches on its own, so this route needs nothing the host could lack.
        return "exit 0", "irm https://astral.sh/uv/install.ps1 | iex"

    @property
    def pip(self) -> tuple[str, str]:
        return "python -m pip --version", "python -m pip install --user --break-system-packages"

    @property
    def pixi_installer(self) -> str:
        # The installer's own inner probes leave an exit code behind that says nothing about
        # the install; the version read right after it is what vouches for the pixi it put down.
        return f"{WINDOWS_INSTALLER}; $LASTEXITCODE = 0"

    def proof(self, plan: ExecutionPlan, root: str) -> str:
        return plan.prefix(root)

    def provisioned(self, plan: ExecutionPlan, root: str) -> str:
        return self.is_directory(self.proof(plan, root))

    def activation_record(self, plan: ExecutionPlan, root: str) -> str:
        return ""


def dialect_for(profile: HostProfile) -> Dialect:
    """The dialect `profile`'s platform speaks."""
    return Windows() if is_windows(profile) else Posix()


class HostShell(ABC):
    """A host's shell staged by an execution plan, the one way a remote command is run.

    Two footings: a bare command gets `cd`, the per-user install dirs on `PATH` and the host's
    modules, all an unprovisioned machine can offer, while an activated one additionally enters
    the environment, which is what proves the environment an install just built actually runs.

    plan: the resolved execution context staging commands.
    root: the workspace root on the host.
    """

    dialect: Dialect

    def __init__(self, plan: ExecutionPlan, root: str) -> None:
        self.plan = plan
        self.root = root

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @abstractmethod
    def close(self) -> None:
        """Release whatever connection the shell held."""

    def ok(self, command: str, *, activate: bool = False) -> bool:
        """Whether `command` exits zero on the host, its output discarded."""
        retcode, _, _ = self.execute(self.stage(command, activate=activate))
        return retcode == 0

    def run(self, command: str, *, activate: bool = False) -> str:
        """`command`'s stdout on the host, raising a `MissionError` naming why it failed.

        command: the command to run in the workspace.
        activate: run it through the plan's activation rather than the bare staging.
        """
        retcode, out, err = self.execute(self.stage(command, activate=activate))
        if retcode:
            reason = failure_reason(err or out, retcode)
            raise MissionError(f"`{command}` failed on {self.plan.host!r}: {reason}")
        return out

    def stage(self, command: str, *, activate: bool) -> str:
        """`command` staged for this host, the dialect's own line."""
        return self.dialect.stage(self.plan, self.root, command=command, activate=activate)

    @property
    def proof(self) -> str:
        """The path that proves the plan's environment was provisioned here."""
        return self.dialect.proof(self.plan, self.root)

    @property
    def provisioned(self) -> str:
        """The probe exiting zero once the plan's environment exists here."""
        return self.dialect.provisioned(self.plan, self.root)

    @property
    def activation_record(self) -> str:
        """The activation script a `HostSetup` records for this host."""
        return self.dialect.activation_record(self.plan, self.root)

    @abstractmethod
    def execute(self, line: str) -> tuple[int, str, str]:
        """Run one already staged `line` and answer its exit status, stdout and stderr."""

    @abstractmethod
    def foreground(self, command: str, *, activate: bool = True) -> int:
        """Run `command` with this terminal's stdio and answer its exit status."""

    @abstractmethod
    def write(self, path: str, text: str) -> None:
        """Write `text` to `path` on the host, private to the user, its directory made."""


class PosixShell(HostShell):
    """A POSIX host reached through plumbum's persistent ssh session.

    remote: the open connection commands ride.
    plan: the resolved execution context staging them.
    root: the workspace root on the host.
    """

    dialect = Posix()

    def __init__(self, remote: Machine, plan: ExecutionPlan, root: str) -> None:
        super().__init__(plan, root)
        self.remote = remote

    def close(self) -> None:
        if isinstance(self.remote, BoundedSshMachine):
            self.remote.close()

    def execute(self, line: str) -> tuple[int, str, str]:
        retcode, out, err = self.remote["bash"][["-lc", line]].run(retcode=None)
        return int(retcode), str(out), str(err)

    def foreground(self, command: str, *, activate: bool = True) -> int:
        return foreground(self.remote["bash"]["-lc", self.stage(command, activate=activate)])

    def write(self, path: str, text: str) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        written = f"umask 077; mkdir -p {parent}; cat > {shlex.quote(path)}"
        (self.remote["bash"]["-c", written] << text)()


class WindowsShell(HostShell):
    """A Windows host reached one PowerShell script at a time over the bounded transport.

    plan: the resolved execution context staging commands.
    root: the workspace root on the host.
    ssh: the bounded SSH policy; the default policy when omitted.
    """

    dialect = Windows()

    def __init__(self, plan: ExecutionPlan, root: str, *, ssh: SshTransport | None = None) -> None:
        super().__init__(plan, root)
        self.ssh = ssh or SshTransport()

    def close(self) -> None:
        """Nothing to release: every script rode its own ssh process."""
        return

    def argv(self, script: str) -> tuple[str, ...]:
        """The ssh argv running `script` through PowerShell on the host."""
        destination = self.ssh.destination(self.plan.host)
        return ("ssh", *self.ssh.options, destination, *POWERSHELL, encoded(script))

    def execute(self, line: str) -> tuple[int, str, str]:
        retcode, out, err = self.ssh.invoke(
            self.argv(line), self.plan.host, operation="run", bounded=False
        )
        return retcode, out, plain_errors(err)

    def foreground(self, command: str, *, activate: bool = True) -> int:
        return subprocess.call(self.argv(self.stage(command, activate=activate)))  # ruff:ignore[subprocess-without-shell-equals-true]  reason=ssh argv built from typed fields, not untrusted input since=2026-09-11

    def write(self, path: str, text: str) -> None:
        parent = quoted(str(PurePosixPath(path).parent))
        self.run(
            f"New-Item -ItemType Directory -Force -Path {parent} | Out-Null; "
            f"Set-Content -LiteralPath {quoted(path)} -Value {quoted(text)} -NoNewline"
        )


def open_shell(plan: ExecutionPlan, root: str, *, ssh: SshTransport | None = None) -> HostShell:
    """The shell `plan`'s host answers, connected: PowerShell one-shots or a login-bash session.

    plan: the resolved execution context, its profile's platform deciding the family.
    root: the workspace root on the host.
    ssh: the bounded SSH policy; the default policy when omitted.
    """
    if is_windows(plan.profile):
        return WindowsShell(plan, root, ssh=ssh)
    return PosixShell(connection(plan.host, ssh), plan, root)
