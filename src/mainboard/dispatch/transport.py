# Bounded SSH transport policy and its shared failure vocabulary. A transport fault is the ssh
# link itself failing; it reads identically to a real failure (exit 255 with a stderr phrase).

import os
import shlex
import signal
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=argv built from typed fields (ssh/scp/rsync options), not untrusted input since=2026-08-17
from contextlib import suppress
from math import ceil
from typing import NoReturn

from patos import FrozenModel
from plumbum.machines.local import LocalMachine
from plumbum.machines.session import ShellSession
from plumbum.machines.ssh_machine import SshMachine
from pydantic import Field

# ssh's own exit status when the transport fails, with the stderr phrases naming the fault. A
# name that does not resolve belongs here too: the host cannot be reached right now (a dropped
# VPN, a DNS outage, an alias that lost its record), which a poll retries and a durable sweep
# reports as one down host, rather than a command on it having genuinely failed.
_SSH_TRANSPORT_RC = 255
_TRANSPORT_MARKERS = (
    "session open refused",
    "connection refused",
    "connection closed",
    "connection timed out",
    "operation timed out",
    "broken pipe",
    "no route to host",
    "could not resolve hostname",
    "name or service not known",
    "kex_exchange",
    "control socket",
    "control master",
    "timed out",
    # An ssh that would not authenticate never opened a session, so no command ran and there is
    # nothing to read an answer out of. Treating it as a real command answer is what let a
    # `Permission denied (keyboard-interactive)` on an expired credential raise a bare
    # `RuntimeError` out of `Job.transcript`, whose own contract says a host that went quiet
    # costs its transcript and no other job's outcome, and take a whole monitor sweep down.
    "permission denied",
    "too many authentication failures",
)


# A dead scheduler daemon (pueue's `pueued`) refuses its own control socket, distinct from an
# ssh transport fault, so it surfaces as `daemon down` and a revive restarts it.
_DAEMON_DOWN_MARKERS = ("connecting to the daemon", "connection refused", ".socket")


class HostUnreachable(Exception):
    """An ssh transport failure, so a host's state is unknown right now rather than settled.

    Raised when the ssh connection itself failed (a refused control-master session, a dropped
    link, a timeout) rather than the remote command running and exiting non-zero. Wait and
    connect loops absorb a few of these with backoff, so a transient blip is never misread as a
    finished or vanished job, nor as a host that cannot be reached at all; a persistent outage
    still surfaces once the retry budget is spent.
    """


class DaemonDown(HostUnreachable):
    """A host's scheduler daemon is down (a dead pueue `pueued`), so its jobs cannot resolve now.

    A subclass of `HostUnreachable`, so every wait/poll/status path that already rides out an
    unreachable host treats a dead daemon the same way rather than crashing on the raw client
    error. The reason it carries, `daemon down`, is what a durable monitor surfaces per host, and
    reviving the host restarts the daemon to recover.
    """


class SshTransport(FrozenModel):
    """One bounded OpenSSH policy while preserving user aliases and ProxyJump settings."""

    connect_timeout: float = Field(default=15.0, gt=0.0)
    server_alive_interval: float = Field(default=15.0, gt=0.0)
    server_alive_count: int = Field(default=3, ge=1)
    batch_mode: bool = True

    @property
    def deadline(self) -> float:
        """The worst-case liveness window for a control operation."""
        return self.connect_timeout + self.server_alive_interval * self.server_alive_count + 5.0

    @property
    def options(self) -> tuple[str, ...]:
        """Only the liveness overrides, leaving every alias setting intact."""
        return (
            "-o",
            f"ConnectTimeout={ceil(self.connect_timeout)}",
            "-o",
            f"ServerAliveInterval={ceil(self.server_alive_interval)}",
            "-o",
            f"ServerAliveCountMax={self.server_alive_count}",
            "-o",
            f"BatchMode={'yes' if self.batch_mode else 'no'}",
        )

    @property
    def rsync_shell(self) -> str:
        """rsync's remote shell under this same SSH policy."""
        return shlex.join(("ssh", *self.options))

    @staticmethod
    def terminate(process: subprocess.Popen[str]) -> None:
        """Terminate the whole SSH process group so ProxyJump children cannot remain."""
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            SshTransport.__force_killpg(process)

    def copy(self, source: str, *, destination: str, host: str) -> None:
        """Copy one file through the bounded SSH policy."""
        self.run(("scp", *self.options, source, destination), host, operation="copy")

    def machine(self, host: str) -> BoundedSshMachine:
        """A persistent SSH session with a dedicated local process group."""
        return BoundedSshMachine(
            host,
            ssh_opts=self.options,
            connect_timeout=self.deadline,
            new_session=True,
        )

    def run(self, command: tuple[str, ...], host: str, *, operation: str) -> str:
        """Run one SSH transfer in a killable process group and surface a typed failure.

        Its stdin is `/dev/null`, which is `ssh -n` spelled where every ssh and scp this policy
        runs inherits it. An ssh client left on the caller's own stdin reads it greedily to
        forward to the far side, and every verb here opens a connection before it does anything,
        so `while read handle; do mainboard submit ...; done < handles` lost the rest of the
        file to the first submit's connection warm-up. Nothing this policy runs wants a caller's
        input: the warm-up runs `true`, scp moves a file, and a real remote command rides
        plumbum's own session, whose stdin is a pipe it writes the command into.
        """
        try:
            process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=ssh/scp argv built from typed fields, not untrusted input since=2026-08-16
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            raise HostUnreachable(
                f"ssh {operation} to {host!r} could not start: {error}"
            ) from error
        try:
            stdout, stderr = process.communicate(timeout=self.deadline)
        except subprocess.TimeoutExpired as error:
            self.__raise_after_terminating(process, host=host, operation=operation, cause=error)
        if process.returncode == 0:
            return stdout
        if "host key verification failed" in stderr.lower():
            raise ConnectionError(f"ssh to {host!r} failed host-key verification")
        detail = (
            stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {process.returncode}"
        )
        if is_transport_failure(process.returncode, stderr):
            raise HostUnreachable(f"ssh {operation} to {host!r} failed: {detail}")
        raise RuntimeError(f"ssh {operation} to {host!r} failed: {detail}")

    def warm(self, host: str) -> None:
        """Validate one bounded SSH connection before Plumbum opens its persistent session."""
        self.run(("ssh", *self.options, host, "true"), host, operation="connect")

    @staticmethod
    def __force_killpg(process: subprocess.Popen[str]) -> None:
        """Escalate to SIGKILL after a SIGTERM'd process group failed to exit in time."""
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    def __raise_after_terminating(
        self, process: subprocess.Popen[str], *, host: str, operation: str, cause: Exception
    ) -> NoReturn:
        """Kill `process`'s group, then translate its `communicate()` timeout for the caller."""
        self.terminate(process)
        raise HostUnreachable(
            f"ssh {operation} to {host!r} timed out after {self.deadline:g}s"
        ) from cause


def is_transport_failure(retcode: int, stderr: str) -> bool:
    """Whether `(retcode, stderr)` is an ssh transport fault, not a real command answer."""
    low = stderr.lower()
    return retcode == _SSH_TRANSPORT_RC and any(marker in low for marker in _TRANSPORT_MARKERS)


def is_daemon_failure(stderr: str) -> bool:
    """Whether a scheduler client's `stderr` names a dead daemon (a refused control socket)."""
    low = stderr.lower()
    return any(marker in low for marker in _DAEMON_DOWN_MARKERS)


class BoundedShellSession(ShellSession):
    """Close the dedicated SSH process group, including ProxyJump children."""

    def close(self) -> None:
        process = self.proc
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGTERM)
        super().close()


class BoundedSshMachine(SshMachine):
    """An SSH machine whose session owns its entire local transport group."""

    def session(self, isatty: bool = False, *, new_session: bool = False) -> ShellSession:
        return BoundedShellSession(
            self.popen(["/bin/sh"], (["-tt"] if isatty else ["-T"]), new_session=new_session),
            self.custom_encoding,
            isatty,
            self.connect_timeout,
            host=self.host,
        )


# A plumbum machine the dispatch subsystem runs commands on, as `machine["cmd"][args]`:
# `local` (the default) or an `SshMachine` for a remote host.
type Machine = LocalMachine | BoundedSshMachine
