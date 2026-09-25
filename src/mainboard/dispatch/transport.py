# Bounded SSH transport policy, the machines it reaches, and its shared failure vocabulary. A
# transport fault is the ssh link itself failing; it reads identically to a real failure (exit
# 255 with a stderr phrase).
#
# A policy carries an `Endpoint` when the machine it opens is not in `~/.ssh/config` at all,
# which is the whole difference between a declared host and one rented for a single job.

import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=argv built from typed fields (ssh/scp options), not untrusted input since=2026-08-17
from contextlib import ExitStack, suppress
from math import ceil
from pathlib import Path
from typing import IO, NoReturn

import psutil
from patos import FrozenModel
from plumbum.machines.local import LocalMachine
from plumbum.machines.session import ShellSession
from plumbum.machines.ssh_machine import SshMachine
from pydantic import Field, field_validator

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


def terminate_process_tree(pid: int, *, force: bool = False) -> None:
    """Terminate one process tree with the operating system's native primitive.

    psutil supplies one process-tree traversal on every platform, including ProxyJump children,
    without maintaining separate POSIX-signal and Windows process APIs.

    pid: root process identifier.
    force: kill rather than request graceful termination.
    """
    root = psutil.Process(pid)
    processes = [*reversed(root.children(recursive=True)), root]
    for process in processes:
        with suppress(psutil.Error):
            (process.kill if force else process.terminate)()


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


class Endpoint(FrozenModel):
    """Where ssh reaches one machine `~/.ssh/config` has never heard of.

    A declared host is an alias and the user's own config answers every question about it. A
    machine rented for one job has no alias and no entry: it is an address, a port, a login and
    a key, all four minted minutes ago, so they ride with the policy that opens the connection
    instead of with a file nobody edited. Its host key is new by construction and will never be
    seen again, so it is accepted on sight and kept out of `known_hosts`, which is also what
    stops a recycled provider address from failing verification against the key some earlier
    rental had at it.

    address: the hostname or IP ssh connects to.
    port: the ssh port, 0 for ssh's own default.
    user: the login, empty for whatever ssh would choose.
    identity: the private key file, empty to leave that to ssh's agent and config.
    """

    address: str
    port: int = 0
    user: str = ""
    identity: str = ""

    @field_validator("identity")
    @classmethod
    def expanded(cls, value: str) -> str:
        """A key path with `~` resolved, since ssh receives it as one argument with no shell."""
        return str(Path(value).expanduser()) if value else value

    @property
    def destination(self) -> str:
        """The `user@address` every ssh and scp command names this machine by."""
        return f"{self.user}@{self.address}" if self.user else self.address

    @property
    def options(self) -> tuple[str, ...]:
        """The ssh options this machine needs beyond the liveness policy."""
        port = ("-p", str(self.port)) if self.port else ()
        key = ("-i", self.identity, "-o", "IdentitiesOnly=yes") if self.identity else ()
        return (
            *port,
            *key,
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={os.devnull}",
            "-o",
            "LogLevel=ERROR",
        )

    @property
    def scp_options(self) -> tuple[str, ...]:
        """`options` as scp spells them, which differs in one letter: its port flag is `-P`."""
        return tuple("-P" if option == "-p" else option for option in self.options)


class SshTransport(FrozenModel):
    """One bounded OpenSSH policy while preserving user aliases and ProxyJump settings.

    An `endpoint` binds the policy to a machine that has no alias to preserve, which is what
    lets a rented box ride the same mirror, install and pin path a declared host does.
    """

    connect_timeout: float = Field(default=15.0, gt=0.0)
    server_alive_interval: float = Field(default=15.0, gt=0.0)
    server_alive_count: int = Field(default=3, ge=1)
    batch_mode: bool = True
    endpoint: Endpoint | None = None

    @property
    def deadline(self) -> float:
        """The worst-case liveness window for a control operation."""
        return self.connect_timeout + self.server_alive_interval * self.server_alive_count + 5.0

    @property
    def stream_deadline(self) -> float:
        """A finite wall bound for bulk evidence, separate from SSH's liveness probes."""
        return max(self.deadline, 600.0)

    @property
    def liveness(self) -> tuple[str, ...]:
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
    def options(self) -> tuple[str, ...]:
        """The liveness overrides plus whatever the bound machine needs to be reached at all."""
        return (*self.liveness, *(self.endpoint.options if self.endpoint else ()))

    def destination(self, host: str) -> str:
        """`host` as ssh must spell it: the bound machine when there is one, else the alias."""
        return self.endpoint.destination if self.endpoint else host

    @staticmethod
    def terminate(process: subprocess.Popen[bytes]) -> None:
        """Terminate the whole SSH process group so ProxyJump children cannot remain."""
        with suppress(ProcessLookupError, PermissionError, psutil.Error):
            terminate_process_tree(process.pid)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            SshTransport.__force_killpg(process)

    def transfer(self, source: str, *, destination: str, host: str) -> None:
        """Copy one file through the bounded SSH policy."""
        scp = (*self.liveness, *(self.endpoint.scp_options if self.endpoint else ()))
        self.run(("scp", *scp, source, destination), host, operation="copy")

    def machine(self, host: str) -> BoundedSshMachine:
        """A persistent SSH session with a dedicated local process group."""
        return BoundedSshMachine(
            host,
            ssh_opts=self.options,
            connect_timeout=self.deadline,
            new_session=True,
        )

    def invoke(
        self,
        command: tuple[str, ...],
        host: str,
        *,
        operation: str,
        input_text: str | None = None,
        bounded: bool = True,
    ) -> tuple[int, str, str]:
        """Run one ssh process and answer its exit status with what it wrote.

        A transport fault or a host-key failure is raised as itself, since neither is an answer;
        any other exit status comes back, because a probe that exits non-zero answered.

        command: the full argv, `ssh` first.
        host: the alias or destination, named in every failure.
        operation: what the command is for, named in every failure.
        input_text: explicit UTF-8 input, otherwise the native null device.
        bounded: hold the process to the control deadline; False lets an install run its course.
        """
        returncode, stdout, stderr = self.__communicate(
            command,
            host,
            operation=operation,
            input_text=input_text,
            sink=subprocess.PIPE,
            timeout=self.deadline if bounded else None,
        )
        self.__check(returncode, stderr, host=host, operation=operation)
        return returncode, stdout or "", stderr or ""

    def run(
        self,
        command: tuple[str, ...],
        host: str,
        *,
        operation: str,
        input_text: str | None = None,
        output: Path | None = None,
    ) -> str:
        """Run one SSH transfer in a killable process group and surface a typed failure.

        input_text: explicit UTF-8 input, otherwise stdin is the native null device. Never
            inherit the caller's input, which an SSH warm-up could consume from a shell loop.
        output: caller-owned staging file for raw stdout bytes, otherwise capture text. A
            streamed operation returns an empty string and retains partial bytes on failure.
            The caller owns validation and publication of the completed staging file.
            Bulk streams allow ten minutes; control calls retain the shorter deadline.
            SSH keepalives and process-tree termination apply to both paths.
        """
        with ExitStack() as stack:
            sink = (
                stack.enter_context(output.open("wb")) if output is not None else subprocess.PIPE
            )
            returncode, stdout, stderr = self.__communicate(
                command,
                host,
                operation=operation,
                input_text=input_text,
                sink=sink,
                timeout=self.stream_deadline if output is not None else self.deadline,
            )
        if returncode == 0:
            return stdout or ""
        self.__check(returncode, stderr, host=host, operation=operation)
        raise RuntimeError(f"ssh {operation} to {host!r} failed: {_detail(stderr, returncode)}")

    def __communicate(
        self,
        command: tuple[str, ...],
        host: str,
        *,
        operation: str,
        input_text: str | None,
        sink: int | IO[bytes],
        timeout: float | None,
    ) -> tuple[int, str, str]:
        """Run `command` in its own process group and return its status, stdout and stderr."""
        try:
            process = subprocess.Popen(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=ssh/scp argv built from typed fields, not untrusted input since=2026-08-16
                command,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            raise HostUnreachable(
                f"ssh {operation} to {host!r} could not start: {error}"
            ) from error
        # Bytes both ways, because a text-mode pipe on Windows writes every `\n` of the input as
        # `\r\n`, and the bash reading it on the host takes that `\r` as part of each command.
        sent = None if input_text is None else input_text.encode("utf-8")
        try:
            stdout, stderr = process.communicate(input=sent, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self.__raise_after_terminating(process, host=host, operation=operation, cause=error)
        return process.returncode, _decoded(stdout), _decoded(stderr)

    @staticmethod
    def __check(returncode: int, stderr: str, *, host: str, operation: str) -> None:
        """Raise the typed failure `(returncode, stderr)` names, if it names one."""
        if "host key verification failed" in stderr.lower():
            raise ConnectionError(f"ssh to {host!r} failed host-key verification")
        # The connect probe runs only `echo`, which never exits 255, so a 255 there is ssh's own
        # failure even when it printed nothing a marker names. Left untyped it escaped the
        # rental's knock loop as a bare error (RTX 4090 52521884, 2026-09-25).
        silent = operation == "connect" and returncode == _SSH_TRANSPORT_RC
        if silent or is_transport_failure(returncode, stderr):
            raise HostUnreachable(
                f"ssh {operation} to {host!r} failed: {_detail(stderr, returncode)}"
            )

    def warm(self, host: str) -> None:
        """Validate one bounded SSH connection before Plumbum opens its persistent session."""
        marker = "mainboard-reachable"
        output = self.run(("ssh", *self.options, host, "echo", marker), host, operation="connect")
        if marker not in output.splitlines():
            raise RuntimeError(f"ssh connect to {host!r} returned without the expected marker")

    @staticmethod
    def __force_killpg(process: subprocess.Popen[bytes]) -> None:
        """Escalate to SIGKILL after a SIGTERM'd process group failed to exit in time."""
        with suppress(ProcessLookupError, PermissionError, psutil.Error):
            terminate_process_tree(process.pid, force=True)
        process.wait()

    def __raise_after_terminating(
        self,
        process: subprocess.Popen[bytes],
        *,
        host: str,
        operation: str,
        cause: subprocess.TimeoutExpired,
    ) -> NoReturn:
        """Kill `process`'s group, then translate its `communicate()` timeout for the caller."""
        self.terminate(process)
        raise HostUnreachable(
            f"ssh {operation} to {host!r} timed out after {cause.timeout:g}s"
        ) from cause


def _decoded(output: bytes | None) -> str:
    """Captured bytes as a text-mode pipe reads them: UTF-8, with universal newlines."""
    text = (output or b"").decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _detail(stderr: str, returncode: int) -> str:
    """The last thing ssh said, or its exit status when it said nothing."""
    return stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {returncode}"


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
        if process is not None and process.pid is not None and process.poll() is None:
            with suppress(ProcessLookupError, PermissionError, psutil.Error):
                terminate_process_tree(process.pid)
        super().close()


class BoundedSshMachine(SshMachine):
    """An SSH machine whose session owns its entire local transport group."""

    def session(self, isatty: bool = False, new_session: bool = False) -> ShellSession:
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
