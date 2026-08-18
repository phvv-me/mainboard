import signal
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=monkeypatches Popen for hermetic tests, never runs a real process since=2026-08-18

import pytest

from mainboard.dispatch import DaemonDown, HostUnreachable, SshTransport
from mainboard.dispatch import transport as transport_module
from mainboard.dispatch.transport import is_daemon_failure, is_transport_failure

# Mirrors transport.py's own private marker vocabulary; kept as a literal here rather than
# imported so each new marker demands a deliberate new test case, not silent inherited coverage.
_TRANSPORT_MARKERS = (
    "session open refused",
    "connection refused",
    "connection closed",
    "connection timed out",
    "operation timed out",
    "broken pipe",
    "no route to host",
    "kex_exchange",
    "control socket",
    "control master",
    "timed out",
)
_DAEMON_DOWN_MARKERS = ("connecting to the daemon", "connection refused", ".socket")


class _FakeProcess:
    """A `subprocess.Popen` stand-in whose `communicate`/`wait` are scripted per test."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        raise_timeout: bool = False,
    ) -> None:
        self.pid = 4242
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._raise_timeout = raise_timeout
        self.communicate_calls = 0
        self.wait_calls: list[float | None] = []

    def communicate(self, timeout: float) -> tuple[str, str]:
        self.communicate_calls += 1
        if self._raise_timeout and self.communicate_calls == 1:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return self._stdout, self._stderr

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)


def test_is_transport_failure_requires_255_and_a_marker() -> None:
    assert is_transport_failure(255, "kex_exchange identification: read: Connection reset") is True
    assert is_transport_failure(255, "some unrelated message") is False
    assert is_transport_failure(1, "connection refused") is False


@pytest.mark.parametrize("marker", _TRANSPORT_MARKERS)
def test_every_transport_marker_is_recognized(marker: str) -> None:
    assert is_transport_failure(255, f"ssh: {marker} happened")


def test_is_daemon_failure_matches_any_marker() -> None:
    assert is_daemon_failure("Error connecting to the daemon") is True
    assert is_daemon_failure("no such file: pueue.socket") is True
    assert is_daemon_failure("totally unrelated") is False


@pytest.mark.parametrize("marker", _DAEMON_DOWN_MARKERS)
def test_every_daemon_marker_is_recognized(marker: str) -> None:
    assert is_daemon_failure(f"client error: {marker}")


def test_daemon_down_is_a_host_unreachable_subclass() -> None:
    assert issubclass(DaemonDown, HostUnreachable)


def test_ssh_transport_options_carry_the_liveness_overrides() -> None:
    policy = SshTransport(connect_timeout=5.0, server_alive_interval=3.0, server_alive_count=2)
    assert policy.options == (
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=3",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "BatchMode=yes",
    )
    assert policy.deadline == pytest.approx(5.0 + 3.0 * 2 + 5.0)


def test_rsync_shell_joins_ssh_and_its_options() -> None:
    policy = SshTransport()
    assert policy.rsync_shell.startswith("ssh -o ConnectTimeout=")


def test_run_returns_stdout_on_a_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SshTransport()
    process = _FakeProcess(returncode=0, stdout="ok\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    assert policy.run(("ssh", "host", "true"), "host", operation="connect") == "ok\n"


def test_run_raises_host_unreachable_when_popen_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = SshTransport()

    def boom(*_a, **_k) -> None:
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(HostUnreachable, match="could not start"):
        policy.run(("ssh", "host", "true"), "host", operation="connect")


def test_run_raises_host_unreachable_on_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SshTransport()
    process = _FakeProcess(returncode=255, stdout="", stderr="kex_exchange identification failed")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(HostUnreachable, match="kex_exchange"):
        policy.run(("ssh", "host", "true"), "host", operation="connect")


def test_run_raises_connection_error_on_host_key_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SshTransport()
    process = _FakeProcess(returncode=255, stderr="Host key verification failed.")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(ConnectionError, match="host-key verification"):
        policy.run(("ssh", "host", "true"), "host", operation="connect")


def test_run_raises_runtime_error_on_a_genuine_non_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = SshTransport()
    process = _FakeProcess(returncode=1, stderr="remote command exploded")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(RuntimeError, match="remote command exploded"):
        policy.run(("ssh", "host", "true"), "host", operation="connect")


def test_run_reports_exit_code_when_stderr_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SshTransport()
    process = _FakeProcess(returncode=7, stderr="")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(RuntimeError, match="exit 7"):
        policy.run(("ssh", "host", "true"), "host", operation="connect")


def test_run_terminates_the_process_group_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SshTransport()
    process = _FakeProcess(raise_timeout=True)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(HostUnreachable, match="timed out"):
        policy.run(("ssh", "host", "true"), "host", operation="connect")
    assert killed == [(process.pid, __import__("signal").SIGTERM)]
    assert process.wait_calls == [2.0]


def test_terminate_escalates_to_sigkill_when_sigterm_does_not_land(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()

    def wait(timeout: float | None = None) -> None:
        process.wait_calls.append(timeout)
        if len(process.wait_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    process.wait = wait  # type: ignore[method-assign]  reason=test double stands in for the bound method since=2026-08-16
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    SshTransport.terminate(process)  # type: ignore[arg-type]  reason=test double stands in for the process handle since=2026-08-16

    assert killed == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert process.wait_calls == [2.0, None]


def test_terminate_tolerates_a_process_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _FakeProcess()

    def raise_lookup(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("os.killpg", raise_lookup)
    SshTransport.terminate(process)  # type: ignore[arg-type]  reason=test double stands in for the process handle since=2026-08-16
    assert process.wait_calls == [2.0]


def test_warm_and_copy_call_run_with_the_right_ssh_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = SshTransport()
    calls: list[tuple[tuple[str, ...], str, str]] = []
    monkeypatch.setattr(
        SshTransport,
        "run",
        lambda self, command, host, *, operation: calls.append((command, host, operation)),
    )
    policy.warm("gold")
    policy.copy("a.txt", destination="gold:b.txt", host="gold")
    assert calls[0][0][:2] == ("ssh", "-o")
    assert calls[0][1:] == ("gold", "connect")
    assert calls[1][0][:2] == ("scp", "-o")
    assert calls[1][1:] == ("gold", "copy")


def test_machine_builds_a_bounded_ssh_machine(monkeypatch: pytest.MonkeyPatch) -> None:

    built: dict[str, str | tuple[str, ...] | float | bool] = {}

    class FakeBoundedSshMachine:
        def __init__(
            self,
            host: str,
            *,
            ssh_opts: tuple[str, ...],
            connect_timeout: float,
            new_session: bool,
        ) -> None:
            built.update(
                host=host,
                ssh_opts=ssh_opts,
                connect_timeout=connect_timeout,
                new_session=new_session,
            )

    monkeypatch.setattr(transport_module, "BoundedSshMachine", FakeBoundedSshMachine)
    policy = SshTransport()
    policy.machine("gold")
    assert built["host"] == "gold"
    assert built["new_session"] is True


class _FakeSshProcess:
    """A minimal stand-in for the process a `ShellSession` wraps: `pid` and `poll()`."""

    def __init__(self, *, alive: bool) -> None:
        self.pid = 999
        self._alive = alive

    def poll(self) -> int | None:
        return None if self._alive else 0


def test_bounded_shell_session_close_kills_a_live_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(transport_module.BoundedShellSession)
    session.proc = _FakeSshProcess(alive=True)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    closed: list[bool] = []
    monkeypatch.setattr(transport_module.ShellSession, "close", lambda self: closed.append(True))
    session.close()
    assert killed == [(999, signal.SIGTERM)]
    assert closed == [True]


def test_bounded_shell_session_close_skips_an_already_exited_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(transport_module.BoundedShellSession)
    session.proc = _FakeSshProcess(alive=False)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(transport_module.ShellSession, "close", lambda self: None)
    session.close()
    assert killed == []


def test_bounded_shell_session_close_tolerates_a_process_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(transport_module.BoundedShellSession)
    session.proc = _FakeSshProcess(alive=True)

    def raise_lookup(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("os.killpg", raise_lookup)
    monkeypatch.setattr(transport_module.ShellSession, "close", lambda self: None)
    session.close()  # must not raise


def test_bounded_shell_session_close_when_proc_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    session = object.__new__(transport_module.BoundedShellSession)
    session.proc = None
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(transport_module.ShellSession, "close", lambda self: None)
    session.close()
    assert killed == []


def test_bounded_ssh_machine_session_builds_a_bounded_shell_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = object.__new__(transport_module.BoundedSshMachine)
    machine.custom_encoding = "utf-8"
    machine.connect_timeout = 15.0
    machine.host = "gold"
    popen_calls: list[tuple[list[str], list[str], bool]] = []
    machine.popen = lambda argv, extra, new_session: (
        popen_calls.append((argv, extra, new_session)) or "PROC"
    )
    built: dict[str, str | bool | float] = {}

    class FakeSession:
        def __init__(
            self, proc: str, encoding: str, isatty: bool, connect_timeout: float, *, host: str
        ) -> None:
            built.update(
                proc=proc,
                encoding=encoding,
                isatty=isatty,
                connect_timeout=connect_timeout,
                host=host,
            )

    monkeypatch.setattr(transport_module, "BoundedShellSession", FakeSession)
    result = machine.session(isatty=True, new_session=True)
    assert isinstance(result, FakeSession)
    assert popen_calls == [(["/bin/sh"], ["-tt"], True)]
    assert built["proc"] == "PROC"
    assert built["host"] == "gold"


def test_bounded_ssh_machine_session_uses_non_interactive_flags_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = object.__new__(transport_module.BoundedSshMachine)
    machine.custom_encoding = "utf-8"
    machine.connect_timeout = 15.0
    machine.host = "gold"
    popen_calls: list[tuple[list[str], list[str], bool]] = []
    machine.popen = lambda argv, extra, new_session: (
        popen_calls.append((argv, extra, new_session)) or "PROC"
    )
    monkeypatch.setattr(transport_module, "BoundedShellSession", lambda *a, **k: "SESSION")
    machine.session()
    assert popen_calls == [(["/bin/sh"], ["-T"], False)]
