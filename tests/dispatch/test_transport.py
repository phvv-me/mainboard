import signal
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=monkeypatches Popen for hermetic tests, never runs a real process since=2026-08-18

import pytest

from mainboard.dispatch import DaemonDown, HostUnreachable, SshTransport
from mainboard.dispatch import transport as transport_module
from mainboard.dispatch.transport import is_daemon_failure, is_transport_failure

# Mirrors transport.py's own private marker vocabulary, kept as a literal here rather than
# imported so each new marker demands a deliberate new test case, not silent inherited coverage.
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
    "permission denied",
    "too many authentication failures",
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
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.raise_timeout = raise_timeout
        self.communicate_calls = 0
        self.wait_calls: list[float | None] = []

    def communicate(self, timeout: float) -> tuple[str, str]:
        self.communicate_calls += 1
        if self.raise_timeout and self.communicate_calls == 1:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return self.stdout_text, self.stderr_text

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)


class _FakeSshProcess:
    """A minimal stand-in for the process a `ShellSession` wraps, just `pid` and `poll()`."""

    def __init__(self, *, alive: bool) -> None:
        self.pid = 999
        self.alive = alive

    def poll(self) -> int | None:
        return None if self.alive else 0


@pytest.mark.parametrize("marker", _TRANSPORT_MARKERS)
def test_a_transport_fault_needs_both_the_ssh_exit_status_and_a_known_marker(marker: str) -> None:
    """A name that will not resolve is a host we cannot reach now, not a command that failed."""
    assert is_transport_failure(255, f"ssh: {marker} happened") is True
    assert is_transport_failure(1, f"ssh: {marker} happened") is False
    assert is_transport_failure(255, "some unrelated message") is False


@pytest.mark.parametrize("marker", _DAEMON_DOWN_MARKERS)
def test_a_dead_scheduler_daemon_is_any_refused_control_socket_marker(marker: str) -> None:
    assert is_daemon_failure(f"client error: {marker}") is True
    assert is_daemon_failure("totally unrelated") is False
    assert issubclass(DaemonDown, HostUnreachable)


def test_the_ssh_policy_overrides_liveness_and_leaves_every_alias_setting_intact() -> None:
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
    assert policy.rsync_shell == "ssh -o ConnectTimeout=5 -o ServerAliveInterval=3 " + (
        "-o ServerAliveCountMax=2 -o BatchMode=yes"
    )


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "raised", "detail"),
    [
        (0, "ok\n", "", None, "ok\n"),
        (255, "", "kex_exchange identification failed", HostUnreachable, "kex_exchange"),
        (255, "", "Host key verification failed.", ConnectionError, "host-key verification"),
        (1, "", "remote command exploded", RuntimeError, "remote command exploded"),
        (7, "", "", RuntimeError, "exit 7"),
    ],
)
def test_run_returns_stdout_on_a_clean_exit_and_types_every_other_ending(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
    raised: type[BaseException] | None,
    detail: str,
) -> None:
    policy = SshTransport()
    process = _FakeProcess(returncode=returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    if raised is None:
        assert policy.run(("ssh", "host", "true"), "host", operation="connect") == detail
        return
    with pytest.raises(raised, match=detail):
        policy.run(("ssh", "host", "true"), "host", operation="connect")


def test_run_reports_a_host_unreachable_when_ssh_cannot_even_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a, **_k) -> None:
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(HostUnreachable, match="could not start"):
        SshTransport().run(("ssh", "host", "true"), "host", operation="connect")


def test_a_timed_out_transfer_takes_its_whole_process_group_down_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ProxyJump child outliving its parent is what leaves an orphaned ssh behind."""
    process = _FakeProcess(raise_timeout=True)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    killed: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        transport_module,
        "terminate_process_tree",
        lambda pid, *, force=False: killed.append((pid, force)),
    )
    with pytest.raises(HostUnreachable, match="timed out"):
        SshTransport().run(("ssh", "host", "true"), "host", operation="connect")
    assert killed == [(process.pid, False)]
    assert process.wait_calls == [2.0]


def test_every_ssh_this_policy_runs_reads_devnull_and_never_the_callers_own_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submit inside a shell loop used to eat the loop's remaining input.

    An ssh client left on its caller's stdin reads it greedily to forward to the far side, and
    every remote verb warms a connection before it does anything, so
    `while read handle; do mainboard submit ...; done < handles` fed the first submit's warm-up
    the rest of the file and the loop ran once. Nothing here wants a caller's input: the warm-up
    runs `true`, scp moves a file, and a real remote command rides plumbum's own piped session.
    """
    opened: list[dict[str, object]] = []

    def record(*args: object, **kwargs: object) -> _FakeProcess:
        opened.append(kwargs)
        return _FakeProcess(stdout="ok\n")

    monkeypatch.setattr(subprocess, "Popen", record)
    policy = SshTransport()
    policy.warm("gold")
    policy.copy("job.sh", destination="gold:/repo/job.sh", host="gold")
    assert [call["stdin"] for call in opened] == [subprocess.DEVNULL, subprocess.DEVNULL]


def test_terminate_escalates_to_sigkill_and_tolerates_a_group_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stubborn = _FakeProcess()

    def wait(timeout: float | None = None) -> None:
        stubborn.wait_calls.append(timeout)
        if len(stubborn.wait_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    stubborn.wait = wait
    killed: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        transport_module,
        "terminate_process_tree",
        lambda pid, *, force=False: killed.append((pid, force)),
    )
    SshTransport.terminate(stubborn)
    assert killed == [(stubborn.pid, False), (stubborn.pid, True)]
    assert stubborn.wait_calls == [2.0, None]

    def raise_lookup(pid: int, *, force: bool = False) -> None:
        del pid, force
        raise ProcessLookupError

    monkeypatch.setattr(transport_module, "terminate_process_tree", raise_lookup)
    gone = _FakeProcess()
    SshTransport.terminate(gone)
    assert gone.wait_calls == [2.0]


def test_warm_and_copy_ride_the_same_policy_and_machine_opens_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], str, str]] = []
    monkeypatch.setattr(
        SshTransport,
        "run",
        lambda self, command, host, *, operation: calls.append((command, host, operation)),
    )
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
    policy.warm("gold")
    policy.copy("a.txt", destination="gold:b.txt", host="gold")
    policy.machine("gold")
    assert calls[0][0][:2] == ("ssh", "-o")
    assert calls[0][1:] == ("gold", "connect")
    assert calls[1][0][:2] == ("scp", "-o")
    assert calls[1][1:] == ("gold", "copy")
    assert built["host"] == "gold"
    assert built["ssh_opts"] == policy.options
    assert built["connect_timeout"] == policy.deadline
    assert built["new_session"] is True


@pytest.mark.parametrize(
    ("process", "signalled"),
    [
        (_FakeSshProcess(alive=True), [(999, signal.SIGTERM)]),
        (_FakeSshProcess(alive=False), []),
        (None, []),
    ],
)
def test_closing_a_bounded_session_kills_only_a_group_that_is_still_alive(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeSshProcess | None,
    signalled: list[tuple[int, int]],
) -> None:
    session = object.__new__(transport_module.BoundedShellSession)
    session.proc = process
    killed: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        transport_module,
        "terminate_process_tree",
        lambda pid, *, force=False: killed.append((pid, force)),
    )
    closed: list[bool] = []
    monkeypatch.setattr(transport_module.ShellSession, "close", lambda self: closed.append(True))
    session.close()
    assert killed == [(pid, False) for pid, _signal in signalled]
    assert closed == [True]


def test_closing_a_bounded_session_tolerates_a_group_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(transport_module.BoundedShellSession)
    session.proc = _FakeSshProcess(alive=True)

    def raise_lookup(pid: int, *, force: bool = False) -> None:
        del pid, force
        raise ProcessLookupError

    monkeypatch.setattr(transport_module, "terminate_process_tree", raise_lookup)
    monkeypatch.setattr(transport_module.ShellSession, "close", lambda self: None)
    session.close()


@pytest.mark.parametrize(("isatty", "flags"), [(True, ["-tt"]), (False, ["-T"])])
def test_a_bounded_machine_opens_its_shell_in_a_dedicated_process_group(
    monkeypatch: pytest.MonkeyPatch, isatty: bool, flags: list[str]
) -> None:
    machine = object.__new__(transport_module.BoundedSshMachine)
    machine.custom_encoding = "utf-8"
    machine.connect_timeout = 15.0
    machine.host = "gold"
    opened: list[tuple[list[str], list[str], bool]] = []
    machine.popen = lambda argv, extra, new_session: (
        opened.append((argv, extra, new_session)) or "PROC"
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
    result = machine.session(isatty=isatty, new_session=isatty)
    assert isinstance(result, FakeSession)
    assert opened == [(["/bin/sh"], flags, isatty)]
    assert (built["proc"], built["host"], built["isatty"]) == ("PROC", "gold", isatty)
