import os
import shlex
import signal
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=monkeypatches Popen for hermetic tests, never runs a real process since=2026-08-18
import sys
from math import inf
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import BinaryIO

import psutil
import pytest

from mainboard.dispatch import DaemonDown, HostUnreachable, SshTransport
from mainboard.dispatch import transport as transport_module
from mainboard.dispatch.transport import (
    Endpoint,
    is_daemon_failure,
    is_transport_failure,
    terminate_process_tree,
)

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

    def communicate(self, timeout: float, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.raise_timeout and self.communicate_calls == 1:
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return self.stdout_text.encode(), self.stderr_text.encode()

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls.append(timeout)


class _FakeSshProcess:
    """A minimal stand-in for the process a `ShellSession` wraps, just `pid` and `poll()`."""

    def __init__(self, *, alive: bool) -> None:
        self.pid = 999
        self.alive = alive

    def poll(self) -> int | None:
        return None if self.alive else 0


class _TreeProcess:
    """One native-process node recording the signal selected by the tree terminator."""

    def __init__(self, pid: int, events: list[str], *, vanished: bool = False) -> None:
        self.pid = pid
        self.events = events
        self.vanished = vanished
        self.descendants: list[_TreeProcess] = []

    def children(self, *, recursive: bool) -> list[_TreeProcess]:
        assert recursive is True
        return self.descendants

    def terminate(self) -> None:
        self._signal("terminate")

    def kill(self) -> None:
        self._signal("kill")

    def _signal(self, action: str) -> None:
        self.events.append(f"{action}:{self.pid}")
        if self.vanished:
            raise psutil.NoSuchProcess(self.pid)


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


def test_a_native_process_tree_is_signalled_children_first_and_tolerates_a_vanished_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    root = _TreeProcess(1, events)
    root.descendants = [_TreeProcess(2, events), _TreeProcess(3, events, vanished=True)]
    monkeypatch.setattr(transport_module.psutil, "Process", lambda pid: root)

    terminate_process_tree(root.pid)
    terminate_process_tree(root.pid, force=True)

    assert events == [
        "terminate:3",
        "terminate:2",
        "terminate:1",
        "kill:3",
        "kill:2",
        "kill:1",
    ]


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


@pytest.mark.parametrize(
    ("operation", "raised"), [("connect", HostUnreachable), ("command", RuntimeError)]
)
def test_a_silent_255_is_unreachable_only_for_the_connect_probe(
    monkeypatch: pytest.MonkeyPatch, operation: str, raised: type[BaseException]
) -> None:
    """The probe runs `echo`, so its 255 is ssh's own; a real command may exit 255 itself.

    Left untyped, a probe that failed without a marker escaped the rental's knock loop as a bare
    error instead of being knocked again (RTX 4090 52521884, 2026-09-25).
    """
    process = _FakeProcess(returncode=255, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(raised) as caught:
        SshTransport().run(("ssh", "host", "true"), "host", operation=operation)
    assert type(caught.value) is raised


@pytest.mark.parametrize(
    ("returncode", "stderr", "raised"),
    [
        pytest.param(0, "", None, id="a-clean-answer"),
        pytest.param(3, "not installed", None, id="a-probe-that-said-no"),
        pytest.param(255, "Connection refused", HostUnreachable, id="a-host-that-dropped"),
        pytest.param(1, "Host key verification failed.", ConnectionError, id="a-changed-host-key"),
    ],
)
def test_invoke_answers_any_exit_a_command_gave_and_raises_only_what_ssh_itself_hit(
    returncode: int, stderr: str, raised: type[BaseException] | None
) -> None:
    """A probe exiting non-zero answered the question; a transport fault answered nothing."""
    script = (
        "import sys; sys.stdout.write('said'); "
        f"sys.stderr.write({stderr!r}); sys.exit({returncode})"
    )
    command = (sys.executable, "-c", script)
    if raised is not None:
        with pytest.raises(raised):
            SshTransport().invoke(command, "host", operation="probe")
        return
    answer = SshTransport().invoke(command, "host", operation="probe", timeout=inf)
    assert answer == (returncode, "said", stderr)


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


@pytest.mark.parametrize("output_file", [False, True])
def test_run_pipes_only_explicit_input_and_streams_stdout_bytes(
    tmp_path: Path, output_file: bool
) -> None:
    output = tmp_path / "stream.bin" if output_file else None
    script = "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)"
    answer = SshTransport().run(
        (sys.executable, "-c", script),
        "local",
        operation="stream",
        input_text="line one\nUnicode: 日本語\n",
        output=output,
    )
    if output is None:
        assert answer == "line one\nUnicode: 日本語\n"
    else:
        assert answer == ""
        assert output.read_bytes() == "line one\nUnicode: 日本語\n".encode()


@pytest.mark.parametrize("exit_code", [0, 1])
def test_stream_keeps_binary_bytes_and_preserves_command_failure(
    tmp_path: Path, exit_code: int
) -> None:
    output = tmp_path / "partial.bin"
    script = (
        "import sys; sys.stdout.buffer.write(bytes(range(256))); "
        f"sys.stderr.write('command failed'); sys.exit({exit_code})"
    )
    command = (sys.executable, "-c", script)
    if exit_code:
        with pytest.raises(RuntimeError, match="command failed"):
            SshTransport().run(command, "local", operation="stream", output=output)
    else:
        assert SshTransport().run(command, "local", operation="stream", output=output) == ""
    assert output.read_bytes() == bytes(range(256))


def test_evidence_stream_can_outlive_the_control_deadline(tmp_path: Path) -> None:
    class ShortPolicy(SshTransport):
        @property
        def deadline(self) -> float:
            return 0.1

        @property
        def stream_deadline(self) -> float:
            return 5.0

    command = (
        sys.executable,
        "-c",
        # Raw bytes, since a Windows `print` would end its own line in `\r\n`.
        "import sys, time; time.sleep(0.3); sys.stdout.buffer.write(b'complete evidence\\n')",
    )
    policy = ShortPolicy()
    with pytest.raises(HostUnreachable, match="timed out after 0.1s"):
        policy.run(command, "local", operation="control")
    output = tmp_path / "evidence.bin"
    assert policy.run(command, "local", operation="collect", output=output) == ""
    assert output.read_bytes() == b"complete evidence\n"


def test_stream_timeout_closes_staging_file_and_terminates_process_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess(raise_timeout=True)
    killed: list[tuple[int, bool]] = []
    opened: list[BinaryIO] = []

    def spawn(command: tuple[str, ...], **kwargs: BinaryIO | str | bool | int) -> _FakeProcess:
        sink = cast("BinaryIO", kwargs["stdout"])
        sink.write(b"partial")
        opened.append(sink)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(
        transport_module,
        "terminate_process_tree",
        lambda pid, *, force=False: killed.append((pid, force)),
    )
    output = tmp_path / "partial.bin"
    with pytest.raises(HostUnreachable, match="timed out"):
        SshTransport().run(("ssh", "host"), "host", operation="stream", output=output)
    assert output.read_bytes() == b"partial"
    assert opened[0].closed
    assert killed == [(process.pid, False)]


def test_every_ssh_this_policy_runs_reads_devnull_and_never_the_callers_own_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submit inside a shell loop used to eat the loop's remaining input.

    An ssh client left on its caller's stdin reads it greedily to forward to the far side, and
    every remote verb warms a connection before it does anything, so
    `while read handle; do mainboard submit ...; done < handles` fed the first submit's warm-up
    the rest of the file and the loop ran once. Nothing here wants a caller's input: the warm-up
    echoes a marker, scp moves a file, and a real remote command rides plumbum's own piped session.
    """
    opened: list[dict[str, object]] = []

    def record(*args: object, **kwargs: object) -> _FakeProcess:
        opened.append(kwargs)
        return _FakeProcess(stdout="mainboard-reachable\n")

    monkeypatch.setattr(subprocess, "Popen", record)
    policy = SshTransport()
    policy.warm("gold")
    policy.transfer("job.sh", destination="gold:/repo/job.sh", host="gold")
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


def test_warm_and_transfer_ride_the_same_policy_and_machine_opens_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], str, str]] = []

    def run(self: SshTransport, command: tuple[str, ...], host: str, *, operation: str) -> str:
        calls.append((command, host, operation))
        return "mainboard-reachable\r\n"

    monkeypatch.setattr(SshTransport, "run", run)
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
    policy.transfer("a.txt", destination="gold:b.txt", host="gold")
    policy.machine("gold")
    assert calls[0][0][:2] == ("ssh", "-o")
    assert calls[0][0][-3:] == ("gold", "echo", "mainboard-reachable")
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


def test_a_policy_bound_to_a_rental_carries_where_that_machine_is_past_the_liveness_options() -> (
    None
):
    """A rented box has no alias, so the address, port, login and key ride with the policy.

    Its host key is new every time and never seen again, which is why the connection accepts it
    on sight and keeps it out of `known_hosts` rather than failing verification against whatever
    key some earlier rental had at a recycled address.
    """
    endpoint = Endpoint(address="ssh5.vast.ai", port=41022, user="root", identity="/keys/id")
    policy = SshTransport(endpoint=endpoint)
    assert endpoint.destination == "root@ssh5.vast.ai"
    assert policy.destination("vast") == "root@ssh5.vast.ai"
    assert policy.options[: len(policy.liveness)] == policy.liveness
    assert policy.options[len(policy.liveness) :] == (
        "-p",
        "41022",
        "-i",
        str(Path("/keys/id")),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={os.devnull}",
        "-o",
        "LogLevel=ERROR",
    )
    key = shlex.join(("-i", str(Path("/keys/id"))))
    assert "-p 41022" in policy.rsync_shell and key in policy.rsync_shell
    assert Path(Endpoint(address="a", identity="~/.ssh/id").identity).is_absolute()


@pytest.mark.parametrize("device", ["/dev/null", "nul"])
def test_rental_known_hosts_uses_the_client_null_device(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    monkeypatch.setattr(transport_module.os, "devnull", device)
    assert f"UserKnownHostsFile={device}" in Endpoint(address="rental").options


@pytest.mark.parametrize("output", ["", "banner only\n", "prefix-mainboard-reachable\n"])
def test_warm_refuses_success_without_its_complete_marker(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(SshTransport, "run", lambda self, command, host, *, operation: output)
    with pytest.raises(RuntimeError, match="without the expected marker"):
        SshTransport().warm("gold")


def test_an_unbound_policy_names_the_alias_and_a_bound_one_spells_the_port_scp_way() -> None:
    """scp differs from ssh in one letter, and a declared host keeps its own config untouched."""
    assert SshTransport().destination("gold") == "gold"
    assert SshTransport().options == SshTransport().liveness
    bare = Endpoint(address="1.2.3.4")
    assert bare.destination == "1.2.3.4" and "-p" not in bare.options and "-i" not in bare.options
    ported = Endpoint(address="1.2.3.4", port=2222)
    assert ported.options[:2] == ("-p", "2222")
    assert ported.scp_options[:2] == ("-P", "2222")
