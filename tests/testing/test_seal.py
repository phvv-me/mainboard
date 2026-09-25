import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from mainboard.testing import REMOTE_TOOLS, RemoteReached, Seal, is_local, program


@pytest.mark.parametrize(
    ("args", "name"),
    [
        ("ssh -o BatchMode=yes gold true", "ssh"),
        (b"/usr/bin/scp a b", "scp"),
        (PurePosixPath("/opt/pbs/bin/qsub"), "qsub"),
        (["C:\\Windows\\System32\\OpenSSH\\SSH.EXE", "box"], "ssh"),
        ([b"rsync", b"-a"], "rsync"),
        ("", ""),
        ([], ""),
    ],
    ids=["a line", "bytes", "a path", "a Windows argv", "a bytes argv", "nothing", "no words"],
)
def test_the_program_is_the_first_word_by_its_bare_name(
    args: str | bytes | PurePosixPath | list[str] | list[bytes], name: str
) -> None:
    assert program(args) == name


@pytest.mark.parametrize(
    ("address", "local"),
    [
        ("/tmp/agent.sock", True),
        (("localhost", 80), True),
        (("127.0.0.53", 53), True),
        (("::1", 22, 0, 0), True),
        (("fe80::1%en0", 22, 0, 0), False),
        (("", 8080), True),
        (("10.0.0.7", 22), False),
        (("gold.example.org", 443), False),
    ],
)
def test_only_this_machine_counts_as_local(address: tuple[str, int] | str, local: bool) -> None:
    assert is_local(address) is local


@pytest.fixture
def seal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Seal:
    """A second seal over the suite's own, so its record is this test's alone."""
    made = Seal.made(tmp_path / "stand-ins")
    made.install(monkeypatch)
    return made


def test_a_remote_tool_is_refused_by_name_and_everything_else_still_runs(seal: Seal) -> None:
    with pytest.raises(RemoteReached, match="spawned ssh"):
        subprocess.run(["ssh", "gold", "true"], check=False)
    done = subprocess.run([sys.executable, "-c", "print('here')"], capture_output=True, text=True)
    assert done.stdout.strip() == "here"
    assert seal.breaches() == ["spawned ssh"]
    seal.attempts.clear()


def test_a_connection_off_this_machine_is_refused_and_loopback_still_connects(
    seal: Seal,
) -> None:
    with (
        socket.create_server(("127.0.0.1", 0)) as server,
        socket.create_connection(server.getsockname(), timeout=5),
    ):
        pass
    probe = socket.socket()
    with probe, pytest.raises(RemoteReached, match=r"connected to 10\.9\.9\.9:22"):
        probe.connect_ex(("10.9.9.9", 22))
    assert seal.breaches() == ["connected to 10.9.9.9:22"]
    seal.attempts.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-ins are POSIX shell scripts")
def test_a_tool_that_starts_ssh_itself_meets_a_stand_in_that_logs_it(seal: Seal) -> None:
    """rsync's `-e` and git over ssh never pass back through Python, so PATH catches them."""
    done = subprocess.run(
        [shutil.which("sh") or "sh", "-c", "ssh gold uptime"], capture_output=True, text=True
    )
    assert done.returncode == 255
    assert "sealed" in done.stderr
    assert seal.breaches() == ["spawned ssh gold uptime"]
    seal.log.write_text("", encoding="utf-8")


def test_every_remote_tool_has_a_stand_in(tmp_path: Path) -> None:
    made = Seal.made(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} >= REMOTE_TOOLS
    assert os.access(made.stand_ins / "ssh", os.X_OK) or sys.platform == "win32"


def test_a_test_that_swallowed_the_refusal_still_fails(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["mainboard.testing"]')
    pytester.makepyfile(
        """
        import subprocess

        def test_quietly_reaches_out():
            try:
                subprocess.run(["scp", "a", "gold:b"])
            except Exception:
                pass
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "-o", "addopts=")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*the test reached another machine: spawned scp*"])
