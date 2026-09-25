"""The rsync-less mirror a Windows host gets: listed by rsync, diffed, streamed through tar.

Only the two processes of the stream are scripted. The listing is rsync's own dry run and the
host's answer comes through the recording transport, so what is asserted is which files this
side decided to send and how it read the host's reply.
"""

import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard import ExecutionPlan, MissionError
from mainboard.dispatch import HostUnreachable
from mainboard.dispatch import tarball as tarball_module
from mainboard.dispatch.tarball import Tarball
from mainboard.manifest import HostProfile

from .support import RecordingTransport, plan

_ROOT = "C:/Users/me/mainboard-managed"

needs_rsync = pytest.mark.skipif(
    shutil.which("rsync") is None, reason="the listing is rsync's own dry run"
)


class Process:
    """One scripted end of the stream: tar packing on this side or ssh unpacking on the host."""

    def __init__(
        self, argv: list[str], *, returncode: int, stderr: str | bytes, stdout: BytesIO | None
    ) -> None:
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout

    def communicate(self) -> tuple[str, str | bytes]:
        return "", self.stderr


class Stream:
    """A `subprocess.Popen` double recording both ends and the names tar was told to pack.

    ssh: the `(returncode, stderr)` the unpacking end answers with.
    warning: what tar writes to stderr while packing.
    piped: whether tar hands the unpacking end a pipe; a real `Popen` always does.
    """

    def __init__(
        self, *, ssh: tuple[int, str] = (0, ""), warning: bytes = b"", piped: bool = True
    ) -> None:
        self.ssh = ssh
        self.warning = warning
        self.piped = piped
        self.processes: list[Process] = []
        self.names: list[str] = []
        self.stdin: list[BytesIO | None] = []

    def __call__(self, argv: list[str], **options: BytesIO | int | str | bool) -> Process:
        if argv[0] == "tar":
            listed = next(arg for arg in argv if arg.startswith("--files-from="))
            self.names = Path(listed.partition("=")[2]).read_text(encoding="utf-8").split("\0")
            process = Process(
                argv, returncode=0, stderr=self.warning, stdout=BytesIO() if self.piped else None
            )
        else:
            stdin = options["stdin"]
            self.stdin.append(stdin if isinstance(stdin, BytesIO) else None)
            process = Process(argv, returncode=self.ssh[0], stderr=self.ssh[1], stdout=None)
        self.processes.append(process)
        return process

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stand in for the tarball module's `Popen` alone; its rsync listing still really runs."""
        real = tarball_module.subprocess
        scripted = SimpleNamespace(run=real.run, PIPE=real.PIPE, Popen=self)
        monkeypatch.setattr(tarball_module, "subprocess", scripted)


def windows_plan() -> ExecutionPlan:
    """A plan for a Windows host whose workspace sits at `_ROOT`."""
    profile = HostProfile(kind="ssh", root=_ROOT, platform="win-64", sync={"include": ["src"]})
    return plan(host="homelab", profile=profile)


def test_the_stream_packs_the_named_files_and_unpacks_them_under_the_root_in_one_channel(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """tar's own complaints, a dangling link it skipped, are warned about and not fatal."""
    stream = Stream(warning=b"tar: vendor/gone: Cannot stat\n")
    stream.install(monkeypatch)
    with caplog.at_level("WARNING", logger="mainboard.dispatch"):
        Tarball(Path("."), RecordingTransport()).stream(
            ["src/a.py", "src/b c.py"], host="homelab", root=_ROOT
        )
    pack, unpack = stream.processes
    assert stream.names == ["src/a.py", "src/b c.py"]
    assert {"--dereference", "--null", "--ignore-failed-read", "--file=-"} <= set(pack.argv)
    assert unpack.argv == ["ssh", "-o", "BatchMode=yes", "homelab", f'tar -xzf - -C "{_ROOT}"']
    assert stream.stdin == [pack.stdout]
    assert pack.stdout is not None and pack.stdout.closed, "only the unpacking end holds the pipe"
    assert any("vendor/gone: Cannot stat" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("ssh", "fault", "reason"),
    [
        pytest.param(
            (255, "ssh: connect to host homelab port 22: Connection refused"),
            HostUnreachable,
            "mirror to 'homelab' failed: ssh: connect",
            id="a-host-that-dropped",
        ),
        pytest.param(
            (1, "tar.exe: Error opening archive"),
            MissionError,
            "'homelab' could not unpack the mirror: tar.exe: Error opening archive",
            id="a-host-whose-tar-refused",
        ),
        pytest.param((1, ""), MissionError, "no detail", id="a-host-that-said-nothing"),
    ],
)
def test_a_stream_the_host_did_not_unpack_is_refused_by_what_went_wrong(
    ssh: tuple[int, str],
    fault: type[Exception],
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped host is retried like any unreachable one; a refused unpack is the host's own."""
    Stream(ssh=ssh, piped=False).install(monkeypatch)
    with pytest.raises(fault, match=reason):
        Tarball(Path("."), RecordingTransport()).stream(["src/a.py"], host="homelab", root=_ROOT)


@needs_rsync
def test_a_mirror_ships_only_what_the_host_lacks_and_follows_the_vendored_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's own listing is the diff, so a second mirror of an unchanged tree sends nothing.

    The vendored tree is links by construction and the host cannot make them, so its files are
    listed through the links while the workspace's own links stay links.
    """
    workspace = tmp_path / "ws"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "held.py").write_text("print(1)\n")
    (workspace / "src" / "new.py").write_text("print(2)\n")
    (workspace / "store").mkdir()
    (workspace / "store" / "lib.py").write_text("x = 1\n")
    (workspace / "vendor").mkdir()
    (workspace / "vendor" / "lib.py").symlink_to(workspace / "store" / "lib.py")
    held = (workspace / "src" / "held.py").stat()
    reply = f"src/held.py\t{held.st_size}\t{int(held.st_mtime)}\n"
    transport = RecordingTransport(rules=[("Get-ChildItem", 0, reply)])
    stream = Stream()
    stream.install(monkeypatch)
    tarball = Tarball(workspace, transport)
    rules = {"include": [], "exclude": [], "hide": [], "filters": []}
    files = tarball.mirror(windows_plan(), _ROOT, paths=["src"], vendored="vendor", **rules)
    assert sorted(files) == ["src/held.py", "src/new.py", "vendor/lib.py"]
    assert transport.ran(f"New-Item -ItemType Directory -Force -Path '{_ROOT}'")
    assert sorted(stream.names) == ["src/new.py", "vendor/lib.py"]
    everything = "".join(
        f"{file}\t{(workspace / file).stat().st_size}\t{int((workspace / file).stat().st_mtime)}\n"
        for file in files
    )
    settled = Stream()
    settled.install(monkeypatch)
    Tarball(workspace, RecordingTransport(rules=[("Get-ChildItem", 0, everything)])).mirror(
        windows_plan(), _ROOT, paths=["src"], **rules
    )
    assert settled.processes == []
