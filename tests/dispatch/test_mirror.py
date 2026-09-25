"""The native mirror end to end: what crosses, what is pruned, what is kept, how a target fails."""

import io
import os
import shlex
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard import MissionError
from mainboard.dispatch import HostUnreachable
from mainboard.dispatch.agent import Agent, AgentRefused, Link, Rules, Scope, SshLink, program
from mainboard.dispatch.agent.runner import BOOTSTRAP
from mainboard.dispatch.mirror import Delta, Mirror, clashing
from mainboard.dispatch.sync import ALWAYS_EXCLUDE, CARD_LEASES, GitignoreFilter, patterns
from mainboard.dispatch.transport import SshTransport

from .support import InProcessLink, links_on_this_host

# A Windows target keeps neither execute bits nor links, the two things its survey declines.
_MODES = sys.platform != "win32"

# This interpreter as a shell would have it typed, which a path with a space needs quoted.
_PYTHON = (
    subprocess.list2cmdline([sys.executable]) if os.name == "nt" else shlex.quote(sys.executable)
)


def seed(root: Path, *files: str) -> None:
    """Create each relative path under `root`, parents included, with its own name as content."""
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def pushed(
    work: Path,
    host: Path,
    roots: list[str],
    *,
    named: tuple[str, ...] = (),
    hidden: tuple[str, ...] = (),
    protect: tuple[str, ...] = (),
    link: Link | None = None,
):
    """One mirror of `roots` from `work` onto `host`, under the rules a dispatch applies."""
    ignores = GitignoreFilter(work)
    deny = patterns([*ALWAYS_EXCLUDE, *CARD_LEASES], paths=hidden)
    agent = Agent(link or InProcessLink(), python=_PYTHON)
    return Mirror(work, agent).push(
        str(host),
        scopes=[ignores.scope(roots, deny=deny)],
        named=named,
        protected=patterns([*CARD_LEASES, *protect], paths=hidden),
    )


def _receive(root: Path) -> dict[str, dict[str, object]]:
    """A receive request for `root` announcing nothing to delete, make, link or write."""
    return {
        "receive": {
            "root": str(root),
            "state": "s",
            "delete": [],
            "directories": [],
            "links": {},
            "files": {},
        }
    }


def test_a_mirror_ships_the_scope_prunes_the_stale_and_keeps_what_the_rules_keep(
    tmp_path: Path,
) -> None:
    """The one proof the whole rule order behaves: ignored, protected, leased, hidden, named."""
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, ".gitignore", "src/run.py", "src/deep/mod.py", "src/local.scratch")
    seed(work, "src/.card.lock.sender", "src/out/local.json", "outside/named.txt")
    (work / ".gitignore").write_text("*.scratch\n", encoding="utf-8")
    (work / "src/empty").mkdir()
    seed(host, "src/stale.py", "src/host-only.scratch", "src/results/e1.json", "src/.card.lock")
    seed(host, "src/out/live.json", "elsewhere/untouched.py", "src/gone/deeper/old.py")
    rules = {"named": ("outside/named.txt",), "hidden": ("src/out",), "protect": ("results/***",)}
    first = pushed(work, host, ["src"], **rules)
    assert (first.files, first.sent) == (3, 3)
    assert set(first.deleted) == {
        "src/stale.py",
        "src/gone",
        "src/gone/deeper",
        "src/gone/deeper/old.py",
    }
    assert (host / "src/deep/mod.py").read_text(encoding="utf-8") == "src/deep/mod.py"
    assert (host / "outside/named.txt").is_file() and (host / "src/empty").is_dir()
    for kept in ("src/host-only.scratch", "src/results/e1.json", "src/.card.lock"):
        assert (host / kept).is_file(), kept
    assert (host / "src/out/live.json").is_file() and not (host / "src/out/local.json").exists()
    assert not (host / "src/local.scratch").exists()
    assert not (host / "src/.card.lock.sender").exists()
    assert (host / "elsewhere/untouched.py").is_file()
    again = pushed(work, host, ["src"], **rules)
    assert (again.sent, again.deleted) == (0, ())


@pytest.mark.skipif(not _MODES, reason="Windows keeps no change time that a write moves")
def test_content_decides_what_differs_never_a_timestamp(tmp_path: Path) -> None:
    """Same size and same time with other bytes still crosses; a touched twin does not."""
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/a.py", "src/b.py")
    pushed(work, host, ["src"])
    edited = work / "src/a.py"
    before = edited.stat()
    edited.write_text("src/x.py", encoding="utf-8")
    os.utime(edited, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.utime(work / "src/b.py", ns=(1, 1))
    assert pushed(work, host, ["src"]).sent == 1
    assert (host / "src/a.py").read_text(encoding="utf-8") == "src/x.py"


@pytest.mark.skipif(not _MODES, reason="Windows keeps no execute bit")
def test_an_execute_bit_crosses_on_its_own(tmp_path: Path) -> None:
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/job.sh")
    pushed(work, host, ["src"])
    (work / "src/job.sh").chmod(0o755)
    assert pushed(work, host, ["src"]).sent == 1
    assert (host / "src/job.sh").stat().st_mode & stat.S_IXUSR


@links_on_this_host
def test_a_path_that_changed_kind_is_pruned_before_it_is_remade(tmp_path: Path) -> None:
    """Neither a directory nor a link is replaced by a rename, so each goes first.

    A Windows target holds no links, so there each link arrives as the file it names.
    """
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/was-dir", "src/target.txt")
    (work / "src/link").symlink_to("target.txt")
    (work / "src/moved").symlink_to("target.txt")
    seed(host, "src/was-dir/inside.py", "src/link")
    (host / "src").mkdir(exist_ok=True)
    (host / "src/moved").symlink_to("elsewhere.txt")
    pushed(work, host, ["src"])
    assert (host / "src/was-dir").is_file()
    for name in ("link", "moved"):
        placed = host / "src" / name
        if _MODES:
            assert os.readlink(placed) == "target.txt"
        else:
            assert placed.read_text(encoding="utf-8") == "src/target.txt"


@links_on_this_host
def test_a_target_that_holds_no_links_takes_each_file_link_as_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows ships no link it can make, and a link to a directory is left behind by name."""
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/real.txt", "src/sub/x.py")
    (work / "src/alias.txt").symlink_to("real.txt")
    (work / "src/folder").symlink_to("sub", target_is_directory=True)
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=lambda *a: None)
    )
    monkeypatch.setattr(program, "WINDOWS", True)
    warned: list[str] = []
    monkeypatch.setattr(
        "mainboard.dispatch.mirror.logger.warning", lambda message, *args: warned.append(args[-1])
    )
    pushed(work, host, ["src"])
    assert (host / "src/alias.txt").read_text(encoding="utf-8") == "src/real.txt"
    assert not (host / "src/alias.txt").is_symlink()
    assert not (host / "src/folder").exists()
    assert warned == ["src/folder"]


def test_a_folding_target_refuses_paths_that_differ_only_in_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two names a case-folding file system holds as one would overwrite each other there."""
    assert clashing(["src/A.py", "src/a.py", "src/b.py"]) == ["src/A.py", "src/a.py"]
    assert clashing(["src/a.py", "src/b.py"]) == []
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/a.py")
    monkeypatch.setattr(program, "_folds", lambda directory: True)
    assert pushed(work, host, ["src"]).sent == 1
    monkeypatch.setattr("mainboard.dispatch.mirror.clashing", lambda paths: ["src/A", "src/a"])
    with pytest.raises(MissionError, match="differ only in case: src/A, src/a"):
        pushed(work, host, ["src"])


@pytest.mark.parametrize("named", [False, True])
def test_a_file_gone_before_the_stream_is_skipped_unless_it_was_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: bool
) -> None:
    """A named file is one the dispatch cannot run without, so its absence fails the mirror."""
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/stays.py", "src/goes.py")
    between = Delta.between

    def vanishing(*args, **kwargs) -> Delta:
        delta = between(*args, **kwargs)
        (work / "src/goes.py").unlink()
        return delta

    monkeypatch.setattr(Delta, "between", vanishing)
    if named:
        with pytest.raises(MissionError, match="src/goes.py vanished"):
            pushed(work, host, ["src"], named=("src/goes.py",))
        return
    assert pushed(work, host, ["src"]).sent == 1
    assert (host / "src/stays.py").is_file() and not (host / "src/goes.py").exists()


def test_a_vendored_tree_ships_what_its_links_refer_to_and_prunes_only_itself(
    tmp_path: Path,
) -> None:
    """A host holds no source to link to, so it gets the files, and never another tree's."""
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "vendor/pkg/mod.py", "vendor/pkg/__pycache__/x.pyc")
    seed(host, "vendor/retired/old.py", "other/kept.py")
    scope = Scope(["vendor"], deny=patterns(["__pycache__/"]), follow=True)
    agent = Agent(InProcessLink(), python=sys.executable)
    done = Mirror(work, agent).push(str(host), scopes=[scope], protected=Rules())
    assert (host / "vendor/pkg/mod.py").is_file()
    assert not (host / "vendor/pkg/__pycache__").exists()
    assert not (host / "vendor/retired").exists() and (host / "other/kept.py").is_file()
    assert "vendor/retired/old.py" in done.deleted


def test_an_agent_that_refuses_or_breaks_is_named_by_its_last_line(tmp_path: Path) -> None:
    """A root that is a file cannot hold a workspace, and the traceback's last line says so."""
    blocked = tmp_path / "blocked"
    blocked.write_text("a file", encoding="utf-8")
    agent = Agent(InProcessLink(), python=sys.executable)
    with pytest.raises(AgentRefused, match="Error"):
        agent.ask({"survey": {"root": str(blocked), "state": "s", "scopes": [], "named": []}})


class _Silent:
    """A process that says nothing and never exits until it is ended."""

    def __init__(self, *, pipes: bool = True) -> None:
        self.stdin = None
        read, write = os.pipe()
        self.stdout = os.fdopen(read, "rb") if pipes else None
        self.stderr = None
        self.feed = os.fdopen(write, "wb")
        self.pid = os.getpid()
        self.returncode: int | None = None

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            time.sleep(timeout or 0.0)
            raise subprocess.TimeoutExpired("agent", timeout or 0.0)
        return self.returncode


class _SilentLink(Link):
    def __init__(self, process: _Silent) -> None:
        super().__init__("quiet")
        self.process = process
        self.ended = False

    def spawn(self, command: str) -> _Silent:
        return self.process

    def end(self, process: _Silent) -> None:
        self.ended = True
        process.returncode = -9
        process.feed.close()


def test_an_agent_that_moves_nothing_for_its_patience_is_ended_as_unreachable() -> None:
    link = _SilentLink(_Silent())
    with pytest.raises(HostUnreachable, match="moved nothing for 0.05s"):
        Agent(link, patience=0.05).ask({"survey": {}})
    assert link.ended


def test_an_agent_without_pipes_answers_nothing() -> None:
    """A process the platform gave no pipes is asked nothing and answers nothing."""
    process = _Silent(pipes=False)
    process.returncode = 0
    assert Agent(_SilentLink(process)).ask({"survey": {}}) == []


def test_a_payload_that_fails_wins_over_what_the_agent_then_says(tmp_path: Path) -> None:
    def failing(sink) -> None:
        sink.write(b"not a tar stream")
        raise MissionError("payload broke")

    agent = Agent(InProcessLink(), python=sys.executable)
    with pytest.raises(MissionError, match="payload broke"):
        agent.ask(_receive(tmp_path), payload=failing)


def test_a_payload_the_agent_stopped_reading_ends_quietly_and_the_refusal_is_what_is_said(
    tmp_path: Path,
) -> None:
    """The agent refuses the first entry it was not told of and closes its end of the pipe."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("unannounced.py")
        archive.addfile(info, io.BytesIO(b""))

    def flooding(sink) -> None:
        sink.write(stream.getvalue())
        for _ in range(256):
            sink.write(b"\0" * (1 << 20))

    agent = Agent(InProcessLink(), python=sys.executable)
    with pytest.raises(AgentRefused, match="unannounced entry"):
        agent.ask(_receive(tmp_path), payload=flooding)


def test_an_unreachable_host_and_a_missing_ssh_both_read_as_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ssh's own 255 with a transport phrase is a fact about the host, not the agent.

    The ssh that answers is this interpreter saying what ssh says of a name no resolver knows,
    so the link's own spawn runs and nothing leaves this machine.
    """
    spawn = subprocess.Popen
    said = "import sys; sys.stderr.write('ssh: Could not resolve hostname nowhere.invalid\\n')"

    def unresolved(argv: list[str], **options) -> subprocess.Popen[bytes]:
        assert argv[0] == "ssh" and "nowhere.invalid" in argv
        return spawn([sys.executable, "-c", f"{said}; sys.exit(255)"], **options)

    monkeypatch.setattr(subprocess, "Popen", unresolved)
    link = SshLink("nowhere.invalid", SshTransport(connect_timeout=2.0))
    assert link.ssh.destination("nowhere.invalid") == "nowhere.invalid"
    with pytest.raises(HostUnreachable, match="unreachable: .*resolve hostname"):
        Agent(link, patience=30.0).ask({"survey": {}})

    def absent(*args, **kwargs):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(subprocess, "Popen", absent)
    with pytest.raises(HostUnreachable, match="could not start"):
        SshLink("gold").spawn("true")


def test_ending_an_ssh_link_kills_and_reaps_its_process() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    SshLink("gold").end(process)
    assert process.returncode is not None


class _ShellLink(Link):
    """A target reached through this machine's own shell, the way ssh hands a command over."""

    def spawn(self, command: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def end(self, process: subprocess.Popen[bytes]) -> None:
        process.kill()
        process.wait()


def test_a_real_interpreter_runs_the_framed_agent_and_streams_at_a_measured_rate(
    tmp_path: Path,
) -> None:
    """The bootstrap passes a real shell unquoted, and a large file streams in bounded memory.

    Sixty-four megabytes of incompressible bytes and a thousand small files cross through a
    fresh interpreter, which is the throughput a local link measures without the network.
    """
    assert '"' not in BOOTSTRAP and "$" not in BOOTSTRAP
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, *(f"src/pkg/m{index}.py" for index in range(1000)))
    with (work / "src/big.bin").open("wb") as big:
        for _ in range(64):
            big.write(os.urandom(1 << 20))
    started = time.perf_counter()
    done = pushed(work, host, ["src"], link=_ShellLink("local"))
    elapsed = time.perf_counter() - started
    rate = done.bytes / elapsed / 1e6
    print(
        f"mirrored {done.sent} files, {done.bytes / 1e6:.1f} MB in {elapsed:.2f}s: {rate:.1f} MB/s"
    )
    assert done.sent == 1001 and rate > 1.0
    assert (host / "src/big.bin").stat().st_size == 64 << 20
