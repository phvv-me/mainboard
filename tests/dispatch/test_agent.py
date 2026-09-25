"""The standard-library agent: its rules, its walk, its memory, its lock and its two requests."""

import gzip
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pathspec
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.dispatch.agent import Digests, Rules, Scope, program, walk
from mainboard.dispatch.agent.program import DIRECTORY, FILE, LINK, Refusal, checked, containers
from mainboard.dispatch.sync import compiled

from .support import links_on_this_host

# Ignore lines of every shape the compiler has to carry: floating and anchored names, directory
# rules, wildcards that stop at a slash and ones that do not, and character classes.
_LINES = ["*.log", "build/", "/anchored", "a/**/b", "[!m]*.tmp", "doc?", "**/cache", "x/y"]
_PART = st.sampled_from(
    ["a", "b", "build", "anchored", "x", "y", "doc1", "m.tmp", "n.tmp", "k.log"]
)


@given(lines=st.lists(st.sampled_from(_LINES), min_size=1), parts=st.lists(_PART, min_size=1))
def test_compiled_rules_judge_a_path_exactly_as_the_ignore_library_does(
    lines: list[str], parts: list[str]
) -> None:
    """The target holds no ignore parser, so the regexes it is sent must decide identically.

    The same lines anchored one directory down decide the same paths beneath that directory.
    """
    path = "/".join(parts)
    spec = pathspec.GitIgnoreSpec.from_lines(lines)
    assert Rules({"": compiled(lines)}).matches(path, directory=False) == spec.match_file(path)
    nested = Rules({"deep": compiled(lines)})
    assert nested.matches(f"deep/{path}", directory=False) == spec.match_file(path)
    assert Rules.of(nested.spec()).matches(f"deep/{path}", directory=False) == spec.match_file(
        path
    )


def test_a_repository_starts_the_judgement_over_and_a_literal_path_names_one_tree() -> None:
    """A parent's `build/` never reaches a nested repository's own tracked `build/` sources."""
    rules = Rules({"": compiled(["build/"]), "pkg": compiled(["*.o"])}, repositories=["pkg"])
    assert rules.matches("build/x.rs", directory=False)
    assert not rules.matches("pkg/build/x.rs", directory=False)
    assert rules.matches("pkg/build/x.o", directory=False)
    assert rules.spec()["repositories"] == ["pkg"]
    literal = Rules(paths=["out/m[t]"])
    assert literal.matches("out/m[t]", directory=True)
    assert literal.matches("out/m[t]/row.json", directory=False)
    assert not literal.matches("out/mt", directory=False)
    discovered = Rules(discover=lambda base: (compiled(["*.o"]), base == "lib"))
    assert discovered.matches("lib/a.o", directory=False)
    assert discovered.spec() == {
        "patterns": {"": compiled(["*.o"]), "lib": compiled(["*.o"])},
        "paths": [],
        "repositories": ["lib"],
    }


@pytest.mark.parametrize(
    ("path", "windows", "safe"),
    [
        ("src/run.py", False, True),
        ("", False, False),
        ("src//run.py", False, False),
        ("./src", False, False),
        ("src/../../etc", False, False),
        ("dir\\name.py", False, True),
        ("dir\\..\\..\\escape", True, False),
        ("C:/escape", True, False),
        ("data/stream:ads", True, False),
        ("src/run.py", True, True),
    ],
)
def test_only_a_path_strictly_below_the_root_on_this_os_is_written(
    monkeypatch: pytest.MonkeyPatch, path: str, windows: bool, safe: bool
) -> None:
    """A backslash or a colon is a name on Linux and a way out of the root on Windows."""
    monkeypatch.setattr(program, "WINDOWS", windows)
    if safe:
        assert checked(path) == path
        return
    with pytest.raises(Refusal, match="unsafe path"):
        checked(path)


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["scripts", "mainboard.toml"], ["."]),
        (["research/compression", "research/bale"], [".", "research"]),
        (["a/b/c"], [".", "a", "a/b"]),
    ],
)
def test_every_directory_a_mirrored_snapshot_creates_is_one_it_fills_from_the_mirror(
    sources: list[str], expected: list[str]
) -> None:
    """A directory the copy invented holds none of the data dirs the mirror keeps beside it."""
    assert containers(sources) == expected


def seed(root: Path, *files: str) -> None:
    """Create each relative path under `root`, parents included, with its own name as content."""
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def test_a_walk_prunes_what_the_rules_claim_and_keeps_what_the_center_names(
    tmp_path: Path,
) -> None:
    """An ignored directory holding a kept file is entered, and nothing else inside it is taken."""
    seed(tmp_path, "src/run.py", "src/build/kept.rs", "src/build/out.o", "src/.cache/x", "skip.py")
    scope = Scope(
        ["src", "gone"],
        ignore=Rules({"": compiled(["build/", "*.py"])}),
        deny=Rules({"": compiled([".cache/"])}),
        keep=["src/build/kept.rs", "src/run.py"],
    )
    found = {entry.path: entry.kind for entry in walk(str(tmp_path), scope)}
    assert found == {
        "src": DIRECTORY,
        "src/build": DIRECTORY,
        "src/build/kept.rs": FILE,
        "src/run.py": FILE,
    }
    assert Scope.of(scope.spec()).keep == scope.keep


def test_a_listed_scope_states_its_files_and_the_directories_below_its_roots(
    tmp_path: Path,
) -> None:
    """Version control already named the files, so nothing is walked and a gone one is skipped.

    The deny rules still apply, and the ignore rules of every directory holding a listed file
    are read at once, so the spec a target prunes by carries them before any walk.
    """
    seed(tmp_path, "pkg/src/build/tracked.rs", "pkg/src/run.py", "pkg/.env", "pkg/dump/x.bin")
    (tmp_path / "pkg/fifo").mkdir()
    read: list[str] = []
    ignore = Rules(discover=lambda base: read.append(base) or (compiled(["build/"]), False))
    scope = Scope(
        ["pkg/src", "pkg/.env", "pkg/dump"],
        ignore=ignore,
        deny=Rules({"": compiled([".env", "dump/"])}),
        keep=["pkg/src/build/tracked.rs"],
        listed=[
            "pkg/.env",
            "pkg/dump/x.bin",
            "pkg/fifo",
            "pkg/src/build/tracked.rs",
            "pkg/src/gone.py",
            "pkg/src/run.py",
        ],
    )
    assert sorted(read) == ["", "pkg", "pkg/src", "pkg/src/build"]
    found = [(entry.path, entry.kind) for entry in walk(str(tmp_path), scope)]
    assert found == [
        ("pkg/src", DIRECTORY),
        ("pkg/src/build", DIRECTORY),
        ("pkg/src/build/tracked.rs", FILE),
        ("pkg/src/run.py", FILE),
    ]
    target = Scope.of(scope.spec())
    assert not target.excluded("pkg/src/build", directory=True)
    assert target.excluded("pkg/src/build/out.o", directory=False)


@links_on_this_host
def test_a_walk_carries_links_as_links_unless_it_follows_them_once(tmp_path: Path) -> None:
    """Following a tree of links never loops on a cycle nor stops at a link to nothing."""
    seed(tmp_path, "outside/lib/mod.py", "tree/plain.py", "listed/real.py")
    (tmp_path / "tree/lib").symlink_to(tmp_path / "outside/lib", target_is_directory=True)
    (tmp_path / "tree/again").symlink_to(tmp_path / "tree", target_is_directory=True)
    (tmp_path / "tree/dangling").symlink_to(tmp_path / "nowhere")
    (tmp_path / "listed/alias.py").symlink_to("real.py")
    (tmp_path / "tree/skipped.log").symlink_to("plain.py")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "tree/pipe")
    logs = Rules({"": compiled(["*.log"])})
    carried = {
        entry.path: entry.kind for entry in walk(str(tmp_path), Scope(["tree"], ignore=logs))
    }
    assert carried["tree/lib"] == LINK and carried["tree/dangling"] == LINK
    assert "tree/skipped.log" not in carried and "tree/pipe" not in carried
    followed = {
        entry.path: entry.kind for entry in walk(str(tmp_path), Scope(["tree"], follow=True))
    }
    assert followed == {
        "tree": DIRECTORY,
        "tree/lib": DIRECTORY,
        "tree/lib/mod.py": FILE,
        "tree/plain.py": FILE,
        "tree/skipped.log": FILE,
    }
    listed = Scope(["listed"], listed=["listed/alias.py", "listed/real.py"])
    stated = {entry.path: (entry.kind, entry.detail) for entry in walk(str(tmp_path), listed)}
    assert stated["listed/alias.py"] == (LINK, "real.py")


def test_a_digest_is_read_once_until_the_file_moves_and_a_torn_memory_starts_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "data.bin"
    source.write_bytes(b"first")
    memory = tmp_path / "state/digests.json"
    digests = Digests(str(memory))
    first = digests.of(str(source), key="data.bin")
    digests.of(str(source), key="other.bin")
    digests.seen.discard("other.bin")
    digests.save(prune=True)
    reads: list[str] = []
    original = program.digest
    monkeypatch.setattr(program, "digest", lambda path: reads.append(path) or original(path))
    again = Digests(str(memory))
    assert set(again.held) == {"data.bin"}
    assert again.of(str(source), key="data.bin") == first and reads == []
    source.write_bytes(b"second")
    assert again.of(str(source), key="data.bin") != first and reads == [str(source)]
    memory.write_text("{", encoding="utf-8")
    assert Digests(str(memory)).held == {}


@pytest.mark.skipif(sys.platform == "win32", reason="the kernel lock here is Windows' own")
def test_the_posix_lock_is_held_for_the_block_and_released_after(tmp_path: Path) -> None:
    lock = tmp_path / "held.lock"
    with program.locked(str(lock)):
        assert lock.exists()
    with program.locked(str(lock)):
        pass


def test_the_windows_lock_waits_out_a_busy_region_and_unlocks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`LK_LOCK` gives up after ten seconds, and a lock that gave up is asked again."""
    calls: list[int] = []

    def locking(descriptor: int, mode: int, length: int) -> None:
        del descriptor, length
        calls.append(mode)
        if len(calls) == 1:
            raise OSError("region busy")

    msvcrt = SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    monkeypatch.setattr(program, "WINDOWS", True)
    with program.locked(str(tmp_path / "held.lock")):
        assert calls == [1, 1]
    assert calls == [1, 1, 0]


def ran(request: dict[str, dict[str, object]], payload: bytes = b"") -> tuple[int, list, str]:
    """Run one request through `run` in this process, answering its status, records and stderr."""
    stdin = io.BytesIO(json.dumps(request).encode() + b"\n" + payload)
    stdout, stderr = io.BytesIO(), io.StringIO()
    code = program.run(stdin, stdout, stderr)
    records = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return code, records, stderr.getvalue()


def packed(**files: bytes) -> bytes:
    """A gzip tar holding `files`, what a center streams behind a receive request."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_a_survey_makes_a_fresh_root_and_states_only_what_is_there(tmp_path: Path) -> None:
    """A rental holds no workspace yet, and a named path that is not a file is not described."""
    root = tmp_path / "fresh"
    seed(root, "named/file.txt")
    (root / "named/folder").mkdir()
    code, records, _ = ran(
        {
            "survey": {
                "root": str(root),
                "state": ".mainboard/dispatch",
                "scopes": [],
                "named": ["named/file.txt", "named/folder", "named/absent.txt"],
            }
        }
    )
    capabilities, *entries = records
    assert code == 0 and set(capabilities) == {"links", "modes", "fold"}
    assert capabilities["links"] is (not program.WINDOWS)
    assert [entry[0] for entry in entries] == ["named/file.txt"]
    assert (root / ".mainboard/dispatch/digests.json").is_file()


def test_a_file_that_vanishes_before_its_hash_is_not_described(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(tmp_path, "src/gone.py")

    def vanished(self: Digests, path: str, *, key: str) -> str:
        raise FileNotFoundError(path)

    monkeypatch.setattr(Digests, "of", vanished)
    request = {
        "root": str(tmp_path),
        "state": "state",
        "scopes": [Scope(["src"]).spec()],
        "named": ["src/gone.py"],
    }
    _, records, _ = ran({"survey": request})
    assert [record[0] for record in records[1:]] == ["src"]


def test_a_receive_prunes_deepest_first_and_keeps_a_directory_that_still_holds_something(
    tmp_path: Path,
) -> None:
    seed(tmp_path, "old/a.py", "held/ignored.log", "src/stale.py")
    code, records, said = ran(
        {
            "receive": {
                "root": str(tmp_path),
                "state": "state",
                "delete": ["old", "old/a.py", "held", "src/stale.py", "never/there.py"],
                "directories": ["empty/dir"],
                "links": {},
                "files": {"src/new.py": [5_000_000_000, True]},
            }
        },
        packed(**{"src/new.py": b"print()\r\n"}),
    )
    assert (code, said) == (0, "")
    assert records == [
        {
            "written": 1,
            "bytes": 9,
            "deleted": ["old/a.py", "src/stale.py", "old"],
            "kept": ["held"],
        }
    ]
    placed = tmp_path / "src/new.py"
    assert placed.read_bytes() == b"print()\r\n"
    assert placed.stat().st_mtime_ns == 5_000_000_000
    assert (tmp_path / "empty/dir").is_dir()


@pytest.mark.parametrize(
    ("files", "stream", "refusal"),
    [
        ({}, {"src/new.py": b"x"}, "unannounced entry"),
        ({"../escape.py": [0, False]}, {"../escape.py": b"x"}, "unsafe path"),
        ({"blocked": [0, False]}, {"blocked": b"x"}, "could not place"),
        ({"blocked/inside.txt/x": [0, False]}, {"blocked/inside.txt/x": b"x"}, "could not place"),
    ],
)
def test_a_receive_refuses_what_it_was_not_told_or_cannot_place(
    tmp_path: Path, files: dict[str, list[object]], stream: dict[str, bytes], refusal: str
) -> None:
    """A refusal is one line and status 3, and nothing half written is left beside its name."""
    seed(tmp_path, "blocked/inside.txt")
    request = {
        "root": str(tmp_path),
        "state": "state",
        "delete": [],
        "directories": [],
        "links": {},
        "files": files,
    }
    code, records, said = ran({"receive": request}, packed(**stream))
    assert (code, records) == (3, [])
    assert said.startswith("mainboard: ") and refusal in said
    assert sorted(path.name for path in tmp_path.iterdir()) == ["blocked", "state"]
    assert [path.name for path in (tmp_path / "blocked").iterdir()] == ["inside.txt"]


def test_a_receive_that_meets_a_torn_stream_fails_loudly(tmp_path: Path) -> None:
    """A stream cut short is not a refusal the agent chose, so it escapes as itself."""
    request = {
        "root": str(tmp_path),
        "state": "state",
        "delete": [],
        "directories": [],
        "links": {},
        "files": {"a": [0, False]},
    }
    with pytest.raises((EOFError, tarfile.ReadError, gzip.BadGzipFile)):
        ran({"receive": request}, packed(a=b"payload")[:20])


@links_on_this_host
def test_a_receive_makes_links_where_it_was_told(tmp_path: Path) -> None:
    code, _, _ = ran(
        {
            "receive": {
                "root": str(tmp_path),
                "state": "state",
                "delete": [],
                "directories": [],
                "links": {"deep/alias": "../target.txt"},
                "files": {},
            }
        },
        packed(),
    )
    assert code == 0 and os.readlink(tmp_path / "deep/alias") == "../target.txt"
