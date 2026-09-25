import io
import json
import os
import stat
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.center import remote
from mainboard.center.state import Parcel, packed
from mainboard.probe.census import Census

from ..git.conftest import Workspace
from ..strategies import WORDS

# The owned submodule of the fixture tree, as `[parent, path]` the way the migration names it.
_LIB = [[".", "packages/lib"]]

# A scripted command table: what a command line starting with each prefix answers.
type Answers = Mapping[tuple[str, ...], tuple[int, str]]

posix_modes = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows keeps no POSIX permission bits to read back"
)


class Runner:
    """A scripted `_run`: each command answered by the first matching prefix, all kept.

    answers: what a command starting with each prefix answers, anything else succeeding silently.
    """

    def __init__(self, answers: Answers) -> None:
        self.answers = dict(answers)
        self.ran: list[tuple[tuple[str, ...], str]] = []

    def __call__(self, command: Sequence[str], stdin: str = "") -> tuple[int, str]:
        self.ran.append((tuple(command), stdin))
        return next(
            (
                answer
                for prefix, answer in self.answers.items()
                if tuple(command[: len(prefix)]) == prefix
            ),
            (0, ""),
        )

    @property
    def commands(self) -> list[tuple[str, ...]]:
        """Every command line run, without what it was fed."""
        return [command for command, _ in self.ran]


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> Callable[[Answers], Runner]:
    """Install a scripted `_run` answering from the given table, and hand it back to read."""

    def install(answers: Answers) -> Runner:
        runner = Runner(answers)
        monkeypatch.setattr(remote, "_run", runner)
        return runner

    return install


def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the agent believe it runs on Windows, without touching the real `os` module."""
    monkeypatch.setattr(remote, "os", SimpleNamespace(name="nt", environ={"USERNAME": "me"}))


def fed(monkeypatch: pytest.MonkeyPatch, stream: bytes) -> None:
    """Hand `stream` to the agent as the stdin ssh would have carried it."""
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(stream)))


def tarred(members: Mapping[str, bytes | None]) -> bytes:
    """A tar holding each member as a file, or as a directory where its content is None."""
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w") as archive:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            if content is None:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return sink.getvalue()


def origin(tree: Workspace) -> str:
    """The root's remote as the center's own git spells it, which is what a migration passes."""
    return tree.git(tree.path, "remote", "get-url", "origin")


def test_where_answers_the_root_and_home_absolute_in_this_machines_spelling(home: Path) -> None:
    """A `~` root lands under the destination's own home, and the separator comes with it."""
    assert remote.where("~/projects") == {
        "root": str(home / "projects"),
        "home": str(home),
        "separator": os.sep,
    }
    assert Path(str(remote.where("relative")["root"])).is_absolute()


def test_census_is_the_shared_census_measured_at_the_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent asks the same census `facts` runs, so a machine reads the same either way."""
    monkeypatch.setattr(Census, "survey", lambda self, root: {"measured": root})
    assert remote.census("/work") == {"measured": "/work"}


def test_inventory_names_every_parcel_the_destination_lacks_byte_for_byte(home: Path) -> None:
    """Content fingerprints compare bytes and tree stamps compare time and size; absent is lacking.

    A second migration skips everything already moved, which is only safe if a stale or missing
    copy is always named, whichever fingerprint kind it was sent under.
    """
    root = home / "projects"
    (root / "d").mkdir(parents=True)
    (root / "d" / "same").write_bytes(b"kept")
    (root / "d" / "stale").write_bytes(b"old")
    (root / "d" / "stamped").write_bytes(b"12345")
    os.utime(root / "d" / "stamped", (1_000, 1_000))
    rows = {
        "same": Parcel(anchor="root", path="d/same", data=b"kept").fingerprint,
        "stale": Parcel(anchor="root", path="d/stale", data=b"new").fingerprint,
        "absent": "sha256:00",
        "stamped": "mtime:1000:5",
        "restamped": "mtime:2000:5",
        "gone": "mtime:1000:5",
    }
    paths = {"restamped": "d/stamped", "gone": "d/gone"}
    parcels = [
        {"key": key, "anchor": "root", "path": paths.get(key, f"d/{key}"), "fingerprint": stamp}
        for key, stamp in rows.items()
    ]
    assert remote.inventory(str(root), parcels) == ["stale", "absent", "restamped", "gone"]


@given(
    st.dictionaries(WORDS, st.tuples(st.binary(max_size=64), st.booleans()), max_size=5),
)
def test_placed_parcels_land_whole_and_a_second_pass_changes_nothing(
    tmp_path: Path, contents: dict[str, tuple[bytes, bool]]
) -> None:
    """Whatever `packed` streams, `place` writes back byte for byte, and then it is idempotent.

    After a placement the inventory finds nothing lacking, under both fingerprint kinds, and a
    repeated placement writes nothing and backs nothing up, which is what makes a migration safe
    to run twice.
    """
    here = Path(mkdtemp(dir=tmp_path))
    root = here / "destination"
    for name, (data, _) in contents.items():
        (here / name).write_bytes(data)
    parcels = [
        Parcel(anchor="root", path=f"tree/{name}", source=here / name)
        if from_file
        else Parcel(anchor="root", path=f"made/{name}.bin", data=data)
        for name, (data, from_file) in contents.items()
    ]
    stream = b"".join(packed(parcels))
    for written in (len(parcels), 0):
        with pytest.MonkeyPatch.context() as patch:
            fed(patch, stream)
            assert remote.place(str(root), []) == {"written": written}
    assert remote.inventory(str(root), [parcel.listing() for parcel in parcels]) == []
    assert {path.name for path in root.rglob("*") if path.is_file()} == {
        Path(parcel.path).name for parcel in parcels
    }
    assert all(
        (root / parcel.path).read_bytes() == contents[Path(parcel.path).stem][0]
        for parcel in parcels
    )


def test_place_keeps_a_differing_original_once(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever the destination held is never lost, however many migrations reach it.

    The first differing write moves the original aside; a later one replaces only the new copy,
    so the backup always holds what the machine had before any migration touched it. A file the
    destination never had needs no backup at all.
    """
    root = home / "projects"
    (root / ".env").parent.mkdir(parents=True)
    (root / ".env").write_bytes(b"theirs")
    for data in (b"ours", b"ours again"):
        fed(monkeypatch, tarred({"root/.env": data, "home/.ssh/id_gold": b"key"}))
        remote.place(str(root), [])
        assert (root / ".env").read_bytes() == data
    assert (root / f".env{remote.BACKUP}").read_bytes() == b"theirs"
    assert not (home / ".ssh" / f"id_gold{remote.BACKUP}").exists()


@posix_modes
def test_credentials_are_written_readable_by_their_owner_alone(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ssh refuses a private key others can read, and a merged login file is a credential too."""
    fed(monkeypatch, tarred({"home/.ssh/id_gold": b"key"}))
    remote.place(str(home / "projects"), ["home/.ssh/id_gold"])
    remote.merge(".claude.json", "projects", {"/w": {}})
    for secret in (home / ".ssh" / "id_gold", home / ".claude.json"):
        assert stat.S_IMODE(secret.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "name",
    ["root/../escaped", "home/.ssh/../../escaped", "root//etc/passwd", "elsewhere/file"],
)
def test_place_refuses_a_member_outside_its_anchor(
    home: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """A shipped name is data from the wire: it may never climb out of the root or home."""
    fed(monkeypatch, tarred({name: b"x"}))
    with pytest.raises(ValueError, match="anchor"):
        remote.place(str(home / "projects"), [])
    assert not (home.parent / "escaped").exists()


def test_place_skips_members_that_are_not_files(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory entry carries no bytes, so it is neither written nor counted."""
    fed(monkeypatch, tarred({"root/tree": None, "root/tree/file": b"x"}))
    assert remote.place(str(home / "projects"), []) == {"written": 1}


def test_clone_checks_out_the_root_and_owned_submodules_and_a_rerun_changes_nothing(
    tree: Workspace, tmp_path: Path
) -> None:
    """An empty root becomes the center's checkout on its branch; a second run is a no-op.

    The branch tracks its remote so the new center pulls and pushes the way the old one did,
    and the owned library arrives at the pointer the root records.
    """
    commit = tree.head(tree.path)
    base = tmp_path / "destination"
    base.mkdir()
    done = [
        {"repo": ".", "outcome": "done", "detail": f"main at {commit[:12]}"},
        {"repo": "packages/lib", "outcome": "done", "detail": "at its pointer"},
    ]
    for _ in range(2):
        assert remote.clone(str(base), origin(tree), "main", commit, _LIB) == done
    assert tree.head(base) == commit
    assert tree.git(base, "rev-parse", "--abbrev-ref", "main@{upstream}") == "origin/main"
    assert tree.head(base / "packages" / "lib") == tree.head(tree.lib)
    assert (base / "packages" / "lib" / "lib.txt").is_file()


def _another(tree: Workspace, base: Path, commit: str) -> str:
    """A root that is some other repository's checkout."""
    tree.forge.clone(tree.forge.url("other", "ref"), base)
    return ""


def _occupied(tree: Workspace, base: Path, commit: str) -> str:
    """A root that already holds files git did not put there."""
    base.mkdir()
    (base / "notes.txt").write_text("mine", encoding="utf-8")
    return ""


def _ahead(tree: Workspace, base: Path, commit: str) -> str:
    """A checkout of this repository holding a commit the center's HEAD lacks."""
    remote.clone(str(base), origin(tree), "main", commit, [])
    return tree.forge.commit(base, "local work", {"local.txt": "only here\n"})


@pytest.mark.parametrize(
    ("occupy", "said"),
    [
        (_another, "is another repository"),
        (_occupied, "is not empty"),
        (_ahead, "holds commits"),
    ],
)
def test_clone_holds_a_root_it_did_not_make_and_leaves_it_as_it_was(
    tree: Workspace,
    tmp_path: Path,
    occupy: Callable[[Workspace, Path, str], str],
    said: str,
) -> None:
    """Nothing the destination already keeps is overwritten or reset away; it is named instead.

    The held row is the only one, so no submodule is touched under a root the clone refused.
    """
    base = tmp_path / "destination"
    commit = tree.head(tree.path)
    kept = occupy(tree, base, commit)
    steps = remote.clone(str(base), origin(tree), "main", commit, _LIB)
    assert steps == [{"repo": ".", "outcome": "held", "detail": steps[0]["detail"]}]
    assert said in steps[0]["detail"]
    if kept:
        assert tree.head(base) == kept


def test_clone_detaches_without_a_branch_and_reports_each_failure_as_its_own_row(
    tree: Workspace, tmp_path: Path
) -> None:
    """A detached center stays detached, and a clone, checkout or submodule failure is a row.

    Each failure carries git's last words, so the report says what went wrong on that machine
    rather than that something did.
    """
    commit = tree.head(tree.path)
    base = tmp_path / "detached"
    steps = remote.clone(str(base), origin(tree), "", commit, [[".", "packages/absent"]])
    assert steps[0] == {"repo": ".", "outcome": "done", "detail": f"detached at {commit[:12]}"}
    assert tree.git(base, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert steps[1]["repo"] == "packages/absent" and steps[1]["outcome"] == "failed"
    assert steps[1]["detail"]
    nowhere = (tmp_path / "nowhere.git").as_uri()
    failed = remote.clone(str(tmp_path / "unreached"), nowhere, "main", commit, _LIB)
    assert [row["outcome"] for row in failed] == ["failed"]
    unknown = remote.clone(str(tmp_path / "fresh"), origin(tree), "main", "0" * 40, _LIB)
    assert [row["outcome"] for row in unknown] == ["failed"]
    assert unknown[0]["detail"]


@pytest.mark.parametrize(
    ("lfs", "system", "expected"),
    [
        (0, "posix", [("git", "lfs", "version"), ("git", "lfs", "install", "--skip-repo")]),
        (127, "posix", [("git", "lfs", "version")]),
        (
            127,
            "nt",
            [("git", "lfs", "version"), ("git", "config", "--global", "core.longpaths", "true")],
        ),
    ],
)
def test_clone_prepares_lfs_when_present_and_long_paths_on_windows(
    run: Callable[[Answers], Runner],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfs: int,
    system: str,
    expected: list[tuple[str, ...]],
) -> None:
    """Large files check out as files only with lfs's filters, and deep paths need longpaths.

    A checkout that fails silently still fails, and its row carries whatever git managed to say,
    even when that was nothing.
    """
    runner = run({("git", "lfs", "version"): (lfs, ""), ("git", "-C"): (1, "")})
    if system == "nt":
        windows(monkeypatch)
    steps = remote.clone(str(tmp_path / "base"), "forge:x.git", "main", "abc", _LIB)
    assert steps == [{"repo": ".", "outcome": "failed", "detail": ""}]
    assert runner.commands[: len(expected)] == expected


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (
            {("gh", "auth", "login"): (1, "token refused")},
            {"signed": False, "detail": "token refused"},
        ),
        (
            {("gh", "auth", "setup-git"): (1, "no git on PATH")},
            {"signed": True, "detail": "no git on PATH"},
        ),
        ({}, {"signed": True, "detail": ""}),
    ],
)
def test_login_hands_the_token_to_gh_on_stdin_and_says_how_far_it_got(
    run: Callable[[Answers], Runner],
    answers: Answers,
    expected: dict[str, str | bool],
) -> None:
    """The token rides stdin and nothing else, and a refused login never wires git to it."""
    runner = run(answers)
    assert remote.login("ghp_secret") == expected
    assert runner.ran[0] == (("gh", "auth", "login", "--with-token"), "ghp_secret")
    assert all("ghp_secret" not in " ".join(command) for command in runner.commands)
    assert len(runner.ran) == (1 if not expected["signed"] else 2)


def test_merge_sets_entries_under_a_key_and_writes_only_when_something_changed(
    home: Path,
) -> None:
    """A tool's own settings on the destination survive, and an unchanged merge leaves the file.

    Only the members named move, the document's other keys and the key's other members stay,
    and a merge that would change nothing does not even rewrite the file.
    """
    target = home / ".config" / "tool" / "state.json"
    assert remote.merge(".config/tool/state.json", "projects", {"/w": {"trust": True}}) == {
        "changed": 1
    }
    target.write_text(
        json.dumps({"account": "mine", "projects": {"/w": {"trust": True}, "/old": {}}}),
        encoding="utf-8",
    )
    os.utime(target, (1_000, 1_000))
    assert remote.merge(".config/tool/state.json", "projects", {"/w": {"trust": True}}) == {
        "changed": 0
    }
    assert target.stat().st_mtime == 1_000
    assert remote.merge(".config/tool/state.json", "projects", {"/w": {"trust": False}}) == {
        "changed": 1
    }
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "account": "mine",
        "projects": {"/w": {"trust": False}, "/old": {}},
    }
    assert [path.name for path in target.parent.iterdir()] == ["state.json"]


def test_restrict_on_windows_also_strips_inherited_access(
    run: Callable[[Answers], Runner],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows ignores the mode bits, so ssh accepts a key only after its ACL names one user."""
    key = tmp_path / "id_gold"
    key.write_bytes(b"key")
    runner = run({})
    windows(monkeypatch)
    remote._restrict(key)
    assert runner.commands == [("icacls", str(key), "/inheritance:r", "/grant:r", "me:F")]


def test_run_answers_status_and_output_and_127_for_a_program_that_cannot_start(
    tmp_path: Path,
) -> None:
    """What is fed arrives on stdin, and a missing tool reads as a missing tool, not a crash."""
    echo = (sys.executable, "-c", "import sys; print(sys.stdin.read().upper())")
    assert remote._run(echo, stdin="token") == (0, "TOKEN\n")
    missing = str(tmp_path / "absent-tool")
    assert remote._run((missing, "--version")) == (127, f"{missing} could not run")
