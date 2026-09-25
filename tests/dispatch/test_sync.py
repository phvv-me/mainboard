import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch import GitignoreFilter, SyncLock
from mainboard.dispatch import sync as sync_module
from mainboard.dispatch.agent import Agent, Rules
from mainboard.dispatch.mirror import Mirror
from mainboard.dispatch.shared import STATE_DIR
from mainboard.dispatch.sync import ALWAYS_EXCLUDE, CARD_LEASES, patterns
from mainboard.dispatch.transport import Endpoint

from .support import InProcessLink

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git on this machine")


def seed(root: Path, *files: str) -> None:
    """Create each relative path under `root` (parents included) with its own name as content."""
    write(root, {relative: relative for relative in files})


def write(root: Path, files: dict[str, str]) -> None:
    """Create each relative path under `root` (parents included) holding its given text."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private home whose git reads no configuration but the one a test writes."""
    home = tmp_path / "home"
    home.mkdir()
    for variable in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    return home


def git(root: Path, *arguments: str) -> None:
    """One git command in `root` under a throwaway identity."""
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
        check=True,
        capture_output=True,
    )


def test_the_denylist_covers_git_env_and_every_generated_directory() -> None:
    assert (
        ".git",
        ".env",
        f"{STATE_DIR}/",
        ".mainboard/",
        ".pixi/",
        "__pycache__/",
        "*/evidence/artifacts/***",
        "*/evidence/receipts/***",
    ) == ALWAYS_EXCLUDE
    assert CARD_LEASES == (".card.lock", ".card.lock.*")


@pytest.mark.parametrize(
    ("pattern", "path", "directory", "claimed"),
    [
        ("data/raw", "deep/data/raw", True, True),
        ("/data/raw", "deep/data/raw", True, False),
        ("/data/raw", "data/raw/x.bin", False, True),
        ("results*/***", "a/results-1", True, True),
        ("results*/***", "a/results-1/e.json", False, True),
        ("*/evidence/receipts/***", "p/evidence/receipts/run=1/part.parquet", False, True),
        ("*/evidence/receipts/***", "evidence/receipts/x", False, False),
        ("**/datasets/experiments/***", "r/datasets/experiments/n/x", False, True),
        ("references", "research/compression/references", True, True),
        ("/research/**/datasets/", "research/x/datasets/a.csv", False, True),
        ("*.parquet", "a/b.parquet", False, True),
        (".card.lock.*", "src/.card.locked", False, False),
        (".card.lock.*", "src/.card.lock.0", False, True),
    ],
)
def test_a_filter_pattern_is_anchored_only_by_a_leading_slash(
    pattern: str, path: str, directory: bool, claimed: bool
) -> None:
    """`dir/***` names a directory and all of it; a slash inside floats unless it leads."""
    assert patterns([pattern]).matches(path, directory=directory) is claimed


def test_sync_transactions_reenter_the_same_endpoint_and_separate_ports(tmp_path: Path) -> None:
    first = Endpoint(address="node", port=22001, user="runner")
    other = first.model_copy(update={"port": 22002})
    with SyncLock(first, tmp_path) as outer:
        with SyncLock(first, tmp_path) as inner, SyncLock(other, tmp_path) as separate:
            assert inner.lock is outer.lock
            assert separate.path != outer.path
            assert separate.lock.is_locked and outer.lock.is_locked
        assert outer.lock.is_locked
    assert (
        SyncLock(Endpoint(address="node", port=0), tmp_path).path
        == SyncLock(Endpoint(address="node", port=22), tmp_path).path
    )


def test_the_sync_lock_releases_its_file_however_the_mirror_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SyncLock("gold", tmp_path) as lock:
        assert lock.lock.is_locked
        assert lock.path.is_file()
    assert not lock.lock.is_locked
    lock.__exit__(None, None, None)
    (tmp_path / "mainboard.toml").touch()
    monkeypatch.chdir(tmp_path)
    assert SyncLock("gold").path == tmp_path / STATE_DIR / "locks" / lock.path.name

    def refuse(*, timeout: float | None = None, poll_interval: float = 0.05) -> None:
        del timeout, poll_interval
        raise OSError("lock unavailable")

    blocked = SyncLock("gold", tmp_path)
    monkeypatch.setattr(blocked.lock, "acquire", refuse)
    with pytest.raises(OSError, match="lock unavailable"):
        blocked.__enter__()
    assert not blocked.lock.is_locked


def test_the_ignore_files_are_read_once_each_and_a_repository_answers_to_its_own(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without git the files are read directly, and a nested repository stops a parent's rules.

    Its `info/exclude` joins its root, through the `.git` file a submodule carries as well.
    """
    monkeypatch.setattr(sync_module.shutil, "which", lambda name: None)
    write(
        tmp_path,
        {
            ".gitignore": "*.scratch\nbuild/\n",
            "pkg/.gitignore": "!keep.scratch\n",
            "lib/.git/info/exclude": "*.local\n",
            "sub/.git": "gitdir: ../modules/sub\n",
            "modules/sub/info/exclude": "*.tmp\n",
        },
    )
    reads: list[Path] = []
    original = sync_module._read
    monkeypatch.setattr(sync_module, "_read", lambda path: reads.append(path) or original(path))
    ignores = GitignoreFilter(tmp_path)
    assert ignores.ignored(tmp_path / "a.scratch")
    assert not ignores.ignored("pkg/keep.scratch")
    assert ignores.ignored("build/output.txt")
    assert not ignores.ignored("lib/build/output.txt")
    assert ignores.ignored("lib/x.local") and ignores.ignored("sub/x.tmp")
    assert not ignores.ignored("/somewhere/else/file.txt")
    assert ignores.ignored("pkg/b.scratch") and ignores.ignored("pkg/c.scratch")
    assert len(reads) == len(set(reads))
    assert ignores.rules.spec()["repositories"] == ["lib", "sub"]
    monkeypatch.chdir(tmp_path)
    assert GitignoreFilter().root == tmp_path


def test_the_files_a_repository_would_ship_count_a_link_that_leads_to_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sync_module.shutil, "which", lambda name: None)
    seed(tmp_path, "src/run.py", "src/.env", "src/__pycache__/x.pyc", "src/data.scratch")
    write(tmp_path, {".gitignore": "*.scratch\n"})
    try:
        (tmp_path / "src/alias.py").symlink_to("run.py")
        (tmp_path / "src/folder").symlink_to("..", target_is_directory=True)
    except OSError as fault:
        pytest.skip(f"this account cannot create symlinks: {fault}")
    assert GitignoreFilter(tmp_path).files("src") == ["src/alias.py", "src/run.py"]


@needs_git
def test_a_submodule_ships_its_tracked_build_sources_under_a_parent_that_ignores_build(
    tmp_path: Path, home: Path
) -> None:
    """The failure this rule exists for: the monorepo's `build/` once dropped mcmr's sources.

    Each repository answers for itself: what it tracks ships whatever any ignore file says, its
    untracked files are filtered by its own ignore files alone, and a nested clone nobody
    registered answers the same way. A host keeps what the repository's own rules ignore and
    loses what they do not.
    """
    write(
        home,
        {".gitconfig": "[core]\n\texcludesFile = ~/global-ignore\n", "global-ignore": "*.swp\n"},
    )
    work, host = tmp_path / "work", tmp_path / "host"
    seed(work, "src/app.py", "build/out.o", "build/forced.txt", "src/x.swp")
    write(work, {".gitignore": "build/\n"})
    git(work, "init", "-q")
    mcmr = work / "packages/mcmr"
    seed(mcmr, "src/graph/build/building.rs", "target/debug/x")
    write(mcmr, {".gitignore": "target/\n"})
    git(mcmr, "init", "-q")
    git(mcmr, "add", ".")
    git(mcmr, "commit", "-q", "-m", "mcmr")
    seed(mcmr, "src/graph/build/fresh.rs")
    clone = work / "vendor/clone"
    seed(clone, "build/keep.rs")
    git(clone, "init", "-q")
    git(work, "add", ".gitignore", "src/app.py", "packages/mcmr")
    git(work, "add", "-f", "build/forced.txt")
    git(work, "update-index", "--add", "--cacheinfo", f"160000,{'0' * 39}1,packages/ghost")
    git(work, "commit", "-q", "-m", "workspace")
    ignores = GitignoreFilter(work)
    roots = ["src", "build", "packages", "vendor"]
    listing = ignores.tracked(roots)
    assert listing.files == (
        "build/forced.txt",
        "packages/mcmr/.gitignore",
        "packages/mcmr/src/graph/build/building.rs",
        "packages/mcmr/src/graph/build/fresh.rs",
        "src/app.py",
        "vendor/clone/build/keep.rs",
    )
    assert listing.kept == ("build/forced.txt",)
    assert ignores.tracked(["packages/mcmr/src"]).files == (
        "packages/mcmr/src/graph/build/building.rs",
        "packages/mcmr/src/graph/build/fresh.rs",
    )
    assert ignores.tracked(["elsewhere"]).files == ()
    assert list(ignores.files("packages")) == list(ignores.tracked(["packages"]).files)
    assert (
        "vendor/clone/build/keep.rs"
        not in ignores.tracked(roots, deny=patterns(["vendor/"])).files
    )
    git(work, "add", "src/x.swp", "-f")
    assert "src/x.swp" in ignores.tracked(["src"]).files
    seed(host, "packages/mcmr/target/debug/bin", "packages/mcmr/src/graph/build/gone.rs")
    seed(host, "build/cache.o", "src/editor.swp")
    agent = Agent(InProcessLink(), python=sys.executable)
    deny = patterns([*ALWAYS_EXCLUDE, *CARD_LEASES])
    done = Mirror(work, agent).push(
        str(host), scopes=[ignores.scope(roots, deny=deny)], protected=Rules()
    )
    assert (host / "packages/mcmr/src/graph/build/building.rs").is_file()
    assert (host / "build/forced.txt").is_file() and not (host / "build/out.o").exists()
    assert (host / "packages/mcmr/target/debug/bin").is_file()
    assert (host / "build/cache.o").is_file() and (host / "src/editor.swp").is_file()
    assert done.deleted == ("packages/mcmr/src/graph/build/gone.rs",)


@needs_git
def test_a_repository_git_cannot_read_refuses_by_what_git_said(tmp_path: Path, home: Path) -> None:
    write(tmp_path, {".git": "not a gitdir\n"})
    with pytest.raises(MissionError, match="git could not list"):
        GitignoreFilter(tmp_path).tracked(["src"])


def test_a_workspace_git_does_not_answer_for_lists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert GitignoreFilter(tmp_path).tracked(["src"]) is None
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(sync_module.shutil, "which", lambda name: None)
    assert GitignoreFilter(tmp_path).tracked(["src"]) is None


@needs_git
def test_the_global_excludes_default_to_the_xdg_file_when_git_names_none(
    tmp_path: Path, home: Path
) -> None:
    write(home, {".config/git/ignore": "*.orig\n"})
    git(tmp_path, "init", "-q")
    assert GitignoreFilter(tmp_path).ignored("patch.orig")
