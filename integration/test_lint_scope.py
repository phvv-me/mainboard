"""What a lint pass reads and what it answers for.

A pass reads the repositories the workspace owns and nothing pinned from elsewhere, and on
2026-10-07 `mb lint .` recursed forever into a submodule never initialized, which git lists as
the directory itself. A check that reads its whole owner (ty, pyrefly, mcmr) answers for the
files the pass reads and no others, so linting a few files is red only for those files.
"""

from pathlib import Path

from mainboard.core.project import Project
from mainboard.lint import Inventory, Linter
from mainboard.lint.process import TIMED_OUT, Outcome
from mainboard.manifest.loading import load
from mainboard.manifest.schema.git import GitPolicy

from .conftest import committed, git


def test_a_pass_reads_owned_repositories_and_skips_an_uninitialized_one(
    tracked: Path, tmp_path_factory
) -> None:
    # Beside the root's own remote is owned; anywhere else is someone else's.
    remote = Path(git(tracked, "remote", "get-url", "origin").stdout.strip())
    owned = committed(remote.parent / "owned")
    absent = committed(remote.parent / "absent")
    foreign = committed(tmp_path_factory.mktemp("elsewhere") / "foreign")
    for repository, place in ((owned, "owned"), (absent, "absent"), (foreign, "foreign")):
        git(tracked, "submodule", "add", "--quiet", repository.as_posix(), place)
    git(tracked, "commit", "--quiet", "-m", "three submodules")
    git(tracked, "submodule", "deinit", "--quiet", "--force", "absent")

    listed = Inventory(tracked, GitPolicy()).under([tracked])

    names = {path.relative_to(tracked).as_posix() for path in listed}
    assert "owned/README" in names
    assert not any(name.startswith(("absent", "foreign/")) for name in names)


def _outcome(output: str, *, owner: str = ".", code: int = 1) -> Outcome:
    return Outcome(step="ty", owner=owner, code=code, seconds=0.1, output=output)


def test_a_whole_owner_check_answers_for_the_files_the_pass_reads(workspace: Path) -> None:
    linter = Linter(workspace, load(Project().manifest(workspace)))
    printed = "a.py:3:1: error[one] first\nsrc/b.py:9:2: error[two] second\nFound 2 diagnostics"

    narrowed = linter._narrowed(_outcome(printed), frozenset({"a.py"}))
    elsewhere = linter._narrowed(_outcome(printed), frozenset({"c.py"}))
    owned = linter._narrowed(_outcome(printed, owner="pkg"), frozenset({"pkg/src/b.py"}))

    assert narrowed.failed and narrowed.output == "a.py:3:1: error[one] first"
    assert not elsewhere.failed
    assert owned.output == "src/b.py:9:2: error[two] second"


def test_a_check_that_broke_or_timed_out_keeps_its_failure(workspace: Path) -> None:
    linter = Linter(workspace, load(Project().manifest(workspace)))
    broken = _outcome("ty: failed to read the configuration")
    late = _outcome("b.py:1:1: error[x] y\nty exceeded its 300s deadline", code=TIMED_OUT)

    assert linter._narrowed(broken, frozenset({"a.py"})).failed
    assert linter._narrowed(late, frozenset({"a.py"})).failed
