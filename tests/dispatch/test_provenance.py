from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch import provenance as provenance_module
from mainboard.dispatch.provenance import (
    Repositories,
    Repository,
    Source,
    Status,
    blob_of,
    commanded,
    listing,
    named,
    tree_source,
)
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target

from ..support import Lab


def sealed(lab: Lab) -> tuple[Source, list]:
    """The lab's job sealed: its source and the listing of what it ships."""
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(target, root=lab.root, distributions=Lab.DISTRIBUTIONS)
    return Repositories(lab.root).seal(closure.owner, closure.files)


def test_a_closure_is_clean_while_only_paths_outside_it_or_another_repository_move(
    lab: Lab,
) -> None:
    """The fault this exists for: a cutok job stamped dirty for hours by llm-head edits."""
    clean, rows = sealed(lab)
    head = lab.git("rev-parse", "HEAD")
    assert clean.identity == lab.git("describe", "--always") and not clean.dirty
    assert clean.commit == head and len(clean.digest) == 64
    assert clean.key == f"{named(clean.identity)}-{clean.digest[:8]}"
    assert {row.status for row in rows} == {Status.CLEAN}
    assert [row.path for row in rows] == sorted(row.path for row in rows)

    lab.write("research/other/experiments/node/run.py", "def main() -> int:\n    return 2\n")
    lab.write("research/other/untracked.py", "x = 1\n")
    lab.write("packages/sub/README.md", "a submodule edit outside the closure\n")
    assert lab.git("status", "--porcelain")
    again, _ = sealed(lab)
    assert again == clean


def test_a_closure_is_dirty_when_one_of_its_own_files_moves_wherever_that_file_lives(
    lab: Lab,
) -> None:
    clean, _ = sealed(lab)

    lab.write("packages/sub/src/sub/thing.py", "THING = 2\n")
    inside_submodule, rows = sealed(lab)
    assert inside_submodule.dirty and inside_submodule.identity == f"{clean.identity}-dirty"
    assert inside_submodule.digest != clean.digest and inside_submodule.key != clean.key
    assert inside_submodule.commit == clean.commit
    [moved] = [row for row in rows if row.status is Status.MODIFIED]
    assert moved.path == "packages/sub/src/sub/thing.py"
    assert moved.blob == blob_of(lab.root / moved.path)

    lab.write("research/camp/experiments/node/new.py", "NEW = 1\n")
    untracked, rows = sealed(lab)
    assert untracked.dirty
    assert {row.path for row in rows if row.status is Status.UNTRACKED} == {
        "research/camp/experiments/node/new.py"
    }
    assert untracked.digest != inside_submodule.digest

    lab.commit("settled")
    lab.git("add", "-A", cwd=lab.root / "packages/sub")
    settled, _ = sealed(lab)
    assert settled.dirty, "the submodule's own index moved and its tree is not committed"


def test_an_ignored_module_the_job_imports_is_digested_but_never_dirt(lab: Lab) -> None:
    lab.write("research/camp/experiments/helper/x_generated.py", "X = 1\n")
    lab.write(
        "research/camp/experiments/node/run.py", "from ..helper import x_generated\napp = 1\n"
    )
    lab.commit("import the generated module")
    source, rows = sealed(lab)
    [ignored] = [row for row in rows if row.status is Status.IGNORED]
    assert ignored.path == "research/camp/experiments/helper/x_generated.py"
    assert not source.dirty
    lab.write("research/camp/experiments/helper/x_generated.py", "X = 2\n")
    regenerated, _ = sealed(lab)
    assert not regenerated.dirty and regenerated.digest != source.digest


def test_a_built_extension_is_recorded_built_outright_and_never_marks_the_tree_dirty(
    lab: Lab,
) -> None:
    """An untracked `.so` git would call `UNTRACKED` (and so dirty) is `built` instead."""
    shipped = "packages/core/src/core/_native.cpython-314-x86_64-linux-gnu.so"
    binary = lab.write(shipped, "not an elf, a stub\n")
    source, rows = Repositories(lab.root).seal(
        Repository.owning(lab.root), ["mainboard.toml", shipped], built=(shipped,)
    )
    [built] = [row for row in rows if row.status is Status.BUILT]
    assert built.path == shipped
    assert built.blob == blob_of(binary)
    assert not source.dirty


def test_a_file_under_no_repository_is_unversioned_and_still_digested(
    lab: Lab, tmp_path: Path
) -> None:
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "a.py").write_text("A = 1\n", encoding="utf-8")
    source, rows = Repositories(loose).seal(None, ["a.py"])
    assert rows == [
        provenance_module.Row(path="a.py", blob=blob_of(loose / "a.py"), status=Status.UNVERSIONED)
    ]
    assert source.identity == "" and source.commit == ""
    assert source.key.startswith("untracked-") and not source.dirty


def test_the_listing_is_what_the_digest_is_taken_over(lab: Lab) -> None:
    source, rows = sealed(lab)
    import hashlib

    assert source.digest == hashlib.sha256(listing(rows).encode()).hexdigest()
    assert listing(rows[:1]) == f"{rows[0].path}\t{rows[0].blob}\tclean\n"


def test_a_repository_answers_the_few_questions_a_dispatch_asks_of_it(lab: Lab) -> None:
    repository = Repository.owning(lab.root / "research/camp")
    assert repository is not None and Path(repository.path) == lab.root
    assert Repository.owning(lab.root / "packages/sub/src") == Repository(
        path=str(lab.root / "packages/sub")
    )
    assert repository.head() == lab.git("rev-parse", "HEAD")
    assert repository.describe() == lab.git("describe", "--always")
    assert "research/camp/registry.toml" in repository.index(["research/camp/registry.toml"])
    assert repository.kept("research/camp/experiments/node") == [
        "research/camp/experiments/node/__init__.py",
        "research/camp/experiments/node/node.md",
        "research/camp/experiments/node/run.py",
    ]
    # A tracked file deleted from the tree is listed by the index and cannot ship.
    (lab.root / "research/camp/experiments/node/node.md").unlink()
    assert "research/camp/experiments/node/node.md" not in repository.kept(
        "research/camp/experiments/node"
    )
    # A staged rename reports two paths on one entry, and the second is skipped.
    lab.git("mv", "research/camp/registry.toml", "research/camp/moved.toml")
    states = repository.states(["research/camp/moved.toml", "research/camp/registry.toml"])
    assert states["research/camp/moved.toml"] is Status.MODIFIED
    assert repository.delta(["research/camp/moved.toml"])
    assert Repository.owning(Path("/")) is None


def test_the_repositories_are_asked_once_each_and_a_loose_directory_is_refused(
    lab: Lab, tmp_path: Path
) -> None:
    repositories = Repositories(lab.root)
    assert repositories.owning(lab.root / "packages") == repositories.owning(lab.root / "packages")
    assert len(repositories.repositories) == 1
    assert repositories.kept("packages/sub/src") == [
        "packages/sub/src/sub/__init__.py",
        "packages/sub/src/sub/thing.py",
    ]
    assert repositories.kept(".") == repositories.kept("")
    loose = tmp_path / "loose"
    loose.mkdir()
    with pytest.raises(MissionError, match="under no git repository"):
        Repositories(loose).kept("x")


def test_a_command_ships_the_tree_owning_its_code_with_other_repositories_dirt_left_out(
    lab: Lab,
) -> None:
    """The whole-tree reading a tool task keeps, blind to a submodule's content changes."""
    top = tree_source(commanded("python -m foo", lab.root))
    assert not top.dirty and top.commit == lab.git("rev-parse", "HEAD")
    assert top.key == named(top.identity) and len(top.digest) == 64
    lab.write("packages/sub/src/sub/thing.py", "THING = 2\n")
    assert "M packages/sub" in lab.git("status", "--porcelain")
    assert tree_source(commanded("python -m foo", lab.root)) == top
    lab.write("research/other/experiments/node/run.py", "changed = True\n")
    moved = tree_source(commanded("python -m foo", lab.root))
    assert moved.dirty and moved.identity == f"{top.identity}-dirty"
    assert moved.key.startswith(f"{named(top.identity)}-dirty-") and moved.key != top.key
    # A token naming a path inside the submodule picks that repository; a data path picks its
    # owner; a token naming nothing on disk is skipped.
    inside = tree_source(commanded("pytest packages/sub/src/sub/thing.py -q", lab.root))
    assert inside.dirty and inside.commit == lab.git(
        "rev-parse", "HEAD", cwd=lab.root / "packages/sub"
    )
    assert commanded("cat missing.txt research/camp", lab.root) == commanded("", lab.root)
    loose = lab.root.parent / "loose.txt"
    loose.write_text("under no repository\n", encoding="utf-8")
    assert commanded(f"cat {loose} packages/sub/src/sub/thing.py", lab.root) == Repository.owning(
        lab.root / "packages/sub"
    )


def test_a_workspace_with_no_git_at_all_still_gets_a_key(tmp_path: Path) -> None:
    assert tree_source(None) == Source(identity="", key="untracked")
    assert commanded("python -m foo", tmp_path) is None


@given(identity=st.text(max_size=120))
def test_a_key_can_never_name_a_path_outside_the_sources_directory(identity: str) -> None:
    """A source identity is git's text, and text reaches the shell that builds the tree."""
    key = named(identity)
    assert key and "/" not in key and not key.startswith(".") and len(key) <= 96
    assert named("../../etc/passwd") == "-..-etc-passwd"
    assert named("..") == "untracked"
