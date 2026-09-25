from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.lint import Inventory
from mainboard.lint.inventory import Attributes
from mainboard.lint.owners import Owners

from .conftest import Repository


@pytest.fixture
def nested(repository: Repository) -> Repository:
    """`repository` holding a committed nested repository at `pkgs/sub`, git's gitlink shape."""
    sub = Repository(repository.root / "pkgs" / "sub")
    sub.write("inner.py", "x = 1\n")
    sub.commit()
    repository.write(".gitignore", "ignored/\n")
    repository.write("kept.txt", "kept\n")
    repository.write("gone.txt", "gone\n")
    repository.commit()
    return repository


def _names(repository: Repository, paths: list[Path]) -> list[str]:
    return [path.relative_to(repository.root).as_posix() for path in paths]


def test_changed_files_are_the_edits_the_deletions_and_the_new_files_inside_submodules_too(
    nested: Repository,
) -> None:
    nested.write("kept.txt", "edited\n")
    (nested.root / "gone.txt").unlink()
    nested.write("new/fresh.md", "fresh\n")
    nested.write("ignored/noise.txt", "noise\n")
    nested.write("pkgs/sub/inner.py", "x = 2\n")
    nested.write("pkgs/sub/added.py", "y = 1\n")

    changed = _names(nested, Inventory(nested.root).changed())

    assert changed == [
        "gone.txt",
        "kept.txt",
        "new/fresh.md",
        "pkgs/sub/added.py",
        "pkgs/sub/inner.py",
    ]


def test_staged_files_are_what_the_commit_records_and_never_a_submodule_pointer(
    nested: Repository,
) -> None:
    nested.write("kept.txt", "staged\n")
    nested.write("unstaged.txt", "not staged\n")
    nested.write("pkgs/sub/inner.py", "x = 3\n")
    Repository(nested.root / "pkgs" / "sub").commit()
    nested.git("add", "kept.txt", "pkgs/sub")
    nested.git("rm", "-q", "gone.txt")

    assert _names(nested, Inventory(nested.root).staged()) == ["gone.txt", "kept.txt"]


def test_a_directory_widens_to_every_file_git_tracks_or_would_beneath_it(
    nested: Repository,
) -> None:
    nested.write("ignored/noise.txt", "noise\n")
    nested.write("pkgs/new.py", "z = 1\n")
    nested.git("rm", "-q", "--cached", "kept.txt")
    (nested.root / "gone.txt").unlink()
    inventory = Inventory(nested.root)

    everything = _names(nested, inventory.under([nested.root]))
    one = _names(nested, inventory.under([nested.root / "pkgs", nested.root / "kept.txt"]))

    assert "gone.txt" not in everything
    assert "ignored/noise.txt" not in everything
    assert {"kept.txt", "pkgs/new.py", "pkgs/sub/inner.py", "tool.py"} <= set(everything)
    assert one == ["kept.txt", "pkgs/new.py", "pkgs/sub/inner.py"]


@pytest.mark.parametrize(
    ("where", "refusal"),
    [(Path("missing.txt"), "nothing to lint"), (Path("..", "elsewhere"), "outside the workspace")],
    ids=["a path that does not exist", "a path outside the workspace"],
)
def test_a_path_that_names_nothing_in_the_workspace_is_refused(
    repository: Repository, where: Path, refusal: str
) -> None:
    with pytest.raises(MissionError, match=refusal):
        Inventory(repository.root).under([(repository.root / where).resolve()])


def test_gitattributes_decide_which_files_are_binary_and_which_keep_crlf(
    repository: Repository,
) -> None:
    repository.write(".gitattributes", "*.bin binary\n*.bat eol=crlf\nvendor/** -text\n")
    files = [repository.write(name, "x\n") for name in ("a.bin", "run.bat", "vendor/v.c", "a.py")]

    attributes = Inventory(repository.root).attributes(files)

    assert [attributes[path] for path in files] == [
        Attributes(binary=True),
        Attributes(newline="\r\n"),
        Attributes(binary=True),
        Attributes(),
    ]


def test_only_a_file_head_already_holds_counts_as_tracked(repository: Repository) -> None:
    inventory = Inventory(repository.root)
    fresh = repository.write("fresh.txt", "new\n")
    repository.git("add", "fresh.txt")

    assert inventory.tracked(repository.root / "tool.py")
    assert not inventory.tracked(fresh)


def test_a_workspace_outside_git_is_refused_with_gits_own_words(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="git diff failed"):
        Inventory(tmp_path).changed()


def test_the_nearest_marked_or_named_directory_owns_a_file_and_the_root_owns_the_rest(
    tmp_path: Path,
) -> None:
    (tmp_path / "research" / "proj" / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "research" / "proj" / "pkg" / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "loose").mkdir()
    owners = Owners(tmp_path, ["research/*"], ["pyproject.toml"])

    assert owners.of(tmp_path / "research" / "proj" / "pkg" / "src") == (
        tmp_path / "research" / "proj" / "pkg"
    )
    assert owners.of(tmp_path / "research" / "proj") == tmp_path / "research" / "proj"
    assert owners.of(tmp_path / "loose") == tmp_path
    assert owners.of(tmp_path) == tmp_path
