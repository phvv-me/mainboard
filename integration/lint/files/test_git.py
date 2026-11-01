"""A lint pass asks git where the workspace is and which files changed, each name read whole."""

from pathlib import Path

from mainboard.lint import Inventory, topmost
from mainboard.lint.files.git import git as asked
from mainboard.manifest.schema.project.git import GitPolicy

from ...builders import committed, git


def test_the_top_working_tree_of_a_submodule_is_its_superproject_and_none_outside_git(
    tmp_path: Path,
) -> None:
    root, inner = committed(tmp_path / "root"), committed(tmp_path / "inner")
    git(root, "submodule", "add", "--quiet", inner.as_posix(), "deps/inner")
    (tmp_path / "plain").mkdir()

    assert (topmost(root / "deps" / "inner"), topmost(tmp_path / "plain")) == (
        root.resolve(),
        None,
    )


def test_an_untracked_file_named_in_japanese_is_listed_by_its_own_name(tmp_path: Path) -> None:
    root = committed(tmp_path / "root")
    (root / "日本語.py").write_text("x = 1\n", encoding="utf-8")

    assert Inventory(root, GitPolicy()).changed() == [root / "日本語.py"]


def test_output_that_is_no_utf8_is_kept_byte_for_byte_in_what_a_lint_pass_reads(
    tmp_path: Path,
) -> None:
    root = committed(tmp_path / "root")
    (root / "blob").write_bytes(b"caf\xe9\n")
    git(root, "add", "blob")
    git(root, "commit", "--quiet", "-m", "blob")

    printed = asked(root, "show", "HEAD:blob").stdout

    assert printed.encode("utf-8", "surrogateescape") == b"caf\xe9\n"
