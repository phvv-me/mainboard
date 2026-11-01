"""The completion scripts and the README command tree are generated and never hand-written."""

from pathlib import Path

import pytest

from mainboard.guide.docs import CommandDocs, Shell


@pytest.mark.parametrize("shell", ["bash", "fish", "zsh"])
def test_a_fresh_script_equals_the_committed_script(shell: Shell, tmp_path: Path) -> None:
    committed = CommandDocs.committed()
    fresh = CommandDocs(readme=tmp_path / "README.md", completions=tmp_path / "completions")
    (tmp_path / "README.md").write_text(committed.readme.read_text())
    fresh.write()
    assert fresh.path(shell).read_text() == committed.path(shell).read_text()


def test_a_fresh_readme_equals_the_committed_readme(tmp_path: Path) -> None:
    committed = CommandDocs.committed()
    fresh = CommandDocs(readme=tmp_path / "README.md", completions=tmp_path / "completions")
    (tmp_path / "README.md").write_text(committed.readme.read_text())
    fresh.write()
    assert fresh.readme.read_text() == committed.readme.read_text()
    assert committed.stale() == []


def test_a_readme_without_the_markers_is_refused(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# mainboard\n")
    docs = CommandDocs(readme=tmp_path / "README.md", completions=tmp_path / "completions")
    with pytest.raises(ValueError, match="mb-commands"):
        docs.render()
    assert not (tmp_path / "completions").exists()
