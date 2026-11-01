"""`Git` runs git in one working tree and decodes what it printed."""

from pathlib import Path

from mainboard.git import Git

from ..builders import committed, git


def test_output_that_is_no_utf8_is_replaced_with_the_replacement_character(
    tmp_path: Path,
) -> None:
    root = committed(tmp_path / "root")
    (root / "blob").write_bytes(b"caf\xe9\n")
    git(root, "add", "blob")
    git(root, "commit", "--quiet", "-m", "blob")

    assert Git(root).run("show", "HEAD:blob").stdout == "caf�\n"
