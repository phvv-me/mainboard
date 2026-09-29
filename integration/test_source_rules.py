"""Rules the source keeps so a class of bug cannot come back, each one met more than once.

Text written without an explicit newline is CRLF on Windows: shell startup files, job scripts and
hashed closure listings written on the Windows center broke or changed identity on Linux hosts.
"""

import ast
from pathlib import Path

import pytest

_SOURCE = Path(__file__).parents[1] / "src" / "mainboard"
_FILES = sorted(_SOURCE.rglob("*.py"))


def _text_writes(tree: ast.AST) -> list[int]:
    """The lines writing text without saying which newline: `write_text` or a text-mode `open`."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {keyword.arg for keyword in node.keywords}
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name == "write_text" and "newline" not in keywords:
            found.append(node.lineno)
        if name in {"open", "fdopen"} and len(node.args) > 1:
            mode = node.args[1]
            if (
                isinstance(mode, ast.Constant)
                and isinstance(mode.value, str)
                and mode.value.strip("+t") in {"w", "a", "x"}
                and "newline" not in keywords
            ):
                found.append(node.lineno)
    return found


@pytest.mark.parametrize("path", _FILES, ids=lambda path: path.relative_to(_SOURCE).as_posix())
def test_text_is_written_with_an_explicit_newline(path: Path) -> None:
    lines = _text_writes(ast.parse(path.read_text(encoding="utf-8")))
    assert not lines, f"{path.name}: text written without newline= at lines {lines}"
