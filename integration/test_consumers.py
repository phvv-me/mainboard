"""Every name the workspace's research and packages import from mainboard still exists.

`mainboard.dispatch.shared.git` was removed on 2026-09-25 while ten experiments in cutok and
reproducibility still imported it; nothing failed until those experiments were collected on a
host four days later. Consumers are read from the monorepo this package sits in, when it does.
"""

import ast
import importlib
from pathlib import Path

import pytest

_WORKSPACE = Path(__file__).parents[3]
# Frozen evidence under `datasets/` records what ran then; it is not a live consumer.
_SKIPPED = (
    "references",
    ".mainboard",
    "node_modules",
    ".pixi",
    ".venv",
    "third_party",
    "data",
    "datasets",
)


def _consumers() -> dict[tuple[str, str], list[str]]:
    """Each (module, name) imported from mainboard outside it, with the files importing it."""
    found: dict[tuple[str, str], list[str]] = {}
    for top in ("research", "packages", "life", "apps"):
        for path in (_WORKSPACE / top).rglob("*.py"):
            parts = path.relative_to(_WORKSPACE).parts
            if parts[:2] == ("packages", "mainboard") or any(skip in parts for skip in _SKIPPED):
                continue
            try:
                tree = ast.parse(path.read_bytes())
            except SyntaxError, ValueError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    ("mainboard.", "mb.")
                ):
                    for alias in node.names:
                        where = path.relative_to(_WORKSPACE).as_posix()
                        found.setdefault((node.module or "", alias.name), []).append(where)
    return found


_FOUND = _consumers() if (_WORKSPACE / "research").is_dir() else {}


@pytest.mark.skipif(not _FOUND, reason="not inside the monorepo")
@pytest.mark.parametrize("imported", sorted(_FOUND), ids=lambda pair: f"{pair[0]}.{pair[1]}")
def test_every_imported_name_exists(imported: tuple[str, str]) -> None:
    module_name, name = imported
    try:
        module = importlib.import_module(module_name.replace("mb.", "mainboard.", 1))
    except ModuleNotFoundError as missing:
        if not (missing.name or "").startswith("mainboard"):
            pytest.skip(f"needs {missing.name}, an optional dependency")
        raise
    assert hasattr(module, name) or importlib.util.find_spec(f"{module_name}.{name}"), (
        f"{module_name}.{name} is gone but {', '.join(_FOUND[imported][:3])} import it"
    )
