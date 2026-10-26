"""Every name the workspace's research and packages import from mainboard still exists.

`mainboard.dispatch.shared.git` was removed on 2026-09-25 while ten experiments in cutok and
reproducibility still imported it; nothing failed until those experiments were collected on a
host four days later. Consumers are read from the monorepo this package sits in, when it does.
"""

import ast
import importlib
import importlib.util
import warnings
from contextlib import suppress
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
# Consumers known to be broken, each with why it is allowed to stay so.
_BROKEN = {
    # Built on the study/fleet layer removed on 2026-10-07; the owner set compression aside.
    "research/compression/experiments/dispatch.py",
}


def _imported(path: Path) -> list[tuple[str, str]]:
    """Each (module, name) `path` imports from mainboard, none when it does not parse."""
    with (
        suppress(SyntaxError, ValueError),
        warnings.catch_warnings(action="ignore", category=SyntaxWarning),
    ):
        return [
            (node.module or "", alias.name)
            for node in ast.walk(ast.parse(path.read_bytes()))
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith(("mainboard.", "mb."))
            for alias in node.names
        ]
    return []


def _consumers() -> dict[tuple[str, str], list[str]]:
    """Each (module, name) imported from mainboard outside it, with the files importing it."""
    wheres = [
        path.relative_to(_WORKSPACE)
        for top in ("research", "packages", "life", "apps")
        for path in (_WORKSPACE / top).rglob("*.py")
    ]
    found: dict[tuple[str, str], list[str]] = {}
    for where in wheres:
        if (
            where.parts[:2] != ("packages", "mainboard")
            and not any(skip in where.parts for skip in _SKIPPED)
            and where.as_posix() not in _BROKEN
        ):
            for imported in _imported(_WORKSPACE / where):
                found.setdefault(imported, []).append(where.as_posix())
    return found


_FOUND = _consumers() if (_WORKSPACE / "research").is_dir() else {}


def _is_gone(module_name: str, *, name: str) -> bool:
    """Whether `name` left `module_name`; a name behind a missing optional dependency is kept."""
    try:
        module = importlib.import_module(module_name.replace("mb.", "mainboard.", 1))
    except ModuleNotFoundError as missing:
        return (missing.name or "").startswith("mainboard")
    if hasattr(module, name):
        return False
    try:
        return importlib.util.find_spec(f"{module_name}.{name}") is None
    except ModuleNotFoundError:
        return True


# One case for every import, so the node id is the same inside the monorepo and outside it.
@pytest.mark.skipif(not _FOUND, reason="not inside the monorepo")
def test_every_imported_name_exists() -> None:
    gone = [
        f"{module_name}.{name} ({', '.join(_FOUND[module_name, name][:3])})"
        for module_name, name in sorted(_FOUND)
        if _is_gone(module_name, name=name)
    ]
    assert not gone, f"gone but still imported: {gone}"
