from pathlib import Path
from typing import NamedTuple


class Case(NamedTuple):
    """The files of a package before the fix and after it, by path inside the package's folder."""

    before: dict[str, str]
    after: dict[str, str]


def code(*lines: str) -> str:
    """The source of `lines`, each ending with a newline."""
    return "".join(f"{line}\n" for line in lines)


def tree(folder: Path) -> dict[str, str]:
    """Every file under `folder` by its path inside it, with its text."""
    return {
        path.relative_to(folder).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(folder.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
