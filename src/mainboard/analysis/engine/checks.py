"""What a fix must not break beyond its own rule: every edited file still parses in its language,
and the project holds no new type errors.

Syntax is judged before a pass is written: Python by `ast`, TOML by `tomllib`, any other language
sitting_duck parses by tree-sitter, whose error nodes and missing tokens must not outnumber those
of the file before. Types are ty's diagnostics over the whole project, read as findings of rules
named `ty:<rule>`, so a fix that adds one is judged like a fix that adds a finding.
"""

import ast
import re
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs ty with fixed argv since=2026-10-14
import sys
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from ..providers.ducks import Ducks

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .apply import Plan, Spot


class Syntax:
    """Whether the files a plan edits still parse."""

    @staticmethod
    def _unparsed(path: str, data: bytes) -> str:
        """Why Python or TOML `path` would not parse, empty when it would or is neither."""
        match PurePosixPath(path).suffix:
            case ".py" | ".pyi":
                try:
                    ast.parse(data)
                except SyntaxError as error:
                    return f"{path}: {error.msg} (line {error.lineno})"
            case ".toml":
                try:
                    tomllib.loads(data.decode())
                except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
                    return f"{path}: {error}"
        return ""

    def __init__(self) -> None:
        self.ducks = Ducks()

    def _grown(self, trees: Mapping[str, tuple[bytes, bytes]]) -> list[str]:
        """Each file whose new text has more tree-sitter errors than its old, with how many."""
        errors = self.ducks.errors(
            {
                f"{side}:{path}": text
                for path, pair in trees.items()
                for side, text in zip(("old", "new"), pair, strict=True)
            }
        )
        return [
            f"{path}: tree-sitter finds {errors[f'new:{path}'] - errors[f'old:{path}']} more "
            "syntax errors"
            for path in trees
            if errors[f"new:{path}"] > errors[f"old:{path}"]
        ]

    def broken(self, plan: Plan) -> list[str]:
        """Each edited file that would no longer parse, with why."""
        written = plan.written()
        trees = {
            path: (plan.source(path).data, data)
            for path, data in written.items()
            if (suffix := PurePosixPath(path).suffix) not in {".py", ".pyi", ".toml"}
            and self.ducks.language(suffix)
        }
        unparsed = [why for path, data in written.items() if (why := Syntax._unparsed(path, data))]
        return unparsed + (self._grown(trees) if trees else [])


class Types:
    """The type checkers' diagnostics for a project, through the Python that runs this: ty's and
    pyrefly's, since a fix one of them accepts the other may refuse (PY-PYDA0004's keyword
    calls passed ty and failed pyrefly)."""

    PATTERN = re.compile(
        r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<column>\d+): \w+\[(?P<rule>[\w-]+)\]"
    )

    PYREFLY = re.compile(
        r"^(?:ERROR|WARN) (?P<path>[^:\n]+):(?P<line>\d+):(?P<column>\d+)-\S*: "
        r".*\[(?P<rule>[\w-]+)\]$"
    )

    @staticmethod
    def _pyrefly(root: Path) -> Counter[Spot]:
        """pyrefly's diagnostics as findings of rule `pyrefly:<rule>` (columns zero-based)."""
        run = subprocess.run(
            [
                str(shutil.which("pyrefly")),
                "check",
                "--python-interpreter-path",
                sys.executable,
                "--output-format",
                "min-text",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        found: Counter[Spot] = Counter()
        base = root.resolve()
        for line in run.stdout.splitlines():
            if match := Types.PYREFLY.match(line):
                path = str((root / match["path"]).resolve().relative_to(base))
                rule = f"pyrefly:{match['rule']}"
                found[path, int(match["line"]), int(match["column"]) - 1, rule] += 1
        return found

    def diagnostics(self, root: Path) -> Counter[Spot]:
        """Every diagnostic as a finding of rule `ty:<rule>` or `pyrefly:<rule>`.

        Columns are zero-based.
        """
        run = subprocess.run(
            [
                str(shutil.which("ty")),
                "check",
                "--python",
                sys.executable,
                "--project",
                str(root),
                "--output-format",
                "concise",
                "--exit-zero",
                str(root),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        found: Counter[Spot] = Counter()
        base = root.resolve()
        for line in run.stdout.splitlines():
            if match := self.PATTERN.match(line):
                path = str(Path(match["path"]).resolve().relative_to(base))
                found[
                    path, int(match["line"]), int(match["column"]) - 1, f"ty:{match['rule']}"
                ] += 1
        return found + Types._pyrefly(root)


def imported(root: Path, path: str) -> set[str]:
    """The project files a Python file's import statements name, existing or not.

    A fix may have removed or renamed the one it imports. Each module counts as `<name>.py` and
    `<name>/__init__.py`, at the root, under `src/`, and beside the file or in a folder above it (a
    script that puts its own folder on `sys.path` imports its siblings).
    """
    try:
        tree = ast.parse((root / path).read_bytes())
    except SyntaxError, OSError:
        return set()
    package = PurePosixPath(path).parent
    names: set[PurePosixPath] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                names |= {PurePosixPath(*alias.name.split(".")) for alias in aliases}
            case ast.ImportFrom(module=module, names=aliases, level=level):
                base = (
                    package.parents[level - 2]
                    if 1 < level <= len(package.parents) + 1
                    else package
                )
                stem = PurePosixPath(*(module or "").split(".")) if module else PurePosixPath()
                origin = base / stem if level else stem
                names |= {origin, *(origin / alias.name for alias in aliases)}
    return {
        str(candidate)
        for name in names
        for prefix in (PurePosixPath("src"), package, *package.parents)
        for candidate in (prefix / f"{name}.py", prefix / name / "__init__.py")
    }
