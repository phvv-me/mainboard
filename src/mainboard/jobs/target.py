# How a job is spelled: `path/to/file.py::name`, an application or a function inside one file,
# and the arguments it runs with. Never an entry in the manifest's task table, which names a tool
# and not an experiment.

import ast
import shlex
from itertools import takewhile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .declare import Declaration, declared

if TYPE_CHECKING:
    from collections.abc import Sequence

# What separates the file from the name inside it.
SEPARATOR = "::"

# The names a bare file spelling means, in the order they are looked for.
DEFAULTS = ("app", "main")

# The file prefix making a target a pytest module, so it gets fixtures, parametrization and
# setup/teardown without mainboard growing an experiment DSL beside them.
TEST_PREFIX = "test_"


def home_of(file: Path, *, root: Path) -> Path:
    """The import root of absolute `file`: above its outermost regular package, climbing through
    identifier-named directories below `root`, since a missing initializer may be a namespace
    portion. A script outside any package keeps its own directory; a wholly namespace-based tree
    needs a separately declared import root."""
    file.relative_to(root)
    climb = takewhile(lambda parent: parent != root and parent.name.isidentifier(), file.parents)
    packages = [parent for parent in climb if (parent / "__init__.py").is_file()]
    return packages[-1].parent if packages else file.parent


def dotted(file: Path, *, home: Path) -> str:
    """The module name `file` is imported by from `home`, its package chain included."""
    parts = file.relative_to(home).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


class Target(FrozenModel):
    """One job as spelled: a file, the name inside it, and the arguments it runs with.

    file: workspace-relative.
    name: a cyclopts `App` or a zero-argument function in that module, or a pytest node id in a
        `test_` file (`Class::test_case[param]`, everything after the first separator), or empty
        meaning the whole test file.
    args: the tokens handed to the application or to pytest, none for a function.
    """

    file: str
    name: str
    args: tuple[str, ...] = ()

    @classmethod
    def spelled(cls, tokens: Sequence[str], root: Path) -> Target | None:
        """The job `tokens` spell, or None when they are an ordinary command line.

        A first token naming a Python file is a job, `::name` picking the target and a bare file
        meaning `app` then `main` (a `test_` file: all its tests). A name given for a missing
        file is refused rather than passed on as a command.
        """
        if not tokens:
            return None
        spelling, separator, name = tokens[0].partition(SEPARATOR)
        if not spelling.endswith(".py"):
            return None
        file = cls.__relative(spelling, root)
        if not (root / file).is_file():
            if separator:
                raise MissionError(f"no job file at {spelling}")
            return None
        start = 2 if len(tokens) > 1 and tokens[1] == "--" else 1
        return cls(file=file, name=name or cls.__default(root / file), args=tuple(tokens[start:]))

    @property
    def spelling(self) -> str:
        """The target as a command line spells it, arguments quoted."""
        head = f"{self.file}{SEPARATOR}{self.name}" if self.name else self.file
        return shlex.join([head, *self.args])

    @property
    def test(self) -> bool:
        """Whether the job file is a pytest module, run through pytest."""
        return PurePosixPath(self.file).stem.startswith(TEST_PREFIX)

    @property
    def registration(self) -> str:
        """The adjacent node for an experiment target, empty for ordinary software."""
        path = PurePosixPath(self.file)
        if path.parts[0] not in ("research", "experiments"):
            return ""
        if not any(parent.name == "experiments" for parent in path.parents):
            return ""
        return (path.parent / "node.md").as_posix()

    @property
    def node(self) -> str:
        """The directory the job file lives in, workspace-relative."""
        return PurePosixPath(self.file).parent.as_posix()

    def declaration(self, root: Path) -> Declaration:
        """What the target declared beyond its imports, read off the file's syntax.

        `Class::test_case[1]` declares what `test_case` declared. A test under `experiments/<x>/`
        declaring no fetch fetches its `datasets/experiments/<x>` sibling.
        """
        declaration = declared(parsed(root / self.file), self.name.partition("[")[0])
        if self.test and not declaration.fetch:
            for parent in PurePosixPath(self.file).parents:
                if parent.name == "experiments":
                    parts = PurePosixPath(self.file).relative_to(parent).parts
                    if len(parts) > 1:
                        fetch = parent.parent / "datasets" / "experiments" / parts[0]
                        return declaration.model_copy(update={"fetch": fetch.as_posix()})
        return declaration

    @staticmethod
    def __relative(spelling: str, root: Path) -> str:
        """Resolve aliases to one workspace-relative path, refusing any escape."""
        try:
            given = (root / spelling).resolve().relative_to(root.resolve())
        except ValueError:
            raise MissionError(f"{spelling} is outside the workspace {root}") from None
        return PurePosixPath(given).as_posix()

    @staticmethod
    def __default(file: Path) -> str:
        """The name a bare file spelling means: a whole pytest file, else `app` then `main`."""
        if PurePosixPath(file.name).stem.startswith(TEST_PREFIX):
            return ""
        body = parsed(file).body
        defined = {
            target.id
            for node in body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        } | {
            node.name for node in body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        if found := next((name for name in DEFAULTS if name in defined), None):
            return found
        choices = " or ".join(f"`{name}`" for name in DEFAULTS)
        raise MissionError(
            f"{file} defines neither {choices}; spell the target as {file}{SEPARATOR}<name>"
        )


def parsed(file: Path) -> ast.Module:
    """The syntax of one job file, which is all a dispatch ever reads of it."""
    return ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
