# How a job is spelled: `path/to/file.py::name`, an application or a function inside one file,
# and the arguments it runs with. Never an entry in the manifest's task table, which names a tool
# and not an experiment.

import ast
import shlex
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

# The file prefix that makes a target a pytest module, the conventional boundary pytest itself
# collects by. A `test_` file runs through pytest, so it gets fixtures, parametrization and
# setup/teardown without mainboard growing an experiment DSL beside them.
TEST_PREFIX = "test_"


def home_of(file: Path, *, root: Path) -> Path:
    """The directory `file` is imported from: above every `__init__.py` its packages stack.

    The import root of a module is where its package chain stops, so a file under
    `experiments/gds_ingest/run.py` whose two parents carry `__init__.py` is imported as
    `experiments.gds_ingest.run` from the directory above `experiments`. A bare script whose
    directory carries none is imported from that directory. Nothing above `root` is climbed.

    file: the module, absolute.
    root: the highest directory the climb may reach.
    """
    home = file.parent
    while home != root and (home / "__init__.py").is_file():
        home = home.parent
    return home


def dotted(file: Path, *, home: Path) -> str:
    """The module name `file` is imported by from `home`, its package chain included."""
    parts = file.relative_to(home).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


class Target(FrozenModel):
    """One job as spelled: a file, the name inside it, and the arguments it runs with.

    file: the job file, workspace-relative.
    name: a cyclopts `App` or a zero-argument function defined in that module, or a pytest node
        id inside a `test_` file (`Class::test_case[param]` included, everything after the
        first separator), or empty meaning the whole test file.
    args: the tokens handed to the application or to pytest, none for a function.
    """

    file: str
    name: str
    args: tuple[str, ...] = ()

    @classmethod
    def spelled(cls, tokens: Sequence[str], root: Path) -> Target | None:
        """The job `tokens` spell, or None when they are an ordinary command line.

        A first token naming a Python file is a job, `::name` picking the target inside it and
        a bare file meaning `app` and then `main`, whichever the file defines -- or, for a
        `test_` file, meaning the whole file's tests. A name given for a file that is not there
        is refused rather than passed on as a command.

        tokens: the command tokens as the caller typed them.
        root: the workspace root the file is spelled from.
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
        return cls(file=file, name=name or cls.__default(root / file), args=tuple(tokens[1:]))

    @property
    def spelling(self) -> str:
        """The target as a command line spells it, its arguments quoted."""
        head = f"{self.file}{SEPARATOR}{self.name}" if self.name else self.file
        return f"{head} {shlex.join(self.args)}" if self.args else head

    @property
    def test(self) -> bool:
        """Whether the job file is a pytest module, and so runs through pytest."""
        return PurePosixPath(self.file).stem.startswith(TEST_PREFIX)

    @property
    def node(self) -> str:
        """The directory the job file lives in, workspace-relative."""
        return PurePosixPath(self.file).parent.as_posix()

    def declaration(self, root: Path) -> Declaration:
        """What the target declared beyond its imports, read off the file's syntax.

        A pytest node id names its function through the class and the parameters it is run
        with, and the declaration sits on the function behind both: `Class::test_case[1]`
        declares what `test_case` declared.
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
        """`spelling` as a workspace-relative posix path, an absolute one rerooted."""
        given = Path(spelling)
        if given.is_absolute():
            try:
                given = given.relative_to(root)
            except ValueError:
                raise MissionError(f"{spelling} is outside the workspace {root}") from None
        return PurePosixPath(given).as_posix()

    @staticmethod
    def __default(file: Path) -> str:
        """The name a bare file spelling means: a whole pytest file, else `app` then `main`."""
        if PurePosixPath(file.name).stem.startswith(TEST_PREFIX):
            return ""
        defined = {
            target.id
            for node in parsed(file).body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        } | {
            node.name
            for node in parsed(file).body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for name in DEFAULTS:
            if name in defined:
                return name
        choices = " or ".join(f"`{name}`" for name in DEFAULTS)
        raise MissionError(
            f"{file} defines neither {choices}; spell the target as {file}{SEPARATOR}<name>"
        )


def parsed(file: Path) -> ast.Module:
    """The syntax of one job file, which is all a dispatch ever reads of it."""
    return ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
