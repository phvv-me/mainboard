# The runner a job's script calls: `python -m mainboard.jobs.call <file>::<name> -- args`.
#
# One process, in the job's own environment, standing in the tree the dispatch pinned. The file
# is imported by the name its package chain gives it, so the relative imports inside it resolve,
# and the target is run as what it is: an application gets the arguments, a function gets none.
#
# THE CLOSURE IS A BOUNDARY, NOT A HINT. The environment's editable installs point at the mirror,
# and a snapshot that ships only what the job imports leaves every other first-party module one
# `sys.path` entry away in mutable source. So before the job file is imported, a finder is armed
# that answers every first-party import by the closure listing: a module whose file the listing
# does not name is refused by name, never read from wherever else it happens to be. A local run
# arms the same finder over the same listing, so a walk that missed a module fails on the
# workstation and not on the node.

import importlib
import os
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from cyclopts import App

from ..dispatch.shared import CLOSURE_VAR, FIRST_PARTY_VAR
from .target import SEPARATOR, dotted, home_of

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# What separates the target from the arguments handed to it.
DELIMITER = "--"


class Guard(MetaPathFinder):
    """The finder that keeps a first-party import inside the closure.

    names: the top-level names the workspace's own import roots define.
    shipped: the files the closure holds, relative to `root`.
    root: the tree the job stands in, the pinned snapshot or the workspace.
    """

    def __init__(self, names: Sequence[str], shipped: Sequence[str], root: Path) -> None:
        self.names = frozenset(names)
        self.shipped = frozenset(shipped)
        self.root = root

    @classmethod
    def armed(cls, root: Path) -> Guard | None:
        """The guard the environment describes, first on `sys.meta_path`; None for a command."""
        listing = os.environ.get(CLOSURE_VAR, "")
        if not listing:
            return None
        rows = Path(listing).read_text(encoding="utf-8").splitlines()
        guard = cls(
            os.environ.get(FIRST_PARTY_VAR, "").split(":"),
            [row.split("\t", maxsplit=1)[0] for row in rows if row],
            root,
        )
        sys.meta_path.insert(0, guard)
        return guard

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        """The spec of a first-party module the closure ships, a refusal for one it does not."""
        del target
        if fullname.partition(".")[0] not in self.names:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and self.holds(spec):
            return spec
        raise ModuleNotFoundError(
            f"{fullname} is first-party code outside this job's closure; import it from the "
            "job file, or declare the file it lives in as a resource, so the dispatch ships it",
            name=fullname,
        )

    def holds(self, spec: ModuleSpec) -> bool:
        """Whether the closure ships the file `spec` was found at; a portion with none passes."""
        if spec.origin is None:
            return True
        try:
            return Path(spec.origin).relative_to(self.root).as_posix() in self.shipped
        except ValueError:
            return False


def main(argv: Sequence[str] | None = None) -> int:
    """Run one target and answer its exit code.

    argv: the tokens after the runner's own name, this process's when None.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] == DELIMITER:
        raise SystemExit(f"usage: python -m {__spec__.name} <file>{SEPARATOR}<name> [-- args]")
    spelling, args = tokens[0], tokens[1:]
    if args and args[0] == DELIMITER:
        args = args[1:]
    file, _, name = spelling.partition(SEPARATOR)
    Guard.armed(Path.cwd())
    return called(getattr(loaded(Path(file)), name), name, args)


def loaded(file: Path) -> ModuleType:
    """Import `file` by the name its package chain gives it, from the root the chain stops at."""
    real = file.resolve()
    home = home_of(real, root=Path(real.anchor))
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    return importlib.import_module(dotted(real, home=home))


def called(target: App | Callable[[], int | None], name: str, args: Sequence[str]) -> int:
    """Run `target`: an application with `args`, a function with none, answering an exit code."""
    if isinstance(target, App):
        outcome = target(list(args))
        return outcome if isinstance(outcome, int) else 0
    if args:
        raise SystemExit(f"{name} is a function and takes no arguments, but got {list(args)}")
    if not callable(target):
        raise SystemExit(f"{name} is neither an application nor a function")
    outcome = target()
    return outcome if isinstance(outcome, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
