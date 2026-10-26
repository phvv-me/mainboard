# The runner a job's script calls: `python -m mainboard.jobs.call <file>::<name> -- args`.
#
# One process, in the job's environment, standing in the pinned tree. The file is imported by the
# name its package chain gives it, so its relative imports resolve; an application gets the
# arguments, a function none.
#
# THE CLOSURE IS A BOUNDARY, NOT A HINT. Editable installs point at the mirror, so every
# first-party module the snapshot did not ship is one `sys.path` entry away in mutable source.
# Before the job file is imported, a finder answers every first-party import by the closure
# listing, refusing by name a module whose file it does not name. A local run arms the same
# finder, so a walk that missed a module fails on the workstation, not the node.
#
# A DEFERRED NAME IS THE ONE EXCEPTION: a package whose compiled extension lives outside the tree
# ships none of itself on purpose, its distribution left to the environment's install, so the
# guard steps aside for it as for anything not first-party.
#
# A TEST FILE IS PYTEST'S TO RUN, handed to `pytest.main` as the node id it spells, with the guard
# already armed. pytest's rewrite hook sits ahead of the guard and walks on its own, so the native
# adapter wraps it at installation (before plugins and conftests import) and judges its spec
# before execution, assertion rewriting intact.

import importlib
import os
import sys
import sysconfig
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from cyclopts import App

from ..dispatch.constants import CLOSURE_VAR, DEFERRED_VAR, FIRST_PARTY_VAR, RECEIPTS_VAR
from ..log import logger
from ..proc import Exit, Processes
from .constants import NESTED
from .targets.pins import staging
from .targets.target import Target, dotted, home_of
from .testing import beacon

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# What separates the target from the arguments handed to it.
_DELIMITER = "--"
# This module's dotted name, also when it runs as `python -m` and `__name__` is `__main__`.
_MODULE = __spec__.name if __spec__ is not None else __name__

# Where the environment's installed distributions live, never first-party code.
_INSTALLED = tuple(
    Path(path)
    for path in dict.fromkeys(sysconfig.get_paths()[key] for key in ("purelib", "platlib"))
)


class Guard(MetaPathFinder):
    """The finder that keeps a first-party import inside the closure.

    names: the top-level names the workspace's own import roots define.
    shipped: the files the closure holds, relative to `root`, the pinned snapshot or workspace.
    deferred: top-level names admitted regardless of `shipped`, left to the environment.
    """

    @classmethod
    def armed(cls, root: Path) -> Guard | None:
        """The guard the environment describes, first on `sys.meta_path`; None for a command."""
        listing = CLOSURE_VAR.read()
        if not listing:
            return None
        rows = Path(listing).read_text(encoding="utf-8").splitlines()
        guard = cls(
            FIRST_PARTY_VAR.read().split(":"),
            shipped=[row.split("\t", maxsplit=1)[0] for row in rows if row],
            root=root,
            deferred=DEFERRED_VAR.read().split(":"),
        )
        sys.meta_path.insert(0, guard)
        return guard

    def __init__(
        self,
        names: Sequence[str],
        *,
        shipped: Sequence[str],
        root: Path,
        deferred: Sequence[str] = (),
    ) -> None:
        self.names = frozenset(names)
        self.shipped = frozenset(shipped)
        self.root = root
        self.deferred = frozenset(deferred)

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        """The spec of a first-party module the closure ships, a refusal for one it does not."""
        del target
        if not self.guards(fullname):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and self.holds(spec):
            return spec
        raise ModuleNotFoundError(
            f"{fullname} is first-party code outside this job's closure; import it from the "
            "job file, or declare the file it lives in as a resource, so the dispatch ships it",
            name=fullname,
        )

    def guards(self, fullname: str) -> bool:
        top = fullname.partition(".")[0]
        return top in self.names and top not in self.deferred

    def holds(self, spec: ModuleSpec) -> bool:
        """Whether the closure ships the file `spec` was found at.

        A portion with no file passes, and so does an installed distribution's file, the other
        portions of a namespace the workspace shares with one.
        """
        if spec.origin is None:
            return True
        origin = Path(spec.origin)
        if origin.is_relative_to(self.root):
            return origin.relative_to(self.root).as_posix() in self.shipped
        return any(origin.is_relative_to(path) for path in _INSTALLED)

    def judged(self, fullname: str, spec: ModuleSpec) -> ModuleSpec:
        """The rewrite hook's `spec` when the closure lets `fullname` import it, else a refusal."""
        if self.guards(fullname) and not self.holds(spec):
            raise ModuleNotFoundError(
                f"{fullname} is first-party code outside this job's closure; import it from the "
                "job file, or declare its file as a resource, so the dispatch ships it",
                name=fullname,
            )
        return spec


def _main(argv: Sequence[str] | None = None) -> int:
    """Run one target, from the tokens after the runner's name (this process's when None)."""
    spelling, args = _target_and_args(list(sys.argv[1:] if argv is None else argv))
    fresh = Fresh.parsed(args)
    if fresh is not None:
        return fresh.run(spelling)
    file, _, name = spelling.partition(Target.SEPARATOR)
    guard = Guard.armed(Path.cwd())
    _pin_hub_cache()
    if Path(file).stem.startswith(Target.TEST_PREFIX):
        return _pytested(spelling if name else file, args, guard)
    return _called(getattr(_loaded(Path(file)), name), name, args)


def _target_and_args(tokens: list[str]) -> tuple[str, list[str]]:
    """The target's spelling and the arguments after it, past any delimiter before them."""
    if not tokens or tokens[0] == _DELIMITER:
        raise SystemExit(f"usage: python -m {_MODULE} <file>{Target.SEPARATOR}<name> [-- args]")
    spelling, args = tokens[0], tokens[1:]
    if args and args[0] == _DELIMITER:
        args = args[1:]
    return spelling, args


def _pin_hub_cache() -> None:
    """Point the Hub cache at the pins a job declared, whatever the host caches elsewhere."""
    pins = Path.cwd() / staging(Path.cwd())
    if pins.is_dir():
        os.environ["HF_HUB_CACHE"] = str(pins)


def _pytested(spelling: str, args: Sequence[str], guard: Guard | None) -> int:
    """Run a test target through pytest, with the guard already armed."""
    # The one environment-dependent import: an env without pytest still runs other targets.
    runner = importlib.import_module(".testing.pytest", package=__package__)
    return runner.Runner(guard).run([spelling, *args])


class Fresh:
    """Several cells of one lane, each run as its own fresh process by this same runner.

    `file.py::test --fresh id1 id2 -- <pytest args>`: a timed acquisition wants nothing of the
    previous cell (warm allocator, cached tokenizer, fragmented card), so each id runs as
    `python -m mainboard.jobs.call file.py::test[id] -- <pytest args>` in turn under a hard
    `timeout` (seconds), and the first failing cell stops the group.
    """

    FLAG = "--fresh"
    TIMEOUT = "--timeout"

    @classmethod
    def parsed(cls, tokens: Sequence[str]) -> Fresh | None:
        """The fresh plan the tokens spell, None when they name no `--fresh`."""
        if not tokens or tokens[0] != cls.FLAG:
            return None
        rest = list(tokens[1:])
        timeout = 900.0
        if rest and rest[0] == cls.TIMEOUT:
            if len(rest) < 2:
                raise SystemExit(f"{cls.TIMEOUT} takes the seconds one cell may run")
            timeout = float(rest[1])
            rest = rest[2:]
        ids, args = Fresh._partitioned(rest)
        if not ids:
            raise SystemExit(
                f"{cls.FLAG} takes the parametrize ids to run, then -- and pytest args"
            )
        return cls(ids, args=args, timeout=timeout)

    @staticmethod
    def _partitioned(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
        """The tokens before the delimiter and after it."""
        if _DELIMITER in tokens:
            cut = list(tokens).index(_DELIMITER)
            return list(tokens[:cut]), list(tokens[cut + 1 :])
        return list(tokens), []

    def __init__(self, ids: Sequence[str], *, args: Sequence[str], timeout: float) -> None:
        self.ids = tuple(ids)
        self.args = tuple(args)
        self.timeout = timeout

    def cells(self, spelling: str, *, dispatched: bool) -> int:
        nested = {**os.environ, **NESTED.exported("1")}
        for identity in self.ids:
            cell = f"{spelling}[{identity}]"
            logger.result(f"mainboard: fresh process for {cell}")
            command = [sys.executable, "-m", _MODULE, cell, _DELIMITER, *self.args]
            code = Processes().timeout(self.timeout, command, env=nested)
            if code == Exit.TIMED_OUT:
                logger.result(f"mainboard: {cell} exceeded {self.timeout:g} s and was killed")
                if dispatched:
                    beacon.say(beacon.Marker.CELL, f"failed {cell}")
            if code:
                return code
        return 0

    def run(self, spelling: str) -> int:
        """Run every id as its own process, answering the first nonzero exit code.

        In a dispatched job the lane is one session to a log reader: it declares every cell and
        ends the session itself, each child reports only its cell, and the lane reports a cell
        killed at its timeout as failed.
        """
        dispatched = RECEIPTS_VAR.present()
        if dispatched:
            beacon.say(beacon.Marker.CELLS, str(len(self.ids)))
        code = self.cells(spelling, dispatched=dispatched)
        if dispatched:
            beacon.say(beacon.Marker.SESSION, str(code))
        return code


def _loaded(file: Path) -> ModuleType:
    """Import `file` by the name its package chain gives it, from the root the chain stops at."""
    real = file.resolve()
    home = home_of(real, root=Path(real.anchor))
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    return importlib.import_module(dotted(real, home=home))


def _called(target: App | Callable[[], int | None], name: str, args: Sequence[str]) -> int:
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
    raise SystemExit(_main())
