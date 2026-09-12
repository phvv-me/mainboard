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
#
# A DEFERRED NAME IS THE ONE EXCEPTION. A package whose compiled extension the closure found
# living outside the tree ships none of it, on purpose, and its whole distribution is left for
# the environment's own install to answer; asking `shipped` about it would refuse every import
# of it by name, which is not a walk that missed a module but a closure that never carried one.
# So the guard steps aside for these names instead, the same way it already does for anything
# that is not first-party at all.
#
# A TEST FILE IS PYTEST'S TO RUN. A `test_` target is handed to `pytest.main` as the node id it
# spells, fixtures and parametrization and exit status all native, with the guard already armed
# so pytest's imports of the shipped modules answer from this tree. pytest's rewrite hook sits
# ahead of the guard on `meta_path` and finds modules on its own walk. The native adapter
# wraps that hook at installation, before config/environment plugins or initial conftests
# import. Its spec is checked before execution, with assertion rewriting left intact.

import importlib
import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs this same runner per cell from typed tokens, never a shell string since=2026-09-12
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from cyclopts import App

from ..dispatch.shared import CLOSURE_VAR, DEFERRED_VAR, FIRST_PARTY_VAR
from .pins import STAGING
from .target import SEPARATOR, TEST_PREFIX, dotted, home_of

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# What separates the target from the arguments handed to it.
DELIMITER = "--"


class Guard(MetaPathFinder):
    """The finder that keeps a first-party import inside the closure.

    names: the top-level names the workspace's own import roots define.
    shipped: the files the closure holds, relative to `root`.
    root: the tree the job stands in, the pinned snapshot or the workspace.
    deferred: top-level names admitted regardless of `shipped`, whose whole distribution the
        closure left to the environment rather than shipping half of it.
    """

    def __init__(
        self,
        names: Sequence[str],
        shipped: Sequence[str],
        root: Path,
        *,
        deferred: Sequence[str] = (),
    ) -> None:
        self.names = frozenset(names)
        self.shipped = frozenset(shipped)
        self.root = root
        self.deferred = frozenset(deferred)

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
            deferred=os.environ.get(DEFERRED_VAR, "").split(":"),
        )
        sys.meta_path.insert(0, guard)
        return guard

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        """The spec of a first-party module the closure ships, a refusal for one it does not."""
        del target
        top = fullname.partition(".")[0]
        if top not in self.names or top in self.deferred:
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

    def judged(self, fullname: str, spec: ModuleSpec) -> ModuleSpec:
        """`spec` when the closure lets `fullname` import it, a refusal when it does not.

        The rewrite hook's answer is judged here before it can execute, since the hook found
        the module on its own walk and would otherwise import first-party code from wherever
        the environment points.
        """
        top = fullname.partition(".")[0]
        if top in self.names and top not in self.deferred and not self.holds(spec):
            raise ModuleNotFoundError(
                f"{fullname} is first-party code outside this job's closure; import it from the "
                "job file, or declare its file as a resource, so the dispatch ships it",
                name=fullname,
            )
        return spec


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
    fresh = Fresh.parsed(args)
    if fresh is not None:
        return fresh.run(spelling)
    file, _, name = spelling.partition(SEPARATOR)
    guard = Guard.armed(Path.cwd())
    pins = Path.cwd() / STAGING
    if pins.is_dir():
        # A job that declared Hub pins reads exactly them, from the tree, whatever the host
        # caches elsewhere.
        os.environ["HF_HUB_CACHE"] = str(pins)
    if Path(file).stem.startswith(TEST_PREFIX):
        # The one environment-dependent import: an env that declares no pytest still runs every
        # other target, and a test target in one fails here naming what is missing.
        runner = importlib.import_module(".pytest", package=__package__)
        return runner.Runner(guard).run([spelling if name else file, *args])
    return called(getattr(loaded(Path(file)), name), name, args)


class Fresh:
    """Several cells of one lane, each run as its own fresh process by this same runner.

    `file.py::test --fresh id1 id2 -- <pytest args>`: a timed acquisition wants nothing of the
    cell before it, no warm allocator, no cached tokenizer, no fragmented card, so every id
    becomes `python -m mainboard.jobs.call file.py::test[id] -- <pytest args>` in turn, under a
    hard timeout, and the first cell that fails stops the group.

    ids: the parametrize ids to run, in order.
    args: the pytest arguments every cell gets.
    timeout: seconds one cell may take before it is killed.
    """

    FLAG = "--fresh"
    TIMEOUT = "--timeout"

    def __init__(self, ids: Sequence[str], args: Sequence[str], timeout: float) -> None:
        self.ids = tuple(ids)
        self.args = tuple(args)
        self.timeout = timeout

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
        ids, _, args = _partitioned(rest)
        if not ids:
            raise SystemExit(
                f"{cls.FLAG} takes the parametrize ids to run, then -- and pytest args"
            )
        return cls(ids, args, timeout)

    def run(self, spelling: str) -> int:
        """Run every id as its own process, answering the first nonzero exit code."""
        for identity in self.ids:
            cell = f"{spelling}[{identity}]"
            print(f"mainboard: fresh process for {cell}", flush=True)
            command = [sys.executable, "-m", __spec__.name, cell, DELIMITER, *self.args]
            try:
                completed = subprocess.run(command, check=False, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                print(f"mainboard: {cell} exceeded {self.timeout:g} s and was killed", flush=True)
                return 124
            if completed.returncode:
                return completed.returncode
        return 0


def _partitioned(tokens: Sequence[str]) -> tuple[list[str], bool, list[str]]:
    """The tokens before the delimiter, whether one was present, and the tokens after it."""
    if DELIMITER in tokens:
        cut = list(tokens).index(DELIMITER)
        return list(tokens[:cut]), True, list(tokens[cut + 1 :])
    return list(tokens), False, []


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
