"""Native pytest integration, loaded only for test targets."""

import os
import sys
from collections.abc import Sequence
from contextlib import suppress
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import TYPE_CHECKING

import pytest
from _pytest import assertion
from _pytest.assertion.rewrite import AssertionRewritingHook
from _pytest.config import Config

from ..dispatch.evidence import RECEIPTS_VAR
from .beacon import CELL, CELLS, NESTED, SESSION, say

if TYPE_CHECKING:
    from .call import Guard


class Judged(MetaPathFinder):
    """Keep pytest's rewrite loader, checking its origin before execution."""

    def __init__(self, hook: AssertionRewritingHook, guard: Guard) -> None:
        self.hook, self.guard = hook, guard

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        spec = self.hook.find_spec(fullname, path, target)
        return None if spec is None else self.guard.judged(fullname, spec)


class Beacon:
    """The pytest plugin a dispatched test job reports its cells through."""

    def __init__(self, *, nested: bool) -> None:
        """nested: whether this session is one cell of a fresh-process lane."""
        self.nested = nested

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Declare how many cells the session holds, once, before the first one runs."""
        if not self.nested:
            say(CELLS, str(len(session.items)))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Report a cell's outcome the moment a phase decides it.

        The call phase decides a cell that ran; a setup that skipped or failed decides one that
        never did, and a teardown that failed turns a passed cell failed.
        """
        if report.when == "call" or report.outcome != "passed":
            say(CELL, f"{report.outcome} {report.nodeid}")

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        del session
        if not self.nested:
            say(SESSION, str(int(exitstatus)))


class Runner:
    """Install the boundary with pytest's hook, before any plugin or conftest loads.

    pytest offers no public hook at this point, so its installer is replaced for this invocation
    only and restored with the meta-path entries even on failure; the original hook stays
    pytest's loader. A dispatched job (its wrapper staged a receipts file) also reports each cell
    through the beacon, since a waiter reads its progress off the log.
    """

    def __init__(self, guard: Guard | None) -> None:
        self.guard = guard
        self.original = assertion.install_importhook
        self.installed: list[Judged] = []

    def run(self, args: Sequence[str]) -> int:
        try:
            assertion.install_importhook = self.install
            return pytest.main(list(args), plugins=self.plugins())
        finally:
            assertion.install_importhook = self.original
            for finder in self.installed:
                with suppress(ValueError):
                    sys.meta_path.remove(finder)

    @staticmethod
    def plugins() -> list[Beacon]:
        """The beacon in a dispatched job, nothing at a terminal."""
        if RECEIPTS_VAR not in os.environ:
            return []
        return [Beacon(nested=NESTED in os.environ)]

    def install(self, config: Config) -> AssertionRewritingHook:
        hook = self.original(config)
        if self.guard is not None:
            finder = Judged(hook, self.guard)
            sys.meta_path[sys.meta_path.index(hook)] = finder
            self.installed.append(finder)
        return hook
