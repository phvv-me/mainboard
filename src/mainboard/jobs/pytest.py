"""Native pytest integration, loaded only for test targets."""

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


class Runner:
    """Install the boundary with pytest's hook, before any plugin or conftest loads.

    pytest offers no public hook at this point. Its installer is replaced only during this
    invocation, and both the installer and meta-path entries are restored even on failure.
    The original hook remains pytest's loader and receives its normal rewrite registrations.
    """

    def __init__(self, guard: Guard | None) -> None:
        self.guard = guard
        self.original = assertion.install_importhook
        self.installed: list[Judged] = []

    def run(self, args: Sequence[str]) -> int:
        try:
            assertion.install_importhook = self.install
            return pytest.main(list(args))
        finally:
            assertion.install_importhook = self.original
            for finder in self.installed:
                with suppress(ValueError):
                    sys.meta_path.remove(finder)

    def install(self, config: Config) -> AssertionRewritingHook:
        hook = self.original(config)
        if self.guard is not None:
            finder = Judged(hook, self.guard)
            sys.meta_path[sys.meta_path.index(hook)] = finder
            self.installed.append(finder)
        return hook
