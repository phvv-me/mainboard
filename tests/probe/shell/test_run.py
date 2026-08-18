import sys
from typing import TYPE_CHECKING

from mainboard.probe.shell import run

if TYPE_CHECKING:
    import pytest

# `shell.__init__` rebinds its own `run` attribute to the function, so this reaches the
# real submodule via `sys.modules` instead of an attribute lookup.
run_mod = sys.modules["mainboard.probe.shell.run"]


class FakeCommand:
    """Mimic a plumbum bound command: indexing binds args, calling returns output."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        return self.output

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> FakeCommand:
        return self


def test_run_returns_stdout_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` invokes the program once per argv and returns its stdout."""
    command = FakeCommand("clang version 17.0.0\n")
    monkeypatch.setattr(run_mod, "local", {"clang": command})
    run_mod.run.cache_clear()
    first = run("clang", "--version")
    second = run("clang", "--version")
    assert first == second == "clang version 17.0.0\n"
    assert command.calls == [("--version",)]
    run_mod.run.cache_clear()
