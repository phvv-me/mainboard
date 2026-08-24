import sys
from pathlib import Path
from typing import NoReturn

import pytest

from mainboard.probe.shell import read_dmi, run, sysctl

# `shell/__init__.py` rebinds each name to the function, so these reach the real submodules
# through `sys.modules` instead of an attribute lookup that would land on the function.
run_mod = sys.modules["mainboard.probe.shell.run"]
sysctl_mod = sys.modules["mainboard.probe.shell.sysctl"]
sysfs_mod = sys.modules["mainboard.probe.shell.sysfs"]


class FakeCommand:
    """Mimic a plumbum bound command, indexing binds arguments and calling returns output."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        return self.output

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> FakeCommand:
        return self


class BoomCommand:
    """A command whose binary is missing, the shape plumbum reports on a foreign platform."""

    def __call__(self, *args: str) -> NoReturn:
        raise OSError("missing tool")

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> BoomCommand:
        return self


def test_run_returns_stdout_and_executes_the_program_once_per_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity probes ask the same binary the same question repeatedly, so argv keys a cache."""
    command = FakeCommand("clang version 17.0.0\n")
    monkeypatch.setattr(run_mod, "local", {"clang": command})
    run_mod.run.cache_clear()
    assert run("clang", "--version") == run("clang", "--version") == "clang version 17.0.0\n"
    assert command.calls == [("--version",)]
    run_mod.run.cache_clear()


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        pytest.param({"sysctl": FakeCommand("Apple M4 Pro\n")}, "Apple M4 Pro", id="present"),
        pytest.param({"sysctl": BoomCommand()}, "", id="missing-binary"),
        pytest.param({}, "", id="unknown-command"),
    ],
)
def test_sysctl_strips_a_reading_and_answers_empty_when_the_key_is_unreachable(
    local: dict[str, FakeCommand | BoomCommand], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callers probe Darwin-only keys without guarding the platform, so a missing binary
    (`OSError`) and an unknown command (`KeyError`) both read as no value at all."""
    monkeypatch.setattr(sysctl_mod, "local", local)
    assert sysctl("machdep.cpu.brand_string") == expected


def test_read_dmi_strips_a_present_field_and_answers_empty_for_an_absent_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DMI files are Linux-only pseudo-files, so an absent one is a value of `""`, not a raise."""
    (tmp_path / "board_vendor").write_text("  ASUSTeK  \n")
    monkeypatch.setattr(sysfs_mod, "_DMI_ROOT", tmp_path)
    assert read_dmi("board_vendor") == "ASUSTeK"
    assert read_dmi("board_name") == ""
