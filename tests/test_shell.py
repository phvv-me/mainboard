from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mainboard import shell

run_mod = sys.modules["mainboard.shell.run"]
sysctl_mod = sys.modules["mainboard.shell.sysctl"]
sp_mod = sys.modules["mainboard.shell.system_profiler"]
whoami_mod = sys.modules["mainboard.shell.whoami"]


class FakeCommand:
    """Mimic a plumbum bound command: indexing binds args, calling returns output."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[object, ...]] = []

    def __getitem__(self, args: object) -> FakeCommand:
        return self

    def __call__(self, *args: object) -> str:
        self.calls.append(args)
        return self.output


class BoomCommand:
    """A plumbum command stand-in that raises on use."""

    def __getitem__(self, args: object) -> BoomCommand:
        return self

    def __call__(self, *args: object) -> str:
        raise OSError("missing tool")


def test_run_returns_stdout_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` invokes the program once per argv and returns its stdout."""
    command = FakeCommand("clang version 17.0.0\n")
    monkeypatch.setattr(run_mod, "local", {"clang": command})
    run_mod.run.cache_clear()
    first = shell.run("clang", "--version")
    second = shell.run("clang", "--version")
    assert first == second == "clang version 17.0.0\n"
    assert command.calls == [("--version",)]
    run_mod.run.cache_clear()


def test_sysctl_reads_and_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sysctl` returns the stripped value for a known key."""
    monkeypatch.setattr(sysctl_mod, "local", {"sysctl": FakeCommand("Apple M4 Pro\n")})
    assert shell.sysctl("machdep.cpu.brand_string") == "Apple M4 Pro"


@pytest.mark.parametrize(
    "local",
    [
        pytest.param({"sysctl": BoomCommand()}, id="missing-tool-oserror"),
        pytest.param({}, id="unknown-key-keyerror"),
    ],
)
def test_sysctl_tolerates_failure(
    local: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both a missing binary (`OSError`) and an unknown command (`KeyError`) yield ``""``."""
    monkeypatch.setattr(sysctl_mod, "local", local)
    assert shell.sysctl("kern.osrelease") == ""


def test_system_profiler_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """`system_profiler` parses the JSON payload for the requested datatype."""
    payload = json.dumps({"SPHardwareDataType": [{"chip_type": "Apple M4 Pro"}]})
    monkeypatch.setattr(sp_mod, "local", {"system_profiler": FakeCommand(payload)})
    sp_mod.system_profiler.cache_clear()
    profile = shell.system_profiler("SPHardwareDataType")
    assert profile["SPHardwareDataType"][0]["chip_type"] == "Apple M4 Pro"
    sp_mod.system_profiler.cache_clear()


@pytest.mark.parametrize(
    "command",
    [
        pytest.param(BoomCommand(), id="missing-tool-oserror"),
        pytest.param(FakeCommand("not json"), id="bad-json"),
    ],
)
def test_system_profiler_tolerates_failure(
    command: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing tool (`OSError`) and unparseable output (`JSONDecodeError`) both give ``{}``."""
    monkeypatch.setattr(sp_mod, "local", {"system_profiler": command})
    sp_mod.system_profiler.cache_clear()
    assert shell.system_profiler("SPHardwareDataType") == {}
    sp_mod.system_profiler.cache_clear()


def test_read_dmi_strips_present_field_and_tolerates_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`read_dmi` reads and strips a present DMI field and gives ``""`` for an absent one."""
    (tmp_path / "board_vendor").write_text("  ASUSTeK  \n")
    monkeypatch.setattr(shell.sysfs, "DMI_ROOT", tmp_path)
    assert shell.read_dmi("board_vendor") == "ASUSTeK"
    assert shell.read_dmi("board_name") == ""


def test_whoami_groups_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """`whoami_groups` returns the group name from each CSV row of `whoami /groups`."""
    output = (
        '"Everyone","Well-known group","S-1-1-0","Mandatory group"\r\n'
        '"BUILTIN\\Users","Alias","S-1-5-32-545","Enabled"\r\n'
    )
    monkeypatch.setattr(whoami_mod, "local", {"whoami": FakeCommand(output)})
    assert shell.whoami_groups() == ("Everyone", "BUILTIN\\Users")


def test_whoami_groups_tolerates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing or failing `whoami` yields () rather than raising."""
    monkeypatch.setattr(whoami_mod, "local", {"whoami": BoomCommand()})
    assert shell.whoami_groups() == ()
