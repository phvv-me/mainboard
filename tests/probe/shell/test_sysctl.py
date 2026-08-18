import sys

import pytest

from mainboard.probe.shell import sysctl

# see test_run.py for why this goes through `sys.modules` rather than a dotted import.
sysctl_mod = sys.modules["mainboard.probe.shell.sysctl"]


class FakeCommand:
    def __init__(self, output: str) -> None:
        self.output = output

    def __call__(self, *args: str) -> str:
        return self.output

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> FakeCommand:
        return self


class BoomCommand:
    def __call__(self, *args: str) -> str:
        raise OSError("missing tool")

    def __getitem__(self, args: str | list[str] | tuple[str, ...]) -> BoomCommand:
        return self


def test_sysctl_reads_and_strips(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sysctl` returns the stripped value for a known key."""
    monkeypatch.setattr(sysctl_mod, "local", {"sysctl": FakeCommand("Apple M4 Pro\n")})
    assert sysctl("machdep.cpu.brand_string") == "Apple M4 Pro"


@pytest.mark.parametrize(
    "local",
    [
        pytest.param({"sysctl": BoomCommand()}, id="missing-tool-oserror"),
        pytest.param({}, id="unknown-key-keyerror"),
    ],
)
def test_sysctl_tolerates_failure(
    local: dict[str, BoomCommand], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing binary (`OSError`) and an unknown command (`KeyError`) both yield ""."""
    monkeypatch.setattr(sysctl_mod, "local", local)
    assert sysctl("kern.osrelease") == ""
