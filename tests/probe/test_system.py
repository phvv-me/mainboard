from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from packaging.version import Version

from mainboard.probe.census import Census, Json
from mainboard.probe.system import Card, System

# Capabilities as a driver may spell them, readable or not, so the maxima skip the unreadable.
_CAPABILITIES = st.sampled_from(["", "7.5", "8.9", "9.0", "12.0", "[N/A]"])

_CARDS = st.lists(
    st.builds(
        Card,
        name=st.just("card"),
        capability=_CAPABILITIES,
        vram_mb=st.integers(min_value=0, max_value=200_000),
    ),
    max_size=4,
).map(tuple)


def test_collected_is_this_machines_census_of_the_given_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model is the census read back whole, asked where the workspace lives."""
    asked: list[str] = []

    def survey(self: Census, root: str) -> dict[str, Json]:
        asked.append(root)
        return {"system": "Linux", "arch": "x86_64", "gpus": [{"name": "RTX 4090"}]}

    monkeypatch.setattr(Census, "survey", survey)

    system = System.collected(tmp_path)

    assert asked == [str(tmp_path)]
    assert (system.platform, system.gpus) == ("linux-64", (Card(name="RTX 4090"),))


@pytest.mark.parametrize(
    ("system", "arch", "platform", "windows"),
    [
        ("", "", "", False),
        ("Linux", "aarch64", "linux-aarch64", False),
        ("Darwin", "arm64", "osx-arm64", False),
        ("Windows", "AMD64", "win-64", True),
    ],
    ids=["never surveyed", "linux", "macos", "windows"],
)
def test_a_machine_is_named_as_the_pixi_platform_it_installs_for(
    system: str, arch: str, platform: str, windows: bool
) -> None:
    """A host an older tool onboarded has no census, and then claims no platform at all.

    An empty platform can never match a declared one, so a missing census is never mistaken
    for a supported machine.
    """
    machine = System(system=system, arch=arch)
    assert (machine.surveyed, machine.platform, machine.windows) == (
        bool(system),
        platform,
        windows,
    )


@given(gpus=_CARDS)
def test_the_card_figures_are_the_largest_and_newest_readable_ones(gpus: tuple[Card, ...]) -> None:
    """A job fits the biggest card and runs on the newest kernels any card can take.

    A capability the driver could not report is skipped rather than failing the whole model,
    and a machine with no card answers zero memory and no capability.
    """
    machine = System(gpus=gpus)
    readable = [Version(card.capability) for card in gpus if card.capability[:1].isdigit()]
    assert machine.vram_mb == max((card.vram_mb for card in gpus), default=0)
    assert machine.capability == max(readable, default=None)


@pytest.mark.parametrize(
    ("version", "cuda", "driver", "summary"),
    [
        ("Ubuntu 24.04", "13.0", Version("13.0"), "Ubuntu 24.04 x86_64, driver CUDA 13.0"),
        ("Ubuntu 24.04", "", None, "Ubuntu 24.04 x86_64"),
        ("Ubuntu 24.04", "unknown", None, "Ubuntu 24.04 x86_64, driver CUDA unknown"),
        ("", "", None, "Linux x86_64"),
    ],
    ids=["a driver", "no driver", "an unreadable driver", "no os version"],
)
def test_the_driver_cuda_is_comparable_or_absent_and_the_summary_names_it(
    version: str, cuda: str, driver: Version | None, summary: str
) -> None:
    """An unreadable driver version compares as no driver, never as a crash in a report.

    A census that found no OS version still names the kernel it looked at.
    """
    machine = System(system="Linux", version=version, arch="x86_64", cuda=cuda)
    assert (machine.driver_cuda, machine.summary()) == (driver, summary)
