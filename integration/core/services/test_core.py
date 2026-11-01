"""The service-manager seam itself: periods, the bounded shell, and the platform with neither."""

import platform
import sys
from operator import methodcaller
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.core.services import locally, resident, settler
from mainboard.core.services.base import Every, Unsupported, UnsupportedUnit, complaint

_SECONDS = {"s": 1, "m": 60, "h": 3600}


@given(count=st.integers(1, 10_000), unit=st.sampled_from(sorted(_SECONDS)))
@example(count=20, unit="m")
def test_a_period_is_a_whole_number_of_one_unit_and_keeps_its_spelling(
    count: int, unit: str
) -> None:
    every = Every.parse(f" {count}{unit} ")
    assert (every.seconds, every.written) == (count * _SECONDS[unit], f"{count}{unit}")


@pytest.mark.parametrize(
    "written",
    [
        pytest.param("20", id="a-number-without-a-unit"),
        pytest.param("1d", id="a-unit-systemd-would-read-differently"),
        pytest.param("1h30m", id="two-units"),
        pytest.param("-5m", id="a-negative-period"),
    ],
)
def test_a_period_written_any_other_way_is_refused_naming_the_spellings(written: str) -> None:
    with pytest.raises(MissionError, match="a period is written like 20m, 1h or 90s"):
        Every.parse(written)


def test_a_zero_period_is_the_one_spelling_that_removes_the_pass() -> None:
    assert Every.parse("0").seconds == 0


@given(lines=st.lists(st.text()))
def test_a_service_managers_complaint_is_the_last_line_it_said(lines: list[str]) -> None:
    spoken = [line.strip() for line in "\n".join(lines).splitlines() if line.strip()]
    assert complaint("\n".join(lines)) == (spoken[-1] if spoken else "it said nothing")


def test_a_program_this_machine_lacks_is_an_answer_not_an_exception() -> None:
    assert locally(("mb-no-such-program-anywhere",)) == (
        1,
        "mb-no-such-program-anywhere is not installed here",
    )


def test_a_platform_with_no_service_manager_refuses_both_entries_by_name(tmp_path: Path) -> None:
    unit = UnsupportedUnit(tmp_path)
    with pytest.raises(MissionError, match="so `serve` cannot run here"):
        unit.install(("/bin/true",), stop_seconds=1, log=tmp_path / "unit.log")
    with pytest.raises(MissionError, match="sweep with `mb job list`"):
        Unsupported(tmp_path).install(Every.parse("20m"))
    unit.remove()
    assert (unit.installed(), unit.loaded(), list(tmp_path.iterdir())) == (False, False, [])


def test_a_machine_offers_its_own_service_managers_entries_for_a_workspace(
    tmp_path: Path,
) -> None:
    offered = {"darwin": ("LaunchdAgent", "LaunchdUnit"), "linux": ("SystemdUser", "SystemdUnit")}
    found = (type(settler(tmp_path)).__name__, type(resident(tmp_path)).__name__)
    assert found == offered[sys.platform]


@pytest.mark.parametrize(
    "verb", [pytest.param("start", id="starting"), pytest.param("stop", id="stopping")]
)
def test_a_platform_with_no_service_manager_refuses_to_start_and_to_stop_the_settler(
    tmp_path: Path, verb: str
) -> None:
    with pytest.raises(MissionError, match=f"{platform.system()} has no service manager"):
        methodcaller(verb)(Unsupported(tmp_path))
