"""A plan takes one finding's edits only when they overlap none it holds. A claim holds a range
against the claims of other groups and changes no byte, so a group that needs a block as it stood
when the pass began waits a pass when another group already holds that block."""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mainboard.analysis import Span
from mainboard.analysis.claims import Claim
from mainboard.analysis.edits import Delete, Edit
from mainboard.analysis.engine.apply import Plan

_PATH = "module.py"
_LINES = [f"line {number}\n" for number in range(1, 21)]
# The lines `first` up to, not including, `last`: the bounds of a claim, in order.
_RANGES = st.integers(1, len(_LINES)).flatmap(
    lambda first: st.tuples(st.just(first), st.integers(first + 1, len(_LINES) + 1))
)


def _claim(lines: tuple[int, int]) -> Claim:
    return Claim(span=Span(path=_PATH, start_line=lines[0], end_line=lines[1]))


def _deletion(line: int) -> Delete:
    return Delete(span=Span(path=_PATH, start_line=line))


def _kept(*gone: int) -> bytes:
    """The file without the lines numbered `gone`."""
    return "".join(text for number, text in enumerate(_LINES, 1) if number not in gone).encode()


def _planned(root: Path) -> Plan:
    (root / _PATH).write_text("".join(_LINES), encoding="utf-8")
    return Plan(root)


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(first=(1, 3), second=(3, 5))
@example(first=(2, 6), second=(2, 6))
@given(first=_RANGES, second=_RANGES)
def test_a_second_claim_on_an_overlapping_range_waits_a_pass(
    tmp_path: Path, first: tuple[int, int], second: tuple[int, int]
) -> None:
    plan = _planned(tmp_path)
    plan.add([_claim(first)])

    taken = plan.add([_claim(second)])

    assert taken == (first[1] <= second[0] or second[1] <= first[0])
    assert plan.deferred == (not taken)


@pytest.mark.parametrize(
    ("edits", "written"),
    [
        pytest.param([_claim((1, 5))], {}, id="a-claim-alone"),
        pytest.param(
            [_deletion(2), _claim((1, 5))], {_PATH: _kept(2)}, id="a-claim-beside-a-deletion"
        ),
    ],
)
def test_a_claim_changes_no_byte_of_the_written_file(
    tmp_path: Path, edits: list[Edit], written: dict[str, bytes]
) -> None:
    plan = _planned(tmp_path)

    plan.add(edits)

    assert plan.written() == written
    assert plan.touched() == set(written)


def test_a_group_whose_claim_is_taken_leaves_every_one_of_its_edits_for_the_next_pass(
    tmp_path: Path,
) -> None:
    plan = _planned(tmp_path)
    plan.add([_deletion(2), _claim((1, 5))])

    taken = plan.add([_deletion(3), _claim((2, 6))])

    assert not taken
    assert plan.written() == {_PATH: _kept(2)}


@pytest.mark.parametrize(
    ("held", "later"),
    [
        pytest.param([_claim((1, 5))], [_deletion(2)], id="a-deletion-inside-a-claim"),
        pytest.param([_deletion(2)], [_claim((1, 5))], id="a-claim-around-a-deletion"),
    ],
)
def test_a_claim_collides_with_claims_only(
    tmp_path: Path, held: list[Edit], later: list[Edit]
) -> None:
    plan = _planned(tmp_path)
    plan.add(held)

    taken = plan.add(later)

    assert taken
    assert plan.deferred == 0
