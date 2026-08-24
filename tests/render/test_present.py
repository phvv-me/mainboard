import json
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.render import present

if TYPE_CHECKING:
    from collections.abc import Sequence

_ONE = {"a": "1", "b": {"c": 2}}
_MANY = [{"a": "1"}, {"a": "2"}]


@pytest.mark.parametrize(
    ("json_mode", "agent", "expected"),
    [(False, False, None), (True, False, "json"), (False, True, "agent")],
)
def test_the_mode_follows_whichever_flag_was_passed(
    *, json_mode: bool, agent: bool, expected: str | None
) -> None:
    """No flag is the human render, and each compact flag names its own dispatch key."""
    assert present.mode_of(json_mode=json_mode, agent=agent) == expected


def test_the_two_compact_modes_refuse_to_be_asked_for_at_once() -> None:
    """`--json` and `--agent` are two answers to one question, so asking both is a mistake."""
    with pytest.raises(MissionError, match="only one"):
        present.mode_of(json_mode=True, agent=True)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, ("thing", "1")),
        ("json", ('"a": "1"', '"c": 2')),
        ("agent", ("field\tvalue", "a\t1")),
    ],
)
def test_one_entity_renders_down_its_own_rows_in_every_mode(
    capsys: pytest.CaptureFixture[str], mode: str | None, expected: Sequence[str]
) -> None:
    """An entity has more fields than a terminal is wide, so it goes down rather than across."""
    present.record(_ONE, mode=mode, fields=(), title="thing")
    printed = capsys.readouterr().out
    assert all(token in printed for token in expected)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, ("jobs", "1", "2")),
        ("json", ('"a": "1"', '"a": "2"')),
        ("agent", ("a\n1\n2",)),
    ],
)
def test_many_entities_render_across_columns_in_every_mode(
    capsys: pytest.CaptureFixture[str], mode: str | None, expected: Sequence[str]
) -> None:
    """Many records share one header, which is what makes the compact modes compact."""
    present.rows(_MANY, mode=mode, fields=(), title="jobs")
    printed = capsys.readouterr().out
    assert all(token in printed for token in expected)


def test_a_projection_narrows_every_compact_mode_to_the_named_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--fields` is the same promise for one entity or many, in JSON and in the agent text."""
    present.record({"a": "1", "b": "2"}, mode="json", fields=("b",), title="")
    assert json.loads(capsys.readouterr().out) == {"b": "2"}
    present.record({"a": "1", "b": "2"}, mode="agent", fields=("b",), title="")
    assert capsys.readouterr().out.strip() == "field\tvalue\nb\t2"
    present.rows([{"a": "1", "b": "2"}], mode="json", fields=("a",), title="")
    assert json.loads(capsys.readouterr().out) == [{"a": "1"}]
    present.rows([{"a": "1", "b": "2"}], mode="agent", fields=("a",), title="")
    assert capsys.readouterr().out.strip() == "a\n1"
