import json

import pytest

from mainboard import MissionError
from mainboard.render import present


def test_mode_of_is_none_when_neither_flag_is_set() -> None:
    assert present.mode_of(json_mode=False, agent=False) is None


def test_mode_of_is_json_when_the_json_flag_is_set() -> None:
    assert present.mode_of(json_mode=True, agent=False) == "json"


def test_mode_of_is_agent_when_the_agent_flag_is_set() -> None:
    assert present.mode_of(json_mode=False, agent=True) == "agent"


def test_mode_of_rejects_both_flags_at_once() -> None:
    with pytest.raises(MissionError, match="only one"):
        present.mode_of(json_mode=True, agent=True)


def test_record_defaults_to_the_human_table(capsys: pytest.CaptureFixture[str]) -> None:
    present.record({"a": "1"}, mode=None, fields=(), title="thing")
    out = capsys.readouterr().out
    assert "thing" in out
    assert "1" in out


def test_record_json_mode_prints_the_canonical_dump(capsys: pytest.CaptureFixture[str]) -> None:
    present.record({"a": "1", "b": {"c": 2}}, mode="json", fields=(), title="")
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"a": "1", "b": {"c": 2}}


def test_record_json_mode_projects_to_the_given_fields(capsys: pytest.CaptureFixture[str]) -> None:
    present.record({"a": "1", "b": "2"}, mode="json", fields=("b",), title="")
    assert json.loads(capsys.readouterr().out) == {"b": "2"}


def test_record_agent_mode_prints_one_field_value_row_per_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    present.record({"a": "1", "b": "2"}, mode="agent", fields=(), title="")
    assert capsys.readouterr().out.strip() == "field\tvalue\na\t1\nb\t2"


def test_record_agent_mode_projects_to_the_given_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    present.record({"a": "1", "b": "2"}, mode="agent", fields=("b",), title="")
    assert capsys.readouterr().out.strip() == "field\tvalue\nb\t2"


def test_rows_defaults_to_the_human_table(capsys: pytest.CaptureFixture[str]) -> None:
    present.rows([{"a": "1"}, {"a": "2"}], mode=None, fields=(), title="jobs")
    out = capsys.readouterr().out
    assert "jobs" in out
    assert "1" in out
    assert "2" in out


def test_rows_json_mode_prints_a_canonical_list(capsys: pytest.CaptureFixture[str]) -> None:
    present.rows([{"a": "1"}, {"a": "2"}], mode="json", fields=(), title="")
    assert json.loads(capsys.readouterr().out) == [{"a": "1"}, {"a": "2"}]


def test_rows_json_mode_projects_to_the_given_fields(capsys: pytest.CaptureFixture[str]) -> None:
    present.rows([{"a": "1", "b": "2"}], mode="json", fields=("a",), title="")
    assert json.loads(capsys.readouterr().out) == [{"a": "1"}]


def test_rows_agent_mode_prints_the_tabular_encoding(capsys: pytest.CaptureFixture[str]) -> None:
    present.rows([{"a": "1"}, {"a": "2"}], mode="agent", fields=(), title="")
    assert capsys.readouterr().out.strip() == "a\n1\n2"
