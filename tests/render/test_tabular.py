import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.render import tabular

_HOSTS = ("gold", "miyabi-g", "miyabi-debug", "crimson", "local")
_STATES = ("queued", "running", "ok", "failed", "vanished")
_NAMES = ("train-run", "eval-sweep", "compress-30b", "tokenizer-bench", "exp.run", "")


def _jobs(count: int) -> list[dict[str, str]]:
    """A realistic jobs payload, the shape the `jobs` CLI verb renders."""
    return [
        {
            "handle": str(10000 + i),
            "host": _HOSTS[i % len(_HOSTS)],
            "name": _NAMES[i % len(_NAMES)],
            "state": _STATES[i % len(_STATES)],
            "submitted_at": f"2026-08-{(i % 28) + 1:02d}T{i % 24:02d}:{i % 60:02d}:00+00:00",
        }
        for i in range(count)
    ]


def test_encode_writes_a_header_then_one_line_per_row() -> None:
    text = tabular.encode([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
    assert text == "a\tb\n1\t2\n3\t4"


def test_encode_projects_to_the_given_fields_in_order() -> None:
    text = tabular.encode([{"a": "1", "b": "2", "c": "3"}], fields=["c", "a"])
    assert text == "c\ta\n3\t1"


def test_encode_renders_a_missing_field_as_an_empty_cell() -> None:
    text = tabular.encode([{"a": "1"}], fields=["a", "b"])
    assert text == "a\tb\n1\t"


def test_encode_renders_none_as_an_empty_cell() -> None:
    text = tabular.encode([{"a": None}])
    assert text == "a\n"


@pytest.mark.parametrize(
    ("cell", "encoded"),
    [
        ("one\ttwo", 'a\n"one\\ttwo"'),
        ("one\ntwo", 'a\n"one\\ntwo"'),
        # A plain leading quote would otherwise be indistinguishable from an escaped cell on
        # decode, so it JSON-encodes too.
        ('"quoted', 'a\n"\\"quoted"'),
    ],
)
def test_encode_json_encodes_a_cell_needing_escape(*, cell: str, encoded: str) -> None:
    assert tabular.encode([{"a": cell}]) == encoded


def test_encode_on_no_rows_and_no_fields_is_empty() -> None:
    assert not tabular.encode([])


def test_decode_reverses_a_plain_encoding() -> None:
    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    assert tabular.decode(tabular.encode(rows)) == rows


def test_decode_reverses_an_escaped_cell() -> None:
    rows = [{"a": "one\ttwo\nthree"}]
    assert tabular.decode(tabular.encode(rows)) == rows


def test_decode_of_the_empty_text_is_no_rows() -> None:
    assert tabular.decode("") == []


def test_decode_of_a_header_only_encoding_is_no_rows() -> None:
    assert tabular.decode(tabular.encode([], fields=["a", "b"])) == []


@given(
    st.lists(
        st.dictionaries(
            st.sampled_from(["handle", "host", "name", "state"]),
            st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20),
            min_size=4,
            max_size=4,
        ),
        max_size=10,
    )
)
def test_decode_of_encode_round_trips_any_row_set(rows: list[dict[str, str]]) -> None:
    assert tabular.decode(tabular.encode(rows)) == rows


def test_tabular_is_smaller_than_canonical_json_on_a_realistic_jobs_payload() -> None:
    """Correctness-gated, not perf-gated: measures byte length, no timing assertion."""
    payload = _jobs(50)
    canonical = json.dumps(payload, indent=2)
    compact = tabular.encode(payload)

    assert len(compact.encode()) < len(canonical.encode()) * 0.5
    assert tabular.decode(compact) == [
        {key: str(value) for key, value in row.items()} for row in payload
    ]
