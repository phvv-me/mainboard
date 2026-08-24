import json
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.render import tabular

from ..strategies import TEXT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mainboard.render.values import Row

_HOSTS = ("gold", "miyabi-g", "miyabi-debug", "crimson", "local")
_STATES = ("queued", "running", "ok", "failed", "vanished")
_NAMES = ("train-run", "eval-sweep", "compress-30b", "tokenizer-bench", "exp.run", "")

# Every row carries the same four columns, since a header written once is the whole point of
# the format and a row missing a column is a projection rather than a round trip.
_ROWS = st.lists(
    st.dictionaries(
        st.sampled_from(["handle", "host", "name", "state"]), TEXT, min_size=4, max_size=4
    ),
    max_size=6,
)


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


@given(rows=_ROWS)
@example(rows=[{"a": "one\ttwo"}, {"a": "one\ntwo"}, {"a": '"quoted'}, {"a": ""}])
def test_decoding_an_encoding_returns_the_rows_that_went_in(rows: list[dict[str, str]]) -> None:
    """The escape rule closes on itself, so a tab, a newline and a leading quote all survive."""
    assert tabular.decode(tabular.encode(rows)) == rows


@pytest.mark.parametrize(
    ("rows", "fields", "expected"),
    [
        ([{"a": "1", "b": "2"}, {"a": "3", "b": "4"}], None, "a\tb\n1\t2\n3\t4"),
        ([{"a": "1", "b": "2", "c": "3"}], ["c", "a"], "c\ta\n3\t1"),
        ([{"a": "1"}], ["a", "b"], "a\tb\n1\t"),
        ([{"a": None}], None, "a\n"),
        ([], None, ""),
        ([{"a": "one\ttwo"}], None, 'a\n"one\\ttwo"'),
        # A plain leading quote would otherwise be indistinguishable from an escaped cell on
        # decode, so it JSON-encodes too.
        ([{"a": '"quoted'}], None, 'a\n"\\"quoted"'),
    ],
)
def test_an_encoding_is_a_header_line_then_one_tab_separated_line_per_row(
    rows: Sequence[Row], fields: Sequence[str] | None, expected: str
) -> None:
    """Columns are written once, a missing or absent value is an empty cell, order is the ask."""
    assert tabular.encode(rows, fields=fields) == expected


def test_a_header_only_encoding_decodes_to_no_rows() -> None:
    """Naming columns for an empty result set still says what the columns were."""
    assert tabular.decode(tabular.encode([], fields=["a", "b"])) == []


def test_tabular_is_smaller_than_canonical_json_on_a_realistic_jobs_payload() -> None:
    """Correctness-gated, not perf-gated: measures byte length, no timing assertion."""
    payload = _jobs(50)
    canonical = json.dumps(payload, indent=2)
    compact = tabular.encode(payload)

    assert len(compact.encode()) < len(canonical.encode()) * 0.5
    assert tabular.decode(compact) == [
        {key: str(value) for key, value in row.items()} for row in payload
    ]
