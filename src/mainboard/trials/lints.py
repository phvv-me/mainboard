# Three lints a store owes its own claims, each earned by a referee's ruling.
#
# A receipt says what a lane measured, not whether it COULD have measured anything else:
#
# - An identity is not a result. Terms drawn from one population cancel, and the receipt is a
#   constant to the last bit (`store_crossing` FATAL 2026-08-28, `crossing_cascade` FATAL
#   2026-08-29; such a lane demonstrates its own failure path or settles `known`). Detected as a
#   payload key within one ulp of 0.0 or 1.0 on every row of every run.
# - An unfailable gate: a registered band that is the observed range of the rows it scores
#   (`accuracy_selection`'s `[0.99955, 1.00349]` is the min and max of its thirty rows). Detected
#   as a band constant across a lane whose endpoints are the extremes of a quantity it measured.
# - A registered kill owes coverage: `carried_block_width` W1's kill lane runs only `M > 1` shapes
#   while its sibling saw the pre-registration die at the four `M = 1` shapes. Detected as a lane
#   whose kill never fired beside a sibling that refuted at keys outside the first lane's grid.
#
# These read receipts, never source: what a store holds across every run is a stronger witness
# than the text of a condition. They read every passing row, admissible or not, since the question
# is whether a gate EVER discriminated and a dirty tree must not hide an unfailable band. A finding
# is printed and never fails a session; the exit code is about the apparatus.

import math
from typing import TYPE_CHECKING

from patos import FrozenModel

from .vocabulary import Outcome, Stance

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from pydantic import JsonValue

    from .dataset import Dataset
    from .vocabulary import Vocabulary

# How many rows a lane needs before a constant among them means anything.
ENOUGH = 2

# The constants an identity lands on: a residue and a reproduced ratio. Every other constant is
# somebody's design (a band, a draw count, a tolerance), and reporting those would fire on almost
# every honest lane.
RESIDUES = (0.0, 1.0)


class Finding(FrozenModel):
    """One lint's complaint about one lane.

    lane: the pytest node id up to the bracket.
    detail: what was found, naming the payload key and the value.
    """

    lint: str
    node: str
    lane: str
    detail: str

    def line(self) -> str:
        """One terminal line, the lane first because that is what a reader opens."""
        return f"  {self.lane} [{self.lint}] {self.detail}"


def _payload(row: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    measured = row.get("measured")
    return measured if isinstance(measured, dict) else {}


def numbers(rows: Sequence[Mapping[str, JsonValue]], key: str) -> list[float]:
    """Every finite float one payload key holds across `rows`, skipping the rows that lack it."""
    held = [_payload(row).get(key) for row in rows]
    return [
        float(value)
        for value in held
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    ]


def pinned(values: Sequence[float]) -> float | None:
    """The one constant every value sits within an ulp of, or None where they differ.

    An ulp rather than equality, since an algebraic cancellation still arrives rounded.
    """
    if len(values) < ENOUGH:
        return None
    first = values[0]
    slack = math.ulp(first) if first else math.ulp(1.0)
    return first if all(abs(value - first) <= slack for value in values) else None


def keys_of(rows: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    """Every payload key any of `rows` carries, in first-seen order so a report is stable."""
    return list(dict.fromkeys(key for row in rows for key in _payload(row)))


def lanes_of(rows: Sequence[Mapping[str, JsonValue]]) -> dict[str, list[Mapping[str, JsonValue]]]:
    """`rows` grouped by the lane that settled them, in first-seen order."""
    grouped: dict[str, list[Mapping[str, JsonValue]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("lane", "")), []).append(row)
    return grouped


def identities(node: str, lane: str, rows: Sequence[Mapping[str, JsonValue]]) -> Iterator[Finding]:
    """Payload keys pinned to a residue constant on every row, the identity shape."""
    for key in keys_of(rows):
        constant = pinned(numbers(rows, key))
        if constant is None or constant not in RESIDUES:
            continue
        yield Finding(
            lint="identity",
            node=node,
            lane=lane,
            detail=(
                f"`{key}` is {constant!r} on all {len(rows)} rows of every run, within one ulp, "
                "so nothing this lane ran moved it; an identity-shaped claim demonstrates its "
                "own failure path or settles `known`"
            ),
        )


def unfailable(node: str, lane: str, rows: Sequence[Mapping[str, JsonValue]]) -> Iterator[Finding]:
    """Bands whose endpoints are the extremes of a quantity the same lane measured."""
    scored = {key: numbers(rows, key) for key in keys_of(rows)}
    bands = {
        key: constant
        for key, values in scored.items()
        if len(values) == len(rows) and (constant := pinned(values)) is not None
    }
    for key, values in scored.items():
        if key in bands or len(values) < ENOUGH:
            continue
        low, high = min(values), max(values)
        edges = [name for name, held in bands.items() if held in (low, high)]
        if len(edges) < ENOUGH:
            continue
        yield Finding(
            lint="unfailable",
            node=node,
            lane=lane,
            detail=(
                f"{' and '.join(sorted(edges))} hold [{low!r}, {high!r}], which is the observed "
                f"range of `{key}` over the {len(values)} rows this lane scores, so the interval "
                "is a report of its own outcomes and no row can leave it"
            ),
        )


def uncovered(
    node: str, rows: Sequence[Mapping[str, JsonValue]], refuting: frozenset[str]
) -> Iterator[Finding]:
    """Lanes whose kill never fired, beside a sibling that died at keys they never run."""
    grouped = lanes_of(rows)
    words = {lane: {str(row.get("verdict", "")) for row in held} for lane, held in grouped.items()}
    # An absent parametrization key does not identify a missing grid coordinate.
    grids = {
        lane: {key for row in held if isinstance(key := row.get("key"), str) and key}
        for lane, held in grouped.items()
    }
    died = {lane: grid for lane, grid in grids.items() if words[lane] & refuting}
    for lane, grid in grids.items():
        # A lane of one cell cannot have aimed its grid away from anything.
        if words[lane] & refuting or len(grid) < ENOUGH:
            continue
        missed = sorted(
            {key for other, keys in died.items() if other != lane for key in keys} - grid
        )
        if not missed:
            continue
        yield Finding(
            lint="registered-kill",
            node=node,
            lane=lane,
            detail=(
                f"no run of this lane ever settled a refutation, and this claim's own refutations "
                f"sit at {len(missed)} key(s) its grid never contains: {', '.join(missed[:4])}"
                f"{' ...' if len(missed) > 4 else ''}"
            ),
        )


def findings(store: Dataset, vocabulary: Vocabulary) -> tuple[Finding, ...]:
    """Every lint one claim's whole store answers, across every run it has ever held.

    vocabulary: read for which words refute, since a workspace names its own.
    """
    rows = [
        row
        for row in (store.decoded(record) for record in store.scan().collect().to_dicts())
        if str(row.get("outcome", "")) == Outcome.PASSED
    ]
    if not rows:
        return ()
    node = store.node or str(rows[0].get("node", ""))
    refuting = frozenset(vocabulary.stanced(Stance.REFUTES))
    found = list(uncovered(node, rows, refuting))
    for lane, held in lanes_of(rows).items():
        if len(held) < ENOUGH:
            continue
        found.extend(identities(node, lane, held))
        found.extend(unfailable(node, lane, held))
    return tuple(found)
