# THE THREE LINTS A STORE OWES ITS OWN CLAIMS, EACH ONE EARNED BY A RULING RATHER THAN INVENTED.
#
# A receipt says what a lane measured. It does not say whether the lane COULD have measured
# anything else, and that second question is the one every hostile review of this program has kept
# answering by hand. Three shapes came back often enough to be machinery:
#
# AN IDENTITY IS NOT A RESULT. A quotient, product or decomposition whose terms all come off one
# population reproduces because the arithmetic cancels, and the receipt of it is a constant to the
# last bit. `store_crossing`'s referee ruled a three-factor residue FATAL for it on 2026-08-28,
# `crossing_cascade`'s ruled the successor's repeat FATAL again on 2026-08-29, and the workspace
# rule that followed says a lane like that demonstrates its own failure path or settles `known`.
# The detectable shape is the one that rule names: a payload key within one ulp of an exact
# constant on every row of every run.
#
# AN UNFAILABLE GATE IS AN ASSERTION NO RECEIPT CAN FAIL. The sharpest instance is a registered
# interval that is the observed range of the very rows it scores: `accuracy_selection`'s
# `[0.99955, 1.00349]` is the min and max of its thirty scored rows, two of which sit on the edges
# and define them, so the lane's own `Refutes if:` cannot fire on any input. That is detectable
# without knowing one consumer word: a band held constant across a lane whose endpoints ARE the
# extremes of a quantity that lane measured.
#
# A REGISTERED KILL OWES COVERAGE. A refutation clause naming a case the collected grid never
# contains is a claim that cannot die where it is false. `carried_block_width`'s W1 states its own
# kill on the carried width, its lane runs five shapes that are all `M > 1`, and the four `M = 1`
# shapes where the sibling lane recorded the pre-registration dying are the four the kill lane does
# not visit. Detectable across one node: a lane whose kill has never fired anywhere in the store,
# beside a sibling lane of the same node that settled a refutation at keys the first lane's own
# grid does not contain.
#
# THESE READ RECEIPTS AND NEVER SOURCE. A lint that parsed a lane's assertions would be a second,
# worse type checker and would go stale the moment a helper moved. What a store holds is what was
# actually measured across every run a claim has ever taken, which is a stronger witness than the
# text of a condition: a gate that never discriminated on any row of any generation did not
# discriminate, whatever it says.
#
# AND NOTHING HERE FAILS A SESSION. The exit code is about the apparatus, so a finding is printed
# and the run still exits zero. A lint is a reading a person acts on, not a gate a run trips over.

import math
from typing import TYPE_CHECKING

from patos import FrozenModel

from .vocabulary import Outcome, Stance

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from pydantic import JsonValue

    from .dataset import Dataset
    from .vocabulary import Vocabulary

# How many rows a lane needs before a constant among them means anything. One row is a constant by
# arithmetic and says nothing about whether a second could have differed.
ENOUGH = 2

# The constants an identity lands on. A residue is `0.0` and a reproduced ratio is `1.0`, and
# those are what a cancelling product prints. EVERY OTHER CONSTANT IS SOMEBODY'S DESIGN: a
# registered band is constant on purpose, so is a declared draw count, a precision and a
# tolerance, and a lint reporting those would fire on almost every honest lane and teach a reader
# to skip the section.
RESIDUES = (0.0, 1.0)


class Finding(FrozenModel):
    """One lint's complaint about one lane, in the shape a terminal line and a test both read.

    lint: which of the three fired. node: the claim. lane: the pytest node id up to the bracket.
    detail: what was found, naming the payload key and the value so a reader can go look.
    """

    lint: str
    node: str
    lane: str
    detail: str

    def line(self) -> str:
        """One terminal line, the lane first because that is what a reader goes and opens."""
        return f"  {self.lane} [{self.lint}] {self.detail}"


def numbers(rows: Sequence[Mapping[str, JsonValue]], key: str) -> list[float]:
    """Every finite float one payload key holds across `rows`, skipping the rows that lack it."""
    found = []
    for row in rows:
        value = row.get("measured", {})
        held = value.get(key) if isinstance(value, dict) else None
        if isinstance(held, bool) or not isinstance(held, int | float):
            continue
        if math.isfinite(held):
            found.append(float(held))
    return found


def pinned(values: Sequence[float]) -> float | None:
    """The one constant every value sits within an ulp of, or None where they differ.

    An ulp rather than equality because a quantity that cancels algebraically still arrives
    through floating point, so the residue of a telescoping product is `1.0` give or take the last
    bit rather than exactly `1.0`, and a lint that demanded exactness would miss every real one.
    """
    if len(values) < ENOUGH:
        return None
    first = values[0]
    slack = math.ulp(first) if first else math.ulp(1.0)
    return first if all(abs(value - first) <= slack for value in values) else None


def keys_of(rows: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    """Every payload key any of `rows` carries, in first-seen order so a report is stable."""
    seen: dict[str, None] = {}
    for row in rows:
        measured = row.get("measured", {})
        if isinstance(measured, dict):
            seen.update(dict.fromkeys(measured))
    return list(seen)


def lanes_of(rows: Sequence[Mapping[str, JsonValue]]) -> dict[str, list[Mapping[str, JsonValue]]]:
    """`rows` grouped by the lane that settled them, in first-seen order."""
    grouped: dict[str, list[Mapping[str, JsonValue]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("lane", "")), []).append(row)
    return grouped


def identities(node: str, lane: str, rows: Sequence[Mapping[str, JsonValue]]) -> Iterator[Finding]:
    """Payload keys pinned to one exact constant on every row, which is the identity shape.

    A key that never moved across a whole lane is a quantity the lane did not measure, and at
    `0.0` or `1.0` it is a residue or a reproduced ratio whose terms cancel. Any other constant is
    a registration, a declared budget or a tolerance, which are constant because somebody decided
    they should be.
    """
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
    """Bands whose endpoints are the extremes of a quantity the same lane measured.

    An interval read off the rows it scores is a report and not a test, and its refutation
    condition cannot fire on any input the lane can produce.
    """
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
    """Lanes whose kill never fired, beside a sibling that died at keys they never run.

    The refutation lives at a coordinate the lane carrying the kill does not visit, which is the
    shape that lets a claim survive precisely where it is false.
    """
    grouped = lanes_of(rows)
    words = {lane: {str(row.get("verdict", "")) for row in held} for lane, held in grouped.items()}
    grids = {lane: {str(row.get("key", "")) for row in held} for lane, held in grouped.items()}
    died = {lane: grid for lane, grid in grids.items() if words[lane] & refuting}
    for lane, grid in grids.items():
        # A LANE THAT DECLARES NO GRID CANNOT HAVE AIMED ONE AWAY FROM ANYTHING. One key, or the
        # empty key a lane without a parametrize carries, means the lane is one cell and its
        # coverage question is whether that cell exists rather than which cells it left out.
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

    store: the receipts of one claim. vocabulary: the words this workspace settles on, read for
    which of them refute, since a workspace names its own and this tool knows none of them.
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
