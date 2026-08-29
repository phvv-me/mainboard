# THE STORE AT REST, AND THE ONLY PLACE THAT KNOWS ITS LAYOUT.
#
# A dataset is a directory of `run=<id>` partitions, each holding one immutable parquet fragment
# per trial. Hive partitioning is OFF and the run rides as a column, so one fragment read on its
# own still names the run that produced it rather than depending on the directory it was found in.
#
# THE FRAGMENTS OF TWO RUNS NEED NOT SHARE A SCHEMA and one plain scan over all of them refuses
# when they do not. A dataset is written across time, so the day a receipt gains a field every
# older run becomes unreadable beside every newer one, which is how a cross-architecture campaign
# found this: one host held a run from before a provenance field existed and the whole store
# stopped collecting. Reading DIAGONALLY is the same choice compaction already makes one layer
# down, and it answers with the union of the columns and a null where a run predates one.
#
# AND A MISSING COLUMN IS THE SAME FACT AS AN EMPTY ONE, normalised here so no reader has to
# know that. A lane naming no model writes `model = ""`, while a run taken before that field
# existed carries a null for it, and both mean the same thing. Leaving the two apart splits one
# cell of the current view in two, which is how a re-run of a model-less universe came back with
# `matched 2 current trials, want 1`. The rule is stated once, here, rather than in each reader,
# which is where it was already stated once correctly and once not at all. What that rule must NOT
# do is flatten a broken probe into a host with no device, which is why every axis carries its
# probe outcome in a column of its own and both are normalised and both filter.
#
# COVERAGE AXES ARE CONFIGURATION AND NOTHING IS SPECIAL-CASED. The reference this generalizes
# hand-built two cases, per card and per model, and asked the card question only when the host had
# a card, which silently let a machine with no device read another machine's rows as its own.
# Here a consumer declares the axes it scopes coverage by, every declared axis is normalised and
# every declared axis filters, including at the empty coordinate. A consumer that does not want
# per-card coverage does not declare `card`, which is the whole of the knob.
#
# EVIDENCE IS THE ADMISSIBLE SUBSET AND THE STORE IS EVERYTHING. `passing` and `status` are the
# two questions a CLAIM is answered from, so both read only rows whose producing tree can be
# identified. `scan`, `rows` and `as_jsonl` are the store itself and read every row there is,
# because a person opening a ledger wants what was written and not what counted. A row from before
# admissibility was recorded reads `unrecorded`, which is neither admissible nor a lie about it.
#
# AND RECENCY IS A COORDINATE A RUN WRITES DOWN, NEVER A NAME IT IS SORTED BY. Run names open with
# a second-resolution timestamp and end in random hex, so ordering on the name orders two runs
# inside one second by their random tail, which is how `newest` and the one-row-per-cell view both
# came to pick a winner nothing had decided. A run now persists `opened_at_ns` and every recency
# question reads that. Two runs that answer with the same instant are not orderable at all, and
# this REFUSES rather than picking one, because an arbitrary winner is exactly the defect.

import json
from datetime import UTC, datetime
from itertools import chain
from typing import TYPE_CHECKING

import polars as pl

from .coverage import PROBED, Cell, LaneStatus
from .ledger import NESTED, TrialReceipts, wire
from .provenance import Admissibility
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

# The two columns every receipt carries whatever else it holds, and the pair a coverage read and
# a current view both group on.
_LANE, _KEY = "lane", "key"

# The creation coordinate a session persists and every recency question is answered from, and the
# typed eligibility field that decides whether a row is evidence or scratch work.
OPENED, ADMISSIBILITY = "opened_at_ns", "admissibility"

# How a run name spells the second it opened at, which is all a run written before the creation
# coordinate existed ever knew about its own age. Sixteen characters wide, `20260829T094024Z`.
STAMP, DATED = "%Y%m%dT%H%M%SZ", 16

# The column a recency-ordered read carries while it is being taken, dropped before it is handed
# back. A rank rather than the coordinate itself, so one comparison covers a run that recorded a
# coordinate and a run that only ever had a name.
_ORDER = "_recency"

# Where a retired generation lands beside a store, and the one human-readable file inside a store.
# The ledger is named here rather than in each consumer because `retire` has to move it and
# `as_jsonl` has to rewrite it, and a retirement that spelled it differently from a mint is how a
# live store came to hold a ledger describing runs it no longer counts.
RETIRED, GENERATION, LEDGER = "retired", "generation", "latest.jsonl"


class Ambiguous(RuntimeError):
    """Two runs claim the same creation instant, so `newest` is not a question with one answer."""

    def __init__(self, root: Path, tied: Sequence[str]) -> None:
        """root: the store holding them. tied: the runs nothing can order against each other."""
        self.tied = tuple(tied)
        super().__init__(
            f"{root} cannot say which of {', '.join(self.tied)} is newer: they opened at the same "
            "instant, so any answer here would be arbitrary. Name the run explicitly, or retire "
            "the generation that should not be in the current view."
        )


def opened_at(run: str, recorded: int | None) -> tuple[int, int, str]:
    """When one run opened, and how much that answer is worth, as something sortable.

    A run that recorded its own creation coordinate answers in NANOSECONDS and is ordered by it. A
    run written before that coordinate existed answers with the second its NAME encodes, which is
    exactly as much as such a run ever knew, and is why two of them inside one second are not
    orderable at all. A run whose name encodes no instant is UNDATED: name order is the only
    statement such a store makes, so it is ordered by name, sorts before anything dated, and can
    never tie.

    run: the run's identity. recorded: its persisted coordinate, None where it kept none.
    """
    if recorded is not None:
        return (1, recorded, "")
    try:
        named = datetime.strptime(run[:DATED], STAMP).replace(tzinfo=UTC)
    except ValueError:
        return (0, 0, run)
    return (1, int(named.timestamp()) * 1_000_000_000, "")


class Dataset:
    """One store of trial receipts, read across every run it has ever held.

    root: the directory holding the `run=<id>` partitions.
    axes: the coordinates coverage is scoped by, each a receipt column.
    nested: the columns stored as JSON text, `NESTED` unless a consumer stores other shapes.
    node: which node of a universe this is the store of, carried onto every answer so a status
        line and a partition both name their claim without a caller rejoining them.
    samples: how many passing receipts one cell owes before it is complete. One is right for a
        claim that is either true or false on this machine and wrong for a program measuring
        variance, where a cell owes several readings and a re-run must add to them.
    """

    def __init__(
        self,
        root: Path,
        *,
        axes: Sequence[str] = (),
        nested: Sequence[str] = NESTED,
        node: str = "",
        samples: int = 1,
    ) -> None:
        self.root = root
        self.axes = tuple(axes)
        self.nested = tuple(nested)
        self.node = node
        self.samples = samples

    @property
    def admissible(self) -> tuple[pl.Expr, ...]:
        """What a row must be for a claim to lean on it: it passed, and its tree is identifiable.

        The two halves are one filter because they fail the same way. A row whose lane broke and a
        row whose tree nobody can name both look like a reading and are both worth nothing to a
        claim, and a query that remembered only the first is the query every review of this
        program had to correct by hand.
        """
        return (
            pl.col("outcome") == Outcome.PASSED,
            pl.col(ADMISSIBILITY) == Admissibility.ADMISSIBLE,
        )

    @property
    def coordinates(self) -> tuple[str, ...]:
        """Every column a coverage question pins, each declared axis beside its probe outcome."""
        return tuple(chain.from_iterable((axis, f"{axis}{PROBED}") for axis in self.axes))

    @property
    def newest(self) -> str:
        """The most recent run, empty for a store that has never been written to.

        Refuses rather than choosing where the two most recent runs opened at the same instant,
        because a store that cannot say which of two runs came last must not answer as if it can.
        """
        found = self.runs
        return found[-1] if found else ""

    @property
    def opened(self) -> dict[str, tuple[int, int, str]]:
        """Every run beside the coordinate that orders it, refusing where two of them tie.

        Asked over the WHOLE store rather than over one question's rows, so `newest`, the current
        view and a coverage read all order runs the same way and a store is either orderable or
        it is not.
        """
        frame = self.scan()
        if not frame.collect_schema().names():
            return {}
        held = frame.group_by("run").agg(pl.col(OPENED).max()).collect()
        found = {
            str(run): opened_at(str(run), recorded)
            for run, recorded in zip(held["run"], held[OPENED], strict=True)
        }
        shared: dict[tuple[int, int, str], list[str]] = {}
        for run, coordinate in found.items():
            shared.setdefault(coordinate, []).append(run)
        tied = [runs for runs in shared.values() if len(runs) > 1]
        if tied:
            raise Ambiguous(self.root, sorted(tied[0]))
        return found

    @property
    def parts(self) -> list[Path]:
        """Every committed fragment of every run, in run then write order."""
        return sorted(self.root.glob("run=*/part-*.parquet"))

    @property
    def runs(self) -> tuple[str, ...]:
        """Every run this store holds, oldest first, ordered by when each said it opened."""
        found = self.opened
        return tuple(sorted(found, key=lambda run: found[run]))

    @property
    def stored(self) -> frozenset[str]:
        """Every run this store holds, as a set, which is membership and never an order.

        Separate from `runs` because a retirement asks whether a run is HERE, and a store whose
        runs cannot be ordered is exactly the store a retirement is being run on.
        """
        frame = self.scan()
        if not frame.collect_schema().names():
            return frozenset()
        return frozenset(str(run) for run in frame.select("run").unique().collect()["run"])

    def ranked(self, frame: pl.DataFrame, order: Sequence[str]) -> pl.DataFrame:
        """`frame` carrying each row's index in `order`, so a sort reads time and not a name.

        The order is passed in rather than read here, because a caller that also has to NAME the
        run it selected would otherwise ask the store for the same ordering twice.
        """
        ranks = {run: index for index, run in enumerate(order)}
        return frame.with_columns(
            pl.col("run").replace_strict(ranks, return_dtype=pl.Int64).alias(_ORDER)
        )

    @classmethod
    def holding(
        cls, path: Path, *, axes: Sequence[str] = (), nested: Sequence[str] = NESTED
    ) -> Dataset | None:
        """The dataset `path` names, or None when it names none.

        Both spellings answer, the partition root itself and an evidence directory with the
        partitions one level down, since a person pointing a verb at their evidence should not
        have to remember which of the two the store happens to sit in.

        path: a directory that may hold `run=<id>` partitions, directly or under `receipts/`.
        axes: the coordinates the store is read with. nested: its JSON text columns.
        """
        for candidate in (path, path / "receipts"):
            if next(candidate.glob("run=*/part-*.parquet"), None) is not None:
                return cls(candidate, axes=axes, nested=nested)
        return None

    def as_jsonl(self, target: Path, run: str = "") -> int:
        """Stream one run out as the `trial_receipt` JSON lines a dispatch boundary reads.

        target: the file to write. run: which run to frame, the newest when empty. Returns the
        row count, zero for a store that has never been written to.
        """
        lines = [wire(row) for row in self.rows(run)]
        target.write_text("".join(lines), encoding="utf-8")
        return len(lines)

    def retire(self, generation: str, runs: Sequence[str]) -> Path:
        """Move `runs` out of the current view into a named generation, the ledger with them.

        generation: what the retired runs are called, becoming `retired/generation=<name>/`
            beside this store. runs: which run identities leave the current view.

        Returns the directory the generation landed in. A run that is not in this store is named
        rather than skipped, since retiring a run that was never here is a typo and not a no-op.

        THE LEDGER MOVES AND IS THEN REWRITTEN, WHICH IS THE WHOLE REASON THIS IS A METHOD.
        `latest.jsonl` is the one file in a receipts directory a person can read, and a retirement
        that moved the parquet fragments and left it behind hands that person the exact generation
        the store no longer counts. Three universes of the reproducibility workspace were caught
        with it on 2026-08-29 in one commit (`recovery_cost` 7a, `contiguous_reduction` 4c,
        `corrected_law_transfer` 5b) and a fourth on the same day (`accuracy_selection` 6d), all
        four by hand-rolled retirements that moved directories. So the ledger travels into the
        generation that owns it and is reminted from whatever run is newest afterwards, or removed
        when the retirement emptied the store.
        """
        held = self.stored
        missing = [run for run in runs if run not in held]
        if missing:
            raise ValueError(
                f"{self.root} holds no run {', '.join(missing)}, so there is nothing to retire "
                f"under {generation!r}; it holds {', '.join(sorted(held)) or 'no runs at all'}"
            )
        target = self.root.parent / RETIRED / f"{GENERATION}={generation}"
        target.mkdir(parents=True, exist_ok=True)
        ledger = self.root / LEDGER
        if ledger.exists():
            ledger.replace(target / LEDGER)
        for run in runs:
            (self.root / f"run={run}").replace(target / f"run={run}")
        if not self.stored:
            return target
        self.as_jsonl(ledger)
        return target

    def decoded(self, row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """One stored row with its JSON text columns read back as the objects they hold."""
        return {
            key: json.loads(value) if key in self.nested and isinstance(value, str) else value
            for key, value in row.items()
        }

    def passing(self, *, every: bool = False) -> pl.DataFrame:
        """The store's admissible passing rows, one per cell by default and all of them when asked.

        The default is the coverage rule spent as a SELECTION rather than as a question, which is
        what a table renders from: one row per cell, taken from the run that most recently
        produced it, so a re-run supersedes without anyone choosing a run by hand. A program whose
        cells owe several samples asks for all of them instead, since averaging over the newest
        reading of each cell is averaging over one number.

        MOST RECENTLY IS READ OFF THE CREATION COORDINATE AND NOT OFF THE RUN NAME, and a row a
        moving tree produced is not in here at all, because both of those decide which reading a
        figure prints and neither is a question a run name can answer.

        every: keep every passing receipt rather than the newest of each cell.
        """
        frame = self.scan()
        if not frame.collect_schema().names():
            return pl.DataFrame()
        grouped = [_LANE, _KEY, *self.coordinates]
        passed = self.ranked(frame.filter(*self.admissible).collect(), self.runs)
        if every:
            return passed.sort([*grouped, _ORDER]).drop(_ORDER)
        return (
            passed.sort(_ORDER).group_by(grouped, maintain_order=True).last().sort(grouped)
        ).drop(_ORDER)

    def rows(self, run: str = "") -> list[dict[str, JsonValue]]:
        """One run's receipts as plain records, their JSON columns decoded.

        run: which run to read, the newest the store holds when empty.
        """
        frame = self.scan()
        if not frame.collect_schema().names():
            return []
        chosen = run or self.newest
        return [
            self.decoded(row) for row in frame.filter(pl.col("run") == chosen).collect().to_dicts()
        ]

    def scan(self) -> pl.LazyFrame:
        """Every receipt this store has ever held, across every run, or an empty frame.

        The eligibility field and the creation coordinate are normalised beside the axes, for the
        same reason and with one difference. An axis a run predates reads empty, which is the same
        fact as a lane naming no subject. Eligibility a run predates reads `unrecorded`, which is
        NOT the same fact as admissible and must never be filled in as one. And a coordinate a run
        predates stays NULL rather than becoming zero, because a run that never said when it
        opened has not claimed to be the oldest.
        """
        parts = self.parts
        if not parts:
            return pl.LazyFrame()
        frame = pl.concat(
            [pl.scan_parquet(part, hive_partitioning=False) for part in parts],
            how="diagonal_relaxed",
        )
        held = frame.collect_schema().names()
        return frame.with_columns(
            *(
                pl.col(column).cast(pl.String).fill_null("")
                if column in held
                else pl.lit("").alias(column)
                for column in self.coordinates
            ),
            pl.col(ADMISSIBILITY).cast(pl.String).fill_null(str(Admissibility.UNRECORDED))
            if ADMISSIBILITY in held
            else pl.lit(str(Admissibility.UNRECORDED)).alias(ADMISSIBILITY),
            pl.col(OPENED).cast(pl.Int64)
            if OPENED in held
            else pl.lit(None, dtype=pl.Int64).alias(OPENED),
        )

    def status(self, lane: str, expected: Collection[str], cell: Cell) -> LaneStatus:
        """One lane's completeness at one cell, compared to the grid and the samples it owes.

        expected: the keys the lane's own grid would run, which the lane declares by being
            collected rather than by retyping them anywhere.
        cell: the coordinate this question is asked at. A reading is a fact about the silicon and
            the subject that produced it, so a receipt taken elsewhere does not answer for here
            and every declared axis filters, its probe outcome included.

        A row from a tree nobody can identify never counts toward a lane, so a session run on a
        dirty tree measures, prints and writes, and the next clean session still finds the lane
        owed. That is what makes scratch work free: it costs the claim nothing either way.
        """
        frame = self.scan()
        names = self.runs
        taken: dict[str, tuple[int, int]] = {}
        if frame.collect_schema().names():
            wanted = [pl.col(_LANE) == lane, *self.admissible]
            wanted += [pl.col(column) == value for column, value in cell.filters.items()]
            found = (
                self.ranked(frame.filter(*wanted).collect(), names)
                .group_by(_KEY)
                .agg(pl.len().alias("taken"), pl.col(_ORDER).max())
            )
            taken = {
                str(key): (int(count), int(order))
                for key, count, order in zip(
                    found[_KEY], found["taken"], found[_ORDER], strict=True
                )
            }
        counts = {key: taken.get(key, (0, -1))[0] for key in expected}
        latest = max((taken[key][1] for key in counts if key in taken), default=-1)
        return LaneStatus(
            lane=lane,
            want=len(counts) * self.samples,
            have=sum(min(count, self.samples) for count in counts.values()),
            missing=tuple(sorted(key for key, count in counts.items() if count < self.samples)),
            run=names[latest] if latest >= 0 else "",
            cell=cell,
            node=self.node,
        )

    def writer(self, run: str, common: Mapping[str, JsonValue]) -> TrialReceipts:
        """This run's own writer, its partition named after it and the run stamped on every row.

        run: the run's identity, which NAMES ITS OWN DIRECTORY of fragments.
        common: the fields every receipt of this run carries beyond the run itself.
        """
        return TrialReceipts(self.root / f"run={run}", {"run": run, **common}, nested=self.nested)
