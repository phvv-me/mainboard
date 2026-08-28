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

import json
from itertools import chain
from typing import TYPE_CHECKING

import polars as pl

from .coverage import PROBED, Cell, LaneStatus
from .ledger import NESTED, TrialReceipts, wire
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

# The two columns every receipt carries whatever else it holds, and the pair a coverage read and
# a current view both group on.
_LANE, _KEY = "lane", "key"


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
    def coordinates(self) -> tuple[str, ...]:
        """Every column a coverage question pins, each declared axis beside its probe outcome."""
        return tuple(chain.from_iterable((axis, f"{axis}{PROBED}") for axis in self.axes))

    @property
    def newest(self) -> str:
        """The most recent run, empty for a store that has never been written to.

        Run identities open with a UTC timestamp, so newest is last in name order and no row has
        to be read to find it.
        """
        found = self.runs
        return found[-1] if found else ""

    @property
    def parts(self) -> list[Path]:
        """Every committed fragment of every run, in run then write order."""
        return sorted(self.root.glob("run=*/part-*.parquet"))

    @property
    def runs(self) -> tuple[str, ...]:
        """Every run this store holds, oldest identity first."""
        frame = self.scan()
        if not frame.collect_schema().names():
            return ()
        return tuple(sorted(set(frame.select("run").collect()["run"])))

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

    def decoded(self, row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """One stored row with its JSON text columns read back as the objects they hold."""
        return {
            key: json.loads(value) if key in self.nested and isinstance(value, str) else value
            for key, value in row.items()
        }

    def passing(self, *, every: bool = False) -> pl.DataFrame:
        """The store's passing rows, one per cell by default and all of them when asked.

        The default is the coverage rule spent as a SELECTION rather than as a question, which is
        what a table renders from: one row per cell, taken from the run that most recently
        produced it, so a re-run supersedes without anyone choosing a run by hand. A program whose
        cells owe several samples asks for all of them instead, since averaging over the newest
        reading of each cell is averaging over one number.

        every: keep every passing receipt rather than the newest of each cell.
        """
        frame = self.scan()
        if not frame.collect_schema().names():
            return pl.DataFrame()
        grouped = [_LANE, _KEY, *self.coordinates]
        passed = frame.filter(pl.col("outcome") == Outcome.PASSED).collect().sort("run")
        if every:
            return passed.sort([*grouped, "run"])
        return passed.group_by(grouped, maintain_order=True).last().sort(grouped)

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
        """Every receipt this store has ever held, across every run, or an empty frame."""
        parts = self.parts
        if not parts:
            return pl.LazyFrame()
        frame = pl.concat(
            [pl.scan_parquet(part, hive_partitioning=False) for part in parts],
            how="diagonal_relaxed",
        )
        held = frame.collect_schema().names()
        return frame.with_columns(
            pl.col(column).cast(pl.String).fill_null("")
            if column in held
            else pl.lit("").alias(column)
            for column in self.coordinates
        )

    def status(self, lane: str, expected: Collection[str], cell: Cell) -> LaneStatus:
        """One lane's completeness at one cell, compared to the grid and the samples it owes.

        expected: the keys the lane's own grid would run, which the lane declares by being
            collected rather than by retyping them anywhere.
        cell: the coordinate this question is asked at. A reading is a fact about the silicon and
            the subject that produced it, so a receipt taken elsewhere does not answer for here
            and every declared axis filters, its probe outcome included.
        """
        frame = self.scan()
        taken: dict[str, tuple[int, str]] = {}
        if frame.collect_schema().names():
            wanted = [pl.col(_LANE) == lane, pl.col("outcome") == Outcome.PASSED]
            wanted += [pl.col(column) == value for column, value in cell.filters.items()]
            found = (
                frame.filter(*wanted)
                .group_by(_KEY)
                .agg(pl.len().alias("taken"), pl.col("run").max())
                .collect()
            )
            taken = {
                key: (count, run)
                for key, count, run in zip(found[_KEY], found["taken"], found["run"], strict=True)
            }
        counts = {key: taken.get(key, (0, ""))[0] for key in expected}
        return LaneStatus(
            lane=lane,
            want=len(counts) * self.samples,
            have=sum(min(count, self.samples) for count in counts.values()),
            missing=tuple(sorted(key for key, count in counts.items() if count < self.samples)),
            run=max((taken[key][1] for key in counts if key in taken), default=""),
            cell=cell,
            node=self.node,
        )

    def writer(self, run: str, common: Mapping[str, JsonValue]) -> TrialReceipts:
        """This run's own writer, its partition named after it and the run stamped on every row.

        run: the run's identity, which NAMES ITS OWN DIRECTORY of fragments.
        common: the fields every receipt of this run carries beyond the run itself.
        """
        return TrialReceipts(self.root / f"run={run}", {"run": run, **common}, nested=self.nested)
