# The store at rest, and the only place that knows its layout.
#
# A dataset is a directory of `run=<id>` partitions, each holding one immutable parquet fragment
# per trial. Hive partitioning is off and the run rides as a column, so a fragment read alone still
# names its run. Fragments of two runs need not share a schema: a host holding a run from before a
# provenance field existed once stopped a whole store collecting, so reads are DIAGONAL, the union
# of the columns with nulls where a run predates one.
#
# A missing axis column is the same fact as an empty one (a re-run of a model-less universe once
# came back `matched 2 current trials, want 1`), normalised here once rather than in each reader.
# A broken probe must not flatten into a host with no device, so every axis carries its probe
# outcome in its own column and both filter. Axes are configuration: the reference hand-built per
# card and per model and asked the card question only on a host with a card, letting a cardless
# machine read another's rows. Here every declared axis filters, including at the empty coordinate.
#
# Evidence is the admissible subset; the store is everything. `passing` and `status` answer claims
# and read only rows whose producing tree is identified; `scan`, `rows` and `as_jsonl` read every
# row, because a person opening a ledger wants what was written. Recency is the coordinate a run
# writes down (`opened_at_ns`), never its name, whose random hex tail used to pick `newest` inside
# one second; two runs claiming the same instant are refused rather than picked between.

import json
from datetime import UTC, datetime
from io import BytesIO
from itertools import chain
from typing import TYPE_CHECKING

import polars as pl

from .artifacts import Artifact
from .coverage import PROBED, Cell, LaneStatus
from .ledger import NESTED, TrialReceipts, wire
from .provenance import Admissibility
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

# The two columns every receipt carries, which a coverage read and a current view both group on.
_LANE, _KEY = "lane", "key"

# The creation coordinate every recency question reads, and the field deciding evidence or scratch.
OPENED, ADMISSIBILITY = "opened_at_ns", "admissibility"

# How a run name spells the second it opened, `20260829T094024Z`, all a run written before the
# creation coordinate knew about its own age.
STAMP, DATED = "%Y%m%dT%H%M%SZ", 16

# A recency rank a read carries while ordering and drops before returning, so one comparison
# covers runs that recorded a coordinate and runs that only have a name.
_ORDER = "_recency"

# Where a retired generation lands beside a store, and the store's one human-readable file, named
# once so `retire` moves exactly the ledger `as_jsonl` mints.
RETIRED, GENERATION, LEDGER = "retired", "generation", "latest.jsonl"

# Where a run that did not cover every lane is still written readably, as it may not be `LEDGER`.
PARTIAL = "partial-{}.jsonl"


class Ambiguous(RuntimeError):
    """Two runs claim the same creation instant, so `newest` is not a question with one answer."""

    def __init__(self, root: Path, tied: Sequence[str]) -> None:
        self.tied = tuple(tied)
        super().__init__(
            f"{root} cannot say which of {', '.join(self.tied)} is newer: they opened at the same "
            "instant, so any answer here would be arbitrary. Name the run explicitly, or retire "
            "the generation that should not be in the current view."
        )


def opened_at(run: str, recorded: int | None) -> tuple[int, int, str]:
    """When one run opened, as something sortable.

    A recorded coordinate answers in nanoseconds. A run from before it answers with the second its
    name encodes, all it ever knew, so two such runs in one second tie. A name encoding no instant
    is UNDATED: ordered by name, before anything dated, and never tying.

    recorded: the run's persisted coordinate, None where it kept none.
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
    nested: the columns stored as JSON text.
    node: the universe node this stores, carried onto every answer.
    samples: how many passing receipts one cell owes before it is complete.
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
        """What a row must be for a claim to lean on it: passed, with an identifiable tree.

        One filter, because a broken lane and an unidentifiable tree fail a claim the same way, and
        a query remembering only the first was the one every review had to correct.
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
        """The most recent run, empty for an unwritten store; refuses on a tie (see `opened`)."""
        found = self.runs
        return found[-1] if found else ""

    @property
    def opened(self) -> dict[str, tuple[int, int, str]]:
        """Every run beside the coordinate that orders it, refusing where two of them tie.

        Asked over the whole store, so every reader orders runs the same way.
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
        """Every run this store holds, oldest first."""
        found = self.opened
        return tuple(sorted(found, key=lambda run: found[run]))

    @property
    def stored(self) -> frozenset[str]:
        """Every run this store holds, as membership only, so it answers even when `runs` ties."""
        frame = self.scan()
        if not frame.collect_schema().names():
            return frozenset()
        return frozenset(str(run) for run in frame.select("run").unique().collect()["run"])

    def ranked(self, frame: pl.DataFrame, order: Sequence[str]) -> pl.DataFrame:
        """`frame` with each row's run index in `order`, so a sort reads time and not a name."""
        ranks = {run: index for index, run in enumerate(order)}
        return frame.with_columns(
            pl.col("run").replace_strict(ranks, return_dtype=pl.Int64).alias(_ORDER)
        )

    @classmethod
    def holding(
        cls, path: Path, *, axes: Sequence[str] = (), nested: Sequence[str] = NESTED
    ) -> Dataset | None:
        """The dataset at `path` itself or at `receipts/` below it, None when neither holds one."""
        for candidate in (path, path / "receipts"):
            if next(candidate.glob("run=*/part-*.parquet"), None) is not None:
                return cls(candidate, axes=axes, nested=nested)
        return None

    def as_jsonl(self, target: Path, run: str = "") -> int:
        """Write one run, the newest when empty, as `trial_receipt` lines; returns the rows."""
        lines = [wire(row) for row in self.rows(run)]
        target.write_text("".join(lines), encoding="utf-8")
        return len(lines)

    def full(self, run: str) -> bool:
        """Whether `run` alone shows every lane this store has ever recorded.

        Only such a run may become `latest.jsonl`: a partial run (one file, a `-k` selection) is
        still evidence, but reminting from it would drop every lane it did not touch from the one
        file a person reads.
        """
        return self.lanes(run) >= self.lanes()

    def lanes(self, run: str = "") -> frozenset[str]:
        """Every lane recorded in `run`, or in the whole store when empty."""
        frame = self.scan()
        if not frame.collect_schema().names():
            return frozenset()
        if run:
            frame = frame.filter(pl.col("run") == run)
        return frozenset(str(lane) for lane in frame.select(_LANE).unique().collect()[_LANE])

    def retire(self, generation: str, runs: Sequence[str]) -> Path:
        """Move `runs` into `retired/generation=<name>/` beside this store, the ledger with them.

        Returns the generation directory. A run not in this store is refused by name, since
        retiring a run that was never here is a typo. The ledger travels into the generation and
        is reminted from whatever is newest afterwards, or removed when the store is emptied:
        hand-rolled retirements that moved only the fragments left `latest.jsonl` describing the
        retired generation in four universes on 2026-08-29 (`recovery_cost` 7a,
        `contiguous_reduction` 4c, `corrected_law_transfer` 5b, `accuracy_selection` 6d).
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
        if self.stored:
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

        The default is what a table renders from: each cell's row from the run that most recently
        produced it, by creation coordinate, so a re-run supersedes without anyone choosing a run.
        A program whose cells owe several samples asks for `every` instead.
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
        """One run's receipts, the newest run when empty, as plain records with JSON decoded."""
        frame = self.scan()
        if not frame.collect_schema().names():
            return []
        chosen = run or self.newest
        return [
            self.decoded(row) for row in frame.filter(pl.col("run") == chosen).collect().to_dicts()
        ]

    def tables(self, root: Path, *, schema_name: str, run: str = "") -> pl.DataFrame:
        """Read verified table artifacts across hardware without selecting a winning run.

        root: the project root that artifact references are relative to.
        schema_name: the scientific table schema, shared by compatible producers.
        run: one explicit run, or every run when empty. `_trial` carries JSON provenance;
            ordinary columns retain the experiment's data and units unchanged.
        """
        receipts = self.scan()
        if not receipts.collect_schema().names():
            return pl.DataFrame()
        if run:
            receipts = receipts.filter(pl.col("run") == run)
        tables: list[pl.DataFrame] = []
        for raw in receipts.collect().to_dicts():
            receipt = self.decoded(raw)
            references = receipt.get("artifacts")
            if not isinstance(references, dict):
                continue
            metadata = {
                key: value
                for key, value in receipt.items()
                if key not in {"measured", "artifacts"}
            }
            for label, value in references.items():
                if not isinstance(value, dict) or value.get("schema_name") != schema_name:
                    continue
                reference = Artifact.model_validate(value)
                if reference.media_type != "application/vnd.apache.parquet":
                    raise ValueError(f"{schema_name} names a non-Parquet artifact")
                table = pl.read_parquet(BytesIO(reference.read(root)))
                if "_trial" in table.columns:
                    raise ValueError("table payload uses the reserved _trial provenance column")
                provenance = {
                    **metadata,
                    "artifact_name": label,
                    "artifact_sha256": reference.sha256,
                }
                tables.append(
                    table.with_columns(
                        pl.lit(json.dumps(provenance, sort_keys=True)).alias("_trial")
                    )
                )
        return pl.concat(tables, how="diagonal_relaxed") if tables else pl.DataFrame()

    def scan(self) -> pl.LazyFrame:
        """Every receipt this store has ever held, across every run, or an empty frame.

        An axis a run predates reads empty, the same fact as a lane naming no subject.
        Admissibility a run predates reads `unrecorded`, never admissible. A creation coordinate a
        run predates stays null, since a run that never said when it opened has not claimed to be
        the oldest.
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
        """One lane's completeness at one cell, against its grid and the samples each key owes.

        expected: the keys the lane's own grid would run, declared by being collected.
        cell: the coordinate asked at; every declared axis filters, its probe outcome included,
            since a receipt taken on other silicon or subject does not answer for here.

        Rows from an unidentified tree never count, so a dirty session measures and writes and the
        next clean one still finds the lane owed: scratch work costs the claim nothing.
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
        """This run's writer, into the partition named after it, with the run on every row."""
        return TrialReceipts(self.root / f"run={run}", {"run": run, **common}, nested=self.nested)
