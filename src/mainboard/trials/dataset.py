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
from itertools import chain
from typing import TYPE_CHECKING

from ..state.evidence import EvidenceTree
from ..state.lake import quoted
from ..state.relations import Relations, records
from .artifacts import NO_TABLE, Artifact
from .coverage import PROBED, Cell, LaneStatus
from .ledger import NESTED, TrialReceipts, wire
from .provenance import Admissibility
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

    from ..state.relations import Relation

# The two columns every receipt carries, which a coverage read and a current view both group on.
_LANE, _KEY = "lane", "key"

# Where each run's committed fragments sit under a store.
_PARTS = "run=*/part-*.parquet"

# The creation coordinate every recency question reads, and the field deciding evidence or scratch.
OPENED, ADMISSIBILITY = "opened_at_ns", "admissibility"

# How a run name spells the second it opened, `20260829T094024Z`, all a run written before the
# creation coordinate knew about its own age.
STAMP, DATED = "%Y%m%dT%H%M%SZ", 16

# Where each receipt was written, the fragment's place in `parts` and the row's in its fragment,
# carried by a read so one ordering covers rows a run wrote within one instant, then dropped.
_WRITTEN = ("_part", "_row")

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


def _named(column: str) -> str:
    """`column` as a SQL identifier, whatever it is spelled with."""
    return '"' + column.replace('"', '""') + '"'


def _listed(values: Sequence[str]) -> str:
    """`values` as a SQL list of text, typed even when empty."""
    return f"[{', '.join(quoted(value) for value in values)}]::VARCHAR[]"


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
    def admissible(self) -> str:
        """What a row must be for a claim to lean on it: passed, with an identifiable tree.

        One filter, because a broken lane and an unidentifiable tree fail a claim the same way, and
        a query remembering only the first was the one every review had to correct.
        """
        return (
            f"outcome = {quoted(Outcome.PASSED)} "
            f"AND {ADMISSIBILITY} = {quoted(Admissibility.ADMISSIBLE)}"
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
        held = self._read().aggregate(f"run, max({OPENED})", "run").fetchall()
        found = {str(run): opened_at(str(run), recorded) for run, recorded in held}
        shared: dict[tuple[int, int, str], list[str]] = {}
        for run, coordinate in found.items():
            shared.setdefault(coordinate, []).append(run)
        tied = [runs for runs in shared.values() if len(runs) > 1]
        if tied:
            raise Ambiguous(self.root, sorted(tied[0]))
        return found

    @property
    def parts(self) -> list[Path]:
        """Every committed fragment of every run, in run then write order, those that left the
        tree read back from the lake that keeps them."""
        return EvidenceTree(self.root).files(_PARTS)

    @property
    def runs(self) -> tuple[str, ...]:
        """Every run this store holds, oldest first."""
        found = self.opened
        return tuple(sorted(found, key=lambda run: found[run]))

    @property
    def stored(self) -> frozenset[str]:
        """Every run this store holds, as membership only, so it answers even when `runs` ties."""
        return frozenset(str(run) for (run,) in self._read().project("run").distinct().fetchall())

    @property
    def recency(self) -> str:
        """Each row's run's place among `runs`, oldest first, so a sort reads time and not a
        name."""
        return f"list_position({_listed(self.runs)}, run)"

    @classmethod
    def holding(
        cls, path: Path, *, axes: Sequence[str] = (), nested: Sequence[str] = NESTED
    ) -> Dataset | None:
        """The dataset at `path` itself or at `receipts/` below it, None when neither holds one."""
        for candidate in (path, path / "receipts"):
            if EvidenceTree(candidate).files(_PARTS):
                return cls(candidate, axes=axes, nested=nested)
        return None

    def as_jsonl(self, target: Path, run: str = "") -> int:
        """Write one run, the newest when empty, as `trial_receipt` lines; returns the rows."""
        lines = [wire(row) for row in self.rows(run)]
        target.write_text("".join(lines), encoding="utf-8", newline="\n")
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
        read = self._read()
        if run:
            read = read.filter(f"run = {quoted(run)}")
        return frozenset(str(lane) for (lane,) in read.project(_LANE).distinct().fetchall())

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

    def passing(self, *, every: bool = False) -> Relation:
        """The store's admissible passing rows, one per cell by default and all of them when asked.

        The default is what a table renders from: each cell's row from the run that most recently
        produced it, by creation coordinate, so a re-run supersedes without anyone choosing a run.
        A program whose cells owe several samples asks for `every` instead.
        """
        read = self._read()
        grouped = ", ".join(_named(column) for column in (_LANE, _KEY, *self.coordinates))
        written = ", ".join(_WRITTEN)
        if every:
            query = (
                f"SELECT * EXCLUDE ({written}) FROM receipts WHERE {self.admissible} "
                f"ORDER BY {grouped}, {self.recency}, {written}"
            )
        else:
            query = (
                f"SELECT * EXCLUDE ({written}, _latest) FROM (SELECT *, row_number() OVER "
                f"(PARTITION BY {grouped} ORDER BY {self.recency} DESC, _part DESC, _row DESC) "
                f"AS _latest FROM receipts WHERE {self.admissible}) WHERE _latest = 1 "
                f"ORDER BY {grouped}"
            )
        return read.query("receipts", query)

    def rows(self, run: str = "") -> list[dict[str, JsonValue]]:
        """One run's receipts, the newest run when empty, as plain records with JSON decoded."""
        chosen = run or self.newest
        held = records(self.scan().filter(f"run = {quoted(chosen)}"))
        return [self.decoded(row) for row in held]

    def tables(self, root: Path, *, schema_name: str, run: str = "") -> Relation:
        """Read verified table artifacts across hardware without selecting a winning run.

        root: the project root that artifact references are relative to.
        schema_name: the scientific table schema, shared by compatible producers.
        run: one explicit run, or every run when empty. `_trial` carries JSON provenance;
            ordinary columns retain the experiment's data and units unchanged.
        """
        receipts = self.scan()
        if run:
            receipts = receipts.filter(f"run = {quoted(run)}")
        relations = Relations()
        tables: list[Relation] = []
        for raw in records(receipts):
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
                provenance = {
                    **metadata,
                    "artifact_name": label,
                    "artifact_sha256": reference.sha256,
                }
                table = relations.parquet(reference.read(root))
                if "_trial" in table.columns:
                    raise ValueError("table payload uses the reserved _trial provenance column")
                trial = quoted(json.dumps(provenance, sort_keys=True))
                tables.append(table.project(f"*, {trial} AS _trial"))
        return relations.union(tables, empty=NO_TABLE)

    def scan(self) -> Relation:
        """Every receipt this store has ever held, across every run, in the order written.

        An axis a run predates reads empty, the same fact as a lane naming no subject.
        Admissibility a run predates reads `unrecorded`, never admissible. A creation coordinate a
        run predates stays null, since a run that never said when it opened has not claimed to be
        the oldest.
        """
        written = ", ".join(_WRITTEN)
        return self._read().order(written).project(f"* EXCLUDE ({written})")

    def _read(self) -> Relation:
        """Every receipt read in whole, normalized as `scan` says, beside where it was written.

        An unwritten store reads as no rows of the columns every read groups or filters on.
        """
        relations = Relations()
        # Each normalized column's SQL around the column it reads, NULL where a run predates it.
        normalized = {
            **dict.fromkeys(self.coordinates, "coalesce(CAST({} AS VARCHAR), '')"),
            ADMISSIBILITY: f"coalesce(CAST({{}} AS VARCHAR), {quoted(Admissibility.UNRECORDED)})",
            OPENED: "CAST({} AS BIGINT)",
        }
        parts = self.parts
        if not parts:
            columns = ", ".join(
                f"{template.format('NULL')} AS {_named(column)}"
                for column, template in {
                    **dict.fromkeys(("run", "outcome", _LANE, _KEY), "CAST({} AS VARCHAR)"),
                    **normalized,
                }.items()
            )
            return relations.kept(f"SELECT {columns}, 0 AS _part, 0 AS _row WHERE false")
        listed = _listed([str(part) for part in parts])
        source = (
            f"read_parquet({listed}, union_by_name = true, hive_partitioning = false, "
            "filename = true, file_row_number = true)"
        )
        held = set(relations.files(parts).columns)
        replaced = ", ".join(
            f"{template.format(_named(column))} AS {_named(column)}"
            for column, template in normalized.items()
            if column in held
        )
        added = "".join(
            f", {template.format('NULL')} AS {_named(column)}"
            for column, template in normalized.items()
            if column not in held
        )
        return relations.kept(
            "SELECT * EXCLUDE (filename, file_row_number)"
            f"{f' REPLACE ({replaced})' if replaced else ''}{added}, "
            f"list_position({listed}, filename) AS _part, file_row_number AS _row FROM {source}"
        )

    def status(self, lane: str, expected: Collection[str], cell: Cell) -> LaneStatus:
        """One lane's completeness at one cell, against its grid and the samples each key owes.

        expected: the keys the lane's own grid would run, declared by being collected.
        cell: the coordinate asked at; every declared axis filters, its probe outcome included,
            since a receipt taken on other silicon or subject does not answer for here.

        Rows from an unidentified tree never count, so a dirty session measures and writes and the
        next clean one still finds the lane owed: scratch work costs the claim nothing.
        """
        names = self.runs
        pinned = (f"{_named(name)} = {quoted(str(value))}" for name, value in cell.filters.items())
        wanted = " AND ".join([f"{_LANE} = {quoted(lane)}", self.admissible, *pinned])
        found = (
            self._read()
            .filter(wanted)
            .aggregate(f"{_KEY}, count(*), max(list_position({_listed(names)}, run)) - 1", _KEY)
            .fetchall()
        )
        taken = {str(key): (int(count), int(order)) for key, count, order in found}
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
