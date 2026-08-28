# THE EVIDENCE SINKS, and there are two because there are two kinds of consumer.
#
# `Ledger` is the append-only JSONL and csv sink a driver writes through when its receipts are a
# stream and its measurements are rows. `TrialReceipts` is what a test-shaped harness writes
# through, and it is PARQUET.
#
# The JSONL design flushed per record so a sweep dying at trial 400 of 500 kept 399, and a parquet
# file is not appendable, so this writes a DATASET: one immutable fragment per trial under
# `run=<run>/part-*.parquet`, staged and renamed so a reader never sees a torn file. That also
# fixes something a single JSONL never had, RUN IDENTITY: six runs interleaved in one
# `receipts.jsonl` were indistinguishable, and here they cannot even share a directory.
#
# THE WIRE IS NOT THE REST. The `trial_receipt` line is a PRINTED contract: a dispatched job on a
# remote host prints it to stdout and `mainboard monitor` settles it from there. That boundary
# stays JSON lines and `wire` is where they are minted, both for the `MAINBOARD_RECEIPTS` framing
# file a rented instance hands back and for the adapter that streams one run of a dataset into
# `mainboard verdict`. What is parquet is storage AT REST.
#
# ASYNC WRITES ARE REFUSED HERE ON PURPOSE. Staging a fragment and renaming it is the whole
# crash-safety design: the rename is the commit, and a writer that returns before the bytes are
# named has given a trial a receipt it may not have. Concurrency belongs one layer out, where a
# dispatch runs several jobs at once and each writes its own fragments.

import csv
import json
import os
from typing import TYPE_CHECKING

import polars as pl

from ..dispatch.evidence import RECEIPTS_VAR

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

# The fields whose value is a whole object rather than a scalar. Parquet wants one schema per
# dataset and a lane's measurements are its own shape, so these ride as JSON text in one column
# instead of forcing every lane's struct into every other lane's fragment. Both ends of the store
# read this one tuple, so a writer and a reader cannot disagree about which columns are encoded.
NESTED = ("params", "measured", "versions", "gates")

# The key a printed trial receipt carries its payload under. Spelled here rather than imported
# for the reason `mainboard.verdicts` and `mainboard.dispatch.evidence` each spell it too: this
# is a wire contract, and writing a receipt must never drag a lab framework in to name it.
_RECEIPT = "trial_receipt"


def wire(receipt: Mapping[str, JsonValue]) -> str:
    """One receipt as the `trial_receipt` LINE a dispatch boundary reads, newline included."""
    return json.dumps({_RECEIPT: dict(receipt)}) + "\n"


class Ledger:
    """The append-only receipt and table sink of one run, so a verdict is read from disk.

    Every trial lands as a `trial_receipt` line and the granular rows land beside it as csv,
    which is the split an evidence folder wants: the receipts carry the trial-level outcome and
    the csv carries the measurement at the granularity the run produced it.
    """

    def __init__(self, directory: Path, common: Mapping[str, JsonValue]) -> None:
        """directory: where the two files land. common: the fields every receipt here carries."""
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.common = dict(common)
        self.framed = os.environ.get(RECEIPTS_VAR, "")

    def receipt(self, body: Mapping[str, JsonValue]) -> None:
        """Append one trial receipt, and frame it home when a dispatch staged a file for it."""
        text = wire({**self.common, **body})
        with (self.directory / "receipts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text)
        if self.framed:
            with open(self.framed, "a", encoding="utf-8") as handle:
                handle.write(text)

    def table(self, name: str, rows: Sequence[Mapping[str, JsonValue]]) -> None:
        """Append rows to a csv, writing the header when the file is new.

        name: the file inside this ledger's directory. rows: the granular measurements, whose
        first row's keys are the header.
        """
        if not rows:
            return
        target = self.directory / name
        fresh = not target.exists()
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if fresh:
                writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, list | dict) else value
                        for key, value in row.items()
                    }
                )


class TrialReceipts:
    """One run's parquet fragments, written per trial so a sweep that dies keeps what it took.

    directory: this run's own `run=<run>` partition, which nothing else writes into.
    common: the fields every receipt of this run carries.
    nested: the columns that ride as JSON text, `NESTED` unless a consumer stores other shapes.
    """

    def __init__(
        self,
        directory: Path,
        common: Mapping[str, JsonValue],
        *,
        nested: Sequence[str] = NESTED,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.common = dict(common)
        self.nested = tuple(nested)
        self.framed = os.environ.get(RECEIPTS_VAR, "")
        # Counted rather than started at zero, so a second writer opened on a partition that
        # already holds fragments adds to them instead of overwriting the first writer's trials.
        self.written = len(self.parts)

    @property
    def parts(self) -> list[Path]:
        """This run's committed fragments in name order, which is the order they were written."""
        return sorted(self.directory.glob("part-*.parquet"))

    def compact(self) -> None:
        """Fold this run's fragments into one file, once the run is over and nothing can be lost.

        A ONE-ROW PARQUET FILE PAYS A WHOLE FOOTER AND SCHEMA, measured at 11,194 bytes a row
        against 270 once the same rows share one file, so the fragments buy crash safety DURING a
        run and cost 41 times the space after it. This runs at teardown, which is exactly the
        moment it is safe: a process that was killed never gets here, and its fragments are still
        on disk. The compacted file lands on the first fragment's name before the others are
        removed, so the worst a reader can see mid-compaction is duplicate rows, never no rows.
        """
        parts = self.parts
        if len(parts) < 2:
            return
        frame = pl.concat([pl.read_parquet(part) for part in parts], how="diagonal_relaxed")
        staged = self.directory / "compacting.tmp"
        frame.write_parquet(staged, compression="zstd", compression_level=9)
        staged.replace(parts[0])
        for part in parts[1:]:
            part.unlink()
        self.written = 1

    def write(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Commit one trial as its own fragment, staged and renamed so no reader sees it torn.

        Returns the whole row, common fields included, which is what the caller frames or prints.
        """
        row = {**self.common, **body}
        flat = {
            key: json.dumps(value) if key in self.nested else value for key, value in row.items()
        }
        staged = self.directory / f"part-{self.written:05d}.parquet.tmp"
        pl.DataFrame([flat], infer_schema_length=None).write_parquet(staged, compression="zstd")
        staged.replace(staged.with_suffix(""))
        self.written += 1
        if self.framed:
            with open(self.framed, "a", encoding="utf-8") as handle:
                handle.write(wire(row))
        return row
