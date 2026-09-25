# The evidence sinks. `Ledger` is the append-only JSONL and csv sink a driver writes when its
# receipts are a stream; `TrialReceipts` is what a test-shaped harness writes, and it is parquet.
#
# A parquet file is not appendable, so a run writes a DATASET: one immutable fragment per trial
# under `run=<run>/part-*.parquet`, staged and renamed so a reader never sees a torn file and a
# sweep dying at trial 400 of 500 keeps 399. Runs cannot share a directory, which fixes the six
# indistinguishable runs once interleaved in one `receipts.jsonl`. Writes are synchronous on
# purpose: the rename is the commit, and concurrency belongs to the dispatch layer, where each job
# writes its own fragments.
#
# The wire is not the store. The printed `trial_receipt` line is how `mainboard monitor` settles a
# remote job, so that boundary stays JSON lines, minted by `wire` both for the `MAINBOARD_RECEIPTS`
# framing file a rented instance hands back and for streaming one run into `mainboard verdict`.

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

# The fields holding a whole object, stored as JSON text so one lane's measurement shape is not
# forced into every other lane's fragment. Writer and reader both read this one tuple.
NESTED = ("params", "measured", "versions", "gates", "artifacts")

# The printed receipt's key, spelled here as `mainboard.verdicts` and `mainboard.dispatch.evidence`
# each spell it, so writing a receipt never imports a lab framework to name a wire contract.
_RECEIPT = "trial_receipt"


def wire(receipt: Mapping[str, JsonValue]) -> str:
    """One receipt as the `trial_receipt` line a dispatch boundary reads, newline included."""
    return json.dumps({_RECEIPT: dict(receipt)}) + "\n"


class Ledger:
    """One run's append-only sink: trial receipts as JSONL, granular measurement rows as csv."""

    def __init__(self, directory: Path, common: Mapping[str, JsonValue]) -> None:
        """common: the fields every receipt here carries."""
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
        """Append rows to a csv, writing the first row's keys as header when the file is new."""
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
    nested: the columns that ride as JSON text.
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
        # Counted, so a second writer on a partition adds to its fragments instead of overwriting.
        self.written = len(self.parts)

    @property
    def parts(self) -> list[Path]:
        """This run's committed fragments in write order."""
        return sorted(self.directory.glob("part-*.parquet"))

    def compact(self) -> None:
        """Fold this run's fragments into one file once the run is over.

        A one-row parquet file pays a whole footer and schema, 11,194 bytes a row against 270 in a
        shared file, so fragments buy crash safety during a run and cost 41 times the space after.
        A killed process never gets here and keeps its fragments. The compacted file replaces the
        first fragment before the others go, so a reader mid-compaction sees duplicates, never
        nothing.
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
        """Commit one trial as its own staged-and-renamed fragment; returns the whole row."""
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
