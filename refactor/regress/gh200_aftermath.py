"""What the gap before a call does to the call: one v26 NFC delivery timed after each kind of gap.

Run inside the cutok environment on the node under test (any host works; the question is the
GH200). One process, one cell (ascii 64 KiB by default), the gap kinds interleaved in shuffled
blocks, every call checked against Python's NFC. The harness writes a table row after each call;
this isolates which part of that row, if any, costs the next call.

    python gh200_aftermath.py --out aftermath.jsonl [--blocks 40] [--size 65536]

Gap kinds:
  none         the previous call ended a microsecond ago (the harness's call 0 after a warmup)
  sleep50      50 ms of idle, nothing else running
  spin50       50 ms of busy work on this thread alone
  polars       September's row: one-row frame written to Parquet bytes through Polars
  duckdb_new   v26's row: a fresh in-memory DuckDB database per row (mainboard's Relations)
  duckdb_one   the same row through one long-lived connection (no database per row)
  duckdb_t4    a fresh database per row limited to 4 threads
"""

import argparse
import io
import json
import os
import random
import unicodedata
from time import perf_counter, sleep

import cupy as cp
import numpy as np
import psutil

TEMPLATES = {
    "ascii": "Ordinary ASCII text requires no canonical composition. ",
    "decomposed": "Café, déjà vu, Å, ñ, ạ́, 한글. ",
}
ROW = {
    "corpus": "ascii", "target_bytes": 65536, "engine": "cutok", "call": 0, "repetition": 0,
    "accepted": True, "input_bytes": 65120, "documents": 73, "corpus_digest": "0" * 64,
    "seconds": 0.001, "mb_per_second": 60.0, "boundary": "device_ready_utf8", "workers": 72,
}


def spin(milliseconds: float) -> None:
    end = perf_counter() + milliseconds / 1e3
    while perf_counter() < end:
        pass


def polars_row() -> None:
    import polars as pl

    pl.DataFrame([ROW]).write_parquet(io.BytesIO(), compression="zstd")


def duckdb_new_row() -> None:
    from mainboard.state.relations import Relations, parquet_bytes

    parquet_bytes(Relations().rows([ROW]))


_ONE = None


def duckdb_one_row() -> None:
    global _ONE
    from mainboard.state.relations import Relations, parquet_bytes

    _ONE = _ONE or Relations()
    parquet_bytes(_ONE.rows([ROW]))
    _ONE._made = 0
    _ONE.connection.execute("DROP TABLE IF EXISTS t1")


def duckdb_t4_row() -> None:
    from mainboard.state.relations import Relations, parquet_bytes

    relations = Relations()
    relations.connection.execute("SET threads = 4")
    parquet_bytes(relations.rows([ROW]))


def duckdb_open_close() -> None:
    import duckdb

    duckdb.connect(":memory:").close()


def duckdb_open_query() -> None:
    import duckdb

    connection = duckdb.connect(":memory:")
    connection.execute("SELECT 1").fetchall()
    connection.close()


def duckdb_small_memory_row() -> None:
    import duckdb

    from mainboard.state.relations import Relations, parquet_bytes

    relations = Relations()
    relations.connection.execute("SET memory_limit = '256MB'")
    relations.connection.execute("SET temp_directory = ''")
    parquet_bytes(relations.rows([ROW]))
    del relations
    duckdb.default_connection().execute("SELECT 1").fetchall()


def duckdb_no_threads_row() -> None:
    import duckdb

    from mainboard.state.relations import parquet_bytes

    connection = duckdb.connect(":memory:", config={"threads": 1, "pin_threads": "off"})
    relation = connection.sql("SELECT 1 AS a, 'x' AS b")
    parquet_bytes(relation)
    connection.close()


_SHARED = None


def patched_row() -> None:
    """The proposed `Log.table` body: one process-wide database, the row's table dropped after its bytes."""
    global _SHARED
    from mainboard.state.relations import Relations, parquet_bytes

    _SHARED = _SHARED or Relations()
    relation = _SHARED.rows([ROW])
    try:
        parquet_bytes(relation)
    finally:
        _SHARED.connection.execute(f"DROP TABLE {relation.alias}")


GAPS = {
    "none": lambda: None,
    "patched_row": patched_row,
    "sleep50": lambda: sleep(0.05),
    "spin50": lambda: spin(50),
    "polars": polars_row,
    "duckdb_new": duckdb_new_row,
    "duckdb_one": duckdb_one_row,
    "duckdb_t4": duckdb_t4_row,
    "db_open_close": duckdb_open_close,
    "db_open_query": duckdb_open_query,
    "db_threads1": duckdb_no_threads_row,
}


def main() -> None:
    from cutoken.device.spans import Chunks
    from cutoken.spec.steps import Compose
    from cutoken.stages.pack.host_batch import HostBatch
    from cutoken.stages.rewrite.stages import RewriteStage
    from cutoken.unicode.database import UnicodeDatabase
    from patos.cuda.runtime import Workspace

    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="ascii")
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--blocks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    stage = RewriteStage.of(Compose(form="NFC"), UnicodeDatabase.load("17.0.0"), arrays=cp)
    workspace = Workspace(cp)
    document = TEMPLATES[args.template] * 16
    texts = [document] * max(1, args.size // len(document.encode()))
    expected = np.frombuffer("".join(unicodedata.normalize("NFC", t) for t in texts).encode(), np.uint8)

    def call() -> float:
        cp.cuda.runtime.deviceSynchronize()
        started = perf_counter()
        raw, offsets = HostBatch.from_documents(texts).direct()
        output = stage.stage(Chunks.whole(cp.asarray(raw), cp.asarray(offsets), arrays=cp), workspace=workspace)
        spans = output.spans
        output = output.rewritten(spans.text.bytes.copy(), starts=spans.starts.copy(), ends=spans.ends.copy())
        cp.cuda.runtime.deviceSynchronize()
        seconds = perf_counter() - started
        assert np.array_equal(cp.asnumpy(output.spans.text.bytes), expected)
        return seconds * 1e3

    for _ in range(10):
        call()
        duckdb_new_row()
        polars_row()
    rng = random.Random(args.seed)
    rows = []
    kinds = list(GAPS)
    for block in range(args.blocks):
        rng.shuffle(kinds)
        for kind in kinds:
            call()
            gap_started = perf_counter()
            GAPS[kind]()
            gap_ms = (perf_counter() - gap_started) * 1e3
            threads = psutil.Process().num_threads()
            rows.append(
                {"block": block, "gap": kind, "gap_ms": gap_ms, "ms": call(), "threads": threads,
                 "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", ""), "host": os.uname().nodename}
            )
    with open(args.out, "a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    import statistics as st

    print(f"{'gap':12} {'n':>4} {'median ms':>10} {'p10':>8} {'p90':>8} {'gap ms':>8}  (omp {os.environ.get('OMP_WAIT_POLICY', 'unset')}, {os.uname().nodename})")
    for kind in GAPS:
        values = sorted(r["ms"] for r in rows if r["gap"] == kind)
        gaps = [r["gap_ms"] for r in rows if r["gap"] == kind]
        print(f"{kind:12} {len(values):4d} {st.median(values):10.3f} {values[len(values) // 10]:8.3f} {values[-len(values) // 10 - 1]:8.3f} {st.median(gaps):8.1f}")


if __name__ == "__main__":
    main()
