"""One query surface over project-owned results, including work still running."""

import json
import sqlite3
from compression import zstd
from io import BytesIO
from pathlib import Path

import duckdb
import polars as pl

from .dispatch.shared import db_file
from .observe.frames import parse_tail
from .trials.artifacts import Artifact


class Results:
    """Read the files we already collect; no server writes into a shared DuckDB file."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def query(self, sql: str = "SELECT * FROM runs", *, project: str = "") -> pl.DataFrame:
        """Query a fresh local snapshot of runs, trials, events, artifacts, and dispatch jobs.

        project: a research directory name; omitted means all projects, still labeled.
        Network refresh belongs to Mainboard monitor, not to an implicit SQL side effect.
        """
        with duckdb.connect(config={"TimeZone": "UTC"}) as connection:
            self._views(connection, project)
            statements = connection.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise ValueError("results queries must be one SELECT statement")
            result = connection.execute(sql)
            return pl.DataFrame(
                result.fetchall(),
                schema=[column[0] for column in result.description],
                orient="row",
                infer_schema_length=None,
            )

    def table(self, schema: str, *, project: str = "") -> pl.DataFrame:
        """Read matching Parquet artifacts, including those published before trial settlement."""
        artifacts = self.query("SELECT * FROM artifacts", project=project)
        frames = []
        for row in artifacts.iter_rows(named=True):
            reference = Artifact.model_validate_json(row["reference"])
            if reference.schema_name != schema:
                continue
            if reference.media_type != "application/vnd.apache.parquet":
                raise ValueError(f"{schema} contains a non-Parquet artifact")
            root = Path(row["root"])
            prefix = root.relative_to(self.root).as_posix()
            if prefix != "." and reference.path.startswith(f"{prefix}/"):
                root = self.root
            frame = pl.read_parquet(BytesIO(reference.read(root)))
            if "_trial" in frame.columns:
                raise ValueError("artifact payload reserves the _trial provenance column")
            frames.append(frame.with_columns(pl.lit(row["context"]).alias("_trial")))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    def _projects(self, project: str) -> list[Path]:
        candidates = sorted((self.root / "research").glob("*/datasets/experiments"))
        own = self.root / "datasets/experiments"
        if own.is_dir():
            candidates.append(own)
        return [path for path in candidates if not project or path.parents[1].name == project]

    def _views(self, connection: duckdb.DuckDBPyConnection, project: str) -> None:
        roots = self._projects(project)
        # Explicit file inventory per query is the snapshot. Temporary rsync/Parquet files
        # never match; a later query sees newly published immutable fragments automatically.
        parts = [
            str(path)
            for root in roots
            for path in root.glob("*/evidence/receipts/run=*/part-*.parquet")
        ]
        if parts:
            connection.read_parquet(
                parts, union_by_name=True, hive_partitioning=False, filename=True
            ).create_view("_trials")
            connection.execute("""
                CREATE VIEW trials AS SELECT DISTINCT
                    regexp_extract(filename, '([^/]+)/datasets/experiments/', 1) AS project,
                    * EXCLUDE(filename) FROM _trials
            """)
        else:
            connection.execute(
                "CREATE TABLE trials(project VARCHAR, run VARCHAR, trial VARCHAR, "
                "verdict VARCHAR, artifacts JSON, host VARCHAR, card_name VARCHAR, commit VARCHAR)"
            )
        events = []
        for root in roots:
            owner = root.parents[1]
            for path in sorted(root.glob("*/evidence/artifacts/*/*/events/*.ndjson*")):
                raw = path.read_bytes()
                if path.suffix == ".zst":
                    raw = zstd.decompress(raw)
                # A live transfer may end inside a UTF-8 character as well as inside JSON.
                complete, _, _ = raw.rpartition(b"\n")
                for frame in parse_tail(complete.decode() + "\n"):
                    events.append(
                        json.dumps(
                            {
                                "project": owner.name,
                                "root": str(owner),
                                **frame.model_dump(mode="json"),
                            }
                        )
                    )
        connection.execute(
            """
            CREATE TABLE events AS SELECT DISTINCT
                row->>'project' AS project, row->>'root' AS root,
                row->>'job' AS stream, (row->>'offset')::UBIGINT AS offset,
                (row->>'at')::TIMESTAMPTZ AS recorded_at,
                row->'payload'->>'trial' AS trial,
                row->'payload'->>'topic' AS topic,
                row->'payload'->'metadata' AS metadata,
                row->'payload'->'data' AS data
            FROM unnest(?::JSON[]) AS records(row)
        """,
            [events],
        )
        connection.execute("""
            CREATE VIEW runs AS SELECT DISTINCT project,
                data->>'run' AS run, data->>'host' AS host,
                data->>'card_name' AS hardware, data->>'commit' AS commit
                FROM events WHERE topic = 'started'
                UNION SELECT DISTINCT project, run, host, card_name AS hardware, commit
                FROM trials;
            CREATE VIEW emitted_artifacts AS SELECT DISTINCT e.project, e.stream,
                coalesce(s.data->>'repository', e.root) AS root,
                json_merge_patch(s.data, json_object('verdict',
                    coalesce(t.verdict, v.data->>'verdict'))) AS context,
                e.data->>'name' AS name,
                json_merge_patch(e.data, '{"name":null}') AS reference
            FROM events e JOIN events s ON e.stream = s.stream AND e.project = s.project
            LEFT JOIN events v ON v.stream = s.stream AND v.project = s.project
                AND v.topic = 'settled'
            LEFT JOIN trials t ON t.project = s.project AND t.run = s.data->>'run'
                AND t.trial = s.trial
            WHERE e.topic = 'artifact' AND s.topic = 'started';
            CREATE VIEW metrics AS SELECT e.project, s.data->>'run' AS run,
                e.trial, e.recorded_at, e.metadata, e.data
            FROM events e JOIN events s ON e.stream = s.stream AND e.project = s.project
            WHERE e.topic = 'metrics' AND s.topic = 'started';
        """)
        # Final receipts own artifact references even when a live event stream was interrupted.
        # Preserve surviving events verbatim; this fallback does not invent their lost times.
        connection.execute(
            """
            CREATE TABLE receipt_artifacts AS SELECT DISTINCT
                t.project,
                regexp_extract(t.artifacts::JSON->>'events',
                    'artifacts/([^/]+/[^/]+)/events$', 1) AS stream,
                p.row->>'root' AS root, to_json(t) AS context,
                a.key AS name, a.value AS reference
            FROM trials t, json_each(t.artifacts::JSON) a,
                unnest(?::JSON[]) AS p(row)
            WHERE t.project = p.row->>'project'
                AND json_type(a.value) = 'OBJECT'
                AND a.value->>'media_type' IS NOT NULL
                AND a.value->>'path' IS NOT NULL;
            """,
            [
                [
                    json.dumps({"project": root.parents[1].name, "root": str(root.parents[1])})
                    for root in roots
                ]
            ],
        )
        connection.execute("""
            CREATE VIEW artifacts AS SELECT * FROM emitted_artifacts
            UNION ALL SELECT r.* FROM receipt_artifacts r
            WHERE NOT EXISTS (
                SELECT 1 FROM emitted_artifacts e WHERE e.project = r.project
                    AND e.reference->>'path' = r.reference->>'path'
            );
        """)
        jobs = []
        path = db_file(self.root)
        if path.is_file():
            with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as state:
                jobs = [row[0] for row in state.execute("SELECT data FROM runs")]
        connection.execute(
            """
            CREATE TABLE jobs AS SELECT row->>'target' AS server,
                row->>'handle' AS handle, row->>'submitted_at' AS submitted_at,
                row->>'state' AS state, row->>'verdict' AS verdict,
                row->>'fetch_path' AS results, row AS metadata
            FROM unnest(?::JSON[]) AS records(row)
        """,
            [jobs],
        )
