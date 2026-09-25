"""One query surface over project-owned results, including work still running."""

import json
import os
import sqlite3
from collections.abc import Collection
from contextlib import closing
from datetime import UTC
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import polars as pl

from .dispatch import vocabulary
from .dispatch.shared import db_file
from .observe.files import FrameFile
from .trials.artifacts import Artifact


class Results:
    """Read the files we already collect; no server writes into a shared DuckDB file."""

    def __init__(self, root: Path, *, experiment: str = "") -> None:
        if experiment and (
            experiment in {".", ".."}
            or Path(experiment).name != experiment
            or any(c in experiment for c in "*?[]")
        ):
            raise ValueError("experiment must name one experiment directory")
        self.root = root.resolve()
        self.experiment = experiment

    def query(self, sql: str | Path = "SELECT * FROM runs", *, project: str = "") -> pl.DataFrame:
        """Query a fresh local snapshot of runs, trials, events, artifacts, and dispatch jobs.

        sql: SELECT text or a UTF-8 file Path; strings are never interpreted as filenames.
            Relative file paths and paths inside SQL use the caller's current directory,
            not this Results root or the SQL file's parent.
        project: a research directory name; omitted means all projects, still labeled.
        Network refresh belongs to Mainboard monitor, not to an implicit SQL side effect.
        Jobs separate the last backend_state from the command verdict. The settled flag
        reads the monitor's completion cursor, not current provider liveness.
        Event recorded_at values are UTC timestamps without a timezone annotation; local
        queries never install extensions or need ICU to interpret event offsets.
        """
        if isinstance(sql, Path):
            try:
                sql = sql.expanduser().read_text(encoding="utf-8")
            except UnicodeError as fault:
                raise ValueError(f"SQL file {sql} must contain UTF-8 text: {fault}") from fault
        with duckdb.connect(config={"autoinstall_known_extensions": False}) as connection:
            statements = connection.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise ValueError("results queries must be one SELECT statement")
            self._views(connection, project, sql)
            result = connection.execute(sql)
            return pl.DataFrame(
                result.fetchall(),
                schema=[column[0] for column in result.description],
                orient="row",
                infer_schema_length=None,
            )

    def export(self, sql: str | Path, path: Path, *, project: str = "") -> Path:
        """Export one SELECT to a new CSV, Parquet, or JSON file, inferred from its suffix.

        Publish only a complete file. An existing destination is never overwritten.
        SQL text and UTF-8 file Paths use the same query contract as `query`.
        """
        frame = self.query(sql, project=project)
        writers = {
            ".csv": frame.write_csv,
            ".parquet": frame.write_parquet,
            ".json": frame.write_json,
        }
        try:
            write = writers[path.suffix.casefold()]
        except KeyError:
            raise ValueError("output suffix must be .csv, .parquet, or .json") from None
        path = path.expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as staged:
            temporary = Path(staged) / path.name
            write(temporary)
            with temporary.open("rb+") as completed:
                os.fsync(completed.fileno())
            os.link(temporary, path)
        return path

    def table(
        self, schema: str, *, project: str = "", runs: Collection[str] | None = None
    ) -> pl.DataFrame:
        """Read verified Parquet tables from collected storage, never a source host's path.

        Original repository and machine metadata remain in the _trial provenance column.
        runs: select run identities before reading their artifacts; None selects all runs.
            An empty collection selects none. Selected artifacts still require valid bytes.
        """
        artifacts = self.query("SELECT * FROM artifacts", project=project)
        if runs is not None:
            artifacts = artifacts.filter(
                pl.col("context").str.json_path_match("$.run").is_in(list(runs))
            )
        frames = []
        for row in artifacts.iter_rows(named=True):
            reference = Artifact.model_validate_json(row["reference"])
            if reference.schema_name != schema:
                continue
            if reference.media_type != "application/vnd.apache.parquet":
                raise ValueError(f"{schema} contains a non-Parquet artifact")
            root = Path(row["root"])
            relative = reference.relative
            # References may be project- or workspace-relative. Match only this project's
            # location, including when Results is opened on the project itself.
            root = next(
                (
                    parent
                    for parent in root.parents
                    if relative.is_relative_to(root.relative_to(parent).as_posix())
                ),
                root,
            )
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

    def _views(self, connection: duckdb.DuckDBPyConnection, project: str, sql: str) -> None:
        """Load only the collected sources the SELECT depends on."""
        dependencies: dict[str, set[str]] = {
            "runs": {"trials", "events"},
            "metrics": {"events"},
            "artifacts": {"trials", "events"},
            "trials": set(),
            "events": set(),
            "jobs": set(),
        }
        try:
            requested = {name.casefold() for name in connection.get_table_names(sql)}
        except duckdb.BinderException:
            # DuckDB cannot discover some joins before their column schemas exist.
            # Build the catalog in that case and let execution report real SQL errors.
            requested = set(dependencies)
        needed = requested & dependencies.keys()
        needed |= set().union(*(dependencies[name] for name in tuple(needed)))
        roots = self._projects(project) if needed - {"jobs"} else []
        if "trials" in needed:
            self._trials(connection, roots)
        if "events" in needed:
            self._events(connection, roots)
        if "runs" in needed:
            connection.execute("""
                CREATE VIEW runs AS SELECT DISTINCT project,
                    data->>'run' AS run, data->>'host' AS host,
                    data->>'card_name' AS hardware, data->>'commit' AS commit
                    FROM events WHERE topic = 'started'
                    UNION SELECT DISTINCT project, run, host, card_name AS hardware, commit
                    FROM trials;
            """)
        if "metrics" in needed:
            connection.execute("""
                CREATE VIEW metrics AS SELECT e.project, s.data->>'run' AS run,
                    e.trial, e.recorded_at, e.metadata, e.data
                FROM events e JOIN events s ON e.stream = s.stream AND e.project = s.project
                WHERE e.topic = 'metrics' AND s.topic = 'started';
            """)
        if "artifacts" in needed:
            self._artifacts(connection, roots)
        if "jobs" in needed:
            self._jobs(connection)

    def _trials(self, connection: duckdb.DuckDBPyConnection, roots: list[Path]) -> None:
        # Explicit file inventory per query is the snapshot. Temporary rsync/Parquet files
        # never match; a later query sees newly published immutable fragments automatically.
        connection.execute(
            "CREATE TABLE _receipt_schema(project VARCHAR, run VARCHAR, trial VARCHAR, "
            "verdict VARCHAR, artifacts JSON, host VARCHAR, card_name VARCHAR, commit VARCHAR, "
            "params VARCHAR)"
        )
        inventories = ["SELECT * FROM _receipt_schema"]
        for root in roots:
            parts = [
                str(path)
                for path in root.glob(
                    f"{self.experiment or '*'}/evidence/receipts/run=*/part-*.parquet"
                )
            ]
            if parts:
                name = f"_trials_{len(inventories)}"
                connection.sql(
                    "SELECT ?::VARCHAR AS project, * FROM read_parquet(?, "
                    "union_by_name=true, hive_partitioning=false)",
                    params=[root.parents[1].name, parts],
                ).create_view(name)
                inventories.append(f"SELECT * FROM {name}")
        connection.execute(
            "CREATE VIEW trials AS SELECT DISTINCT * FROM ("
            + " UNION ALL BY NAME ".join(inventories)
            + ")"
        )

    def _events(self, connection: duckdb.DuckDBPyConnection, roots: list[Path]) -> None:
        events: dict[tuple[str, str, int], str] = {}
        for root in roots:
            owner = root.parents[1]
            for path in sorted(
                root.glob(f"{self.experiment or '*'}/evidence/artifacts/*/*/events/*.ndjson*")
            ):
                for frame in FrameFile(path).frames():
                    record = json.dumps(
                        {
                            "project": owner.name,
                            "root": str(owner),
                            **frame.model_dump(mode="json"),
                            "at": frame.at.astimezone(UTC).replace(tzinfo=None).isoformat(),
                        },
                        sort_keys=True,
                    )
                    identity = (owner.name, frame.job, frame.offset)
                    if events.setdefault(identity, record) != record:
                        raise ValueError(
                            f"conflicting event snapshots for project {owner.name!r}, "
                            f"stream {frame.job!r}, offset {frame.offset}"
                        )
        connection.execute(
            """
            CREATE TABLE events AS SELECT DISTINCT
                row->>'project' AS project, row->>'root' AS root,
                row->>'job' AS stream, (row->>'offset')::UBIGINT AS offset,
                (row->>'at')::TIMESTAMP AS recorded_at,
                row->'payload'->>'trial' AS trial,
                row->'payload'->>'topic' AS topic,
                row->'payload'->'metadata' AS metadata,
                row->'payload'->'data' AS data
            FROM unnest(?::JSON[]) AS records(row)
        """,
            [list(events.values())],
        )

    @staticmethod
    def _artifacts(connection: duckdb.DuckDBPyConnection, roots: list[Path]) -> None:
        connection.execute("""
            CREATE VIEW emitted_artifacts AS SELECT DISTINCT e.project, e.stream,
                e.root AS root,
                json_merge_patch(s.data, json_object('verdict',
                    coalesce(t.verdict, v.data->>'verdict'))) AS context,
                e.data->>'name' AS name,
                json_merge_patch(e.data, '{"name":null}') AS reference
            FROM events e JOIN events s ON e.stream = s.stream AND e.project = s.project
            LEFT JOIN events v ON v.stream = s.stream AND v.project = s.project
                AND v.topic = 'settled'
            LEFT JOIN trials t ON t.project = s.project AND t.run = (s.data->>'run')
                AND t.trial = s.trial
            WHERE e.topic = 'artifact' AND s.topic = 'started';
        """)
        # Final receipts own artifact references even when a live event stream was interrupted.
        # Preserve surviving events verbatim; this fallback does not invent their lost times.
        # An artifact's provenance does not contain its siblings' references. Replicating the
        # full artifact index into every row made a wide trial consume quadratic memory.
        connection.execute("""
            CREATE VIEW receipt_contexts AS SELECT project, run, trial,
                json_merge_patch(to_json(t), json_object('params', t.params::JSON)) AS context
            FROM (SELECT * EXCLUDE (artifacts) FROM trials) t;
        """)
        connection.execute(
            """
            CREATE TABLE receipt_artifacts AS SELECT DISTINCT
                t.project,
                regexp_extract(t.artifacts::JSON->>'events',
                    'artifacts/([^/]+/[^/]+)/events$', 1) AS stream,
                p.row->>'root' AS root,
                c.context AS context,
                a.key AS name, a.value AS reference
            FROM trials t JOIN receipt_contexts c USING (project, run, trial),
                json_each(t.artifacts::JSON) a,
                unnest(?::JSON[]) AS p(row)
            WHERE t.project = (p.row->>'project')
                AND json_type(a.value) = 'OBJECT'
                AND (a.value->>'media_type') IS NOT NULL
                AND (a.value->>'path') IS NOT NULL;
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
                    AND e.stream = r.stream AND e.name = r.name
                    AND (e.reference->>'path') = (r.reference->>'path')
            );
        """)

    def _jobs(self, connection: duckdb.DuckDBPyConnection) -> None:
        jobs = []
        path = db_file(self.root)
        if path.is_file():
            with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as state:
                jobs = [row[0] for row in state.execute("SELECT data FROM runs")]
        connection.execute(
            """
            CREATE TABLE jobs AS SELECT row->>'target' AS server,
                row->>'handle' AS handle, row->>'submitted_at' AS submitted_at,
                row->>'state' AS backend_state, row->>'verdict' AS verdict,
                row->>'evidence' AS evidence,
                coalesce((row->>'reported') = (row->>'verdict')
                    AND (row->>'verdict') = ANY(?::VARCHAR[]), false) AS settled,
                row->>'fetch_path' AS results, row AS metadata
            FROM unnest(?::JSON[]) AS records(row)
        """,
            [sorted(vocabulary.TERMINAL), jobs],
        )
