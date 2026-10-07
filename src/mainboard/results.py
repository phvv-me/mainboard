"""One query surface over project-owned results, including work still running."""

import json
import os
from collections.abc import Collection, Generator
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from .core.errors import MissionError
from .dispatch import vocabulary
from .observe.files import FrameFile
from .runtime.tree import FileBudget
from .state.evidence import EvidenceTree
from .state.lake import ALIAS, Lake, ndjson, quoted
from .state.relations import Relation, Relations, records


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

    def rows(self, sql: str | Path = "SELECT * FROM runs", *, project: str = "") -> list[dict]:
        """Query a fresh local snapshot of runs, trials, events, artifacts, dispatch jobs and
        the workspace's state lake (`lake.<table>`), one mapping per row.

        sql: SELECT text or a UTF-8 file Path; strings are never filenames. Relative paths, of
            the file or inside SQL, use the caller's current directory, not this root or the
            SQL file's parent.
        project: a research directory name; omitted means all projects, still labeled.
        Network refresh belongs to `mb job list`, never an SQL side effect.
        """
        relations = Relations()
        with self._connected(sql, project, relations.connection) as text:
            return records(relations.connection.sql(text))

    def query(self, sql: str | Path = "SELECT * FROM runs", *, project: str = "") -> Relation:
        """`rows` as a DuckDB relation, for plotting and experiments, read in whole so it
        outlives the lake and every file the query read."""
        relations = Relations()
        with self._connected(sql, project, relations.connection) as text:
            return relations.kept(text)

    def export(self, sql: str | Path, path: Path, *, project: str = "") -> Path:
        """Export one SELECT to a new CSV, Parquet, or JSON file, inferred from its suffix.

        Only a complete file is published and an existing destination is never overwritten.
        """
        formats = {
            ".csv": "FORMAT csv, HEADER",
            ".parquet": "FORMAT parquet, COMPRESSION zstd",
            ".json": "FORMAT json, ARRAY true",
        }
        try:
            options = formats[path.suffix.casefold()]
        except KeyError:
            raise ValueError("output suffix must be .csv, .parquet, or .json") from None
        path = path.expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as staged:
            temporary = Path(staged) / path.name
            relations = Relations()
            with self._connected(sql, project, relations.connection) as text:
                target = temporary.as_posix().replace("'", "''")
                relations.connection.execute(f"COPY ({text}) TO '{target}' ({options})")
            with temporary.open("rb+") as completed:
                os.fsync(completed.fileno())
            os.link(temporary, path)
        return path

    @contextmanager
    def _connected(
        self, sql: str | Path, project: str, connection: duckdb.DuckDBPyConnection
    ) -> Generator[str]:
        """`connection` holding every results view, and the lake while the SQL that names it
        runs; the SQL text."""
        if isinstance(sql, Path):
            try:
                sql = sql.expanduser().read_text(encoding="utf-8")
            except UnicodeError as fault:
                raise ValueError(f"SQL file {sql} must contain UTF-8 text: {fault}") from fault
        statements = connection.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("results queries must be one SELECT statement")
        attached = f"{ALIAS}." in sql.lower()
        with FileBudget.permitted():
            self._views(connection, project, sql)
            if attached:
                # Everything the workspace recorded, read-only beside the collected results; a
                # workspace that recorded nothing yet gets its empty lake, not a missing schema.
                Lake.at(self.root).current().attach(connection)
            try:
                yield sql
            except duckdb.Error as fault:
                raise MissionError(str(fault).strip()) from None
            finally:
                if attached:
                    connection.execute(f"DETACH {ALIAS}")

    def table(
        self, schema: str, *, project: str = "", runs: Collection[str] | None = None
    ) -> Relation:
        """Read verified Parquet tables from collected storage, never a source host's path.

        Original repository and machine metadata remain in the _trial provenance column.
        runs: select run identities before reading their artifacts; None selects all runs.
            An empty collection selects none. Selected artifacts still require valid bytes.
        """
        from .trials.artifacts import NO_TABLE, Artifact

        listed = self.rows("SELECT root, context, reference FROM artifacts", project=project)
        if runs is not None:
            wanted = set(runs)
            listed = [row for row in listed if json.loads(row["context"]).get("run") in wanted]
        relations = Relations()
        tables = []
        for row in listed:
            reference = Artifact.model_validate_json(row["reference"])
            if reference.schema_name != schema:
                continue
            if reference.media_type != "application/vnd.apache.parquet":
                raise ValueError(f"{schema} contains a non-Parquet artifact")
            # References may be project- or workspace-relative. Match only this project's
            # location, including when Results is opened on the project itself.
            root = Path(row["root"])
            root = next(
                (
                    parent
                    for parent in root.parents
                    if reference.relative.is_relative_to(root.relative_to(parent).as_posix())
                ),
                root,
            )
            table = relations.parquet(reference.read(root))
            if "_trial" in table.columns:
                raise ValueError("artifact payload reserves the _trial provenance column")
            tables.append(table.project(f"*, {quoted(str(row['context']))} AS _trial"))
        return relations.union(tables, empty=NO_TABLE)

    def _projects(self, project: str) -> list[Path]:
        tree = EvidenceTree(self.root)
        candidates = [
            *tree.directories("research/*/datasets/experiments"),
            *tree.directories("datasets/experiments"),
        ]
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
            # A qualified name (`lake.runs`) is a table somewhere else, never one of these views.
            named = connection.get_table_names(sql, qualified=True)
            requested = {name.casefold() for name in named if "." not in name}
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
        # Explicit file inventory per query is the snapshot. Temporary transfer/Parquet files
        # never match; a later query sees newly published immutable fragments automatically.
        # Read in once as a table: every view below reads it again, and a lazy DISTINCT over
        # cutok's 3,625 fragments feeding an expansion took 19 GiB where the table takes 0.14.
        connection.execute(
            "CREATE TABLE _receipt_schema(project VARCHAR, run VARCHAR, trial VARCHAR, "
            "verdict VARCHAR, artifacts JSON, host VARCHAR, card_name VARCHAR, commit VARCHAR, "
            "params VARCHAR)"
        )
        inventories = ["SELECT * FROM _receipt_schema"]
        for root in roots:
            parts = [
                str(path)
                for path in EvidenceTree(root).files(
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
            "CREATE TABLE trials AS SELECT DISTINCT * FROM ("
            + " UNION ALL BY NAME ".join(inventories)
            + ")"
        )

    def _events(self, connection: duckdb.DuckDBPyConnection, roots: list[Path]) -> None:
        events: dict[tuple[str, str, int], str] = {}
        for root in roots:
            owner = root.parents[1]
            for path in EvidenceTree(root).files(
                f"{self.experiment or '*'}/evidence/artifacts/*/*/events/*.ndjson*"
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
        with ndjson(events.values()) as staged:
            connection.execute(f"""
                CREATE TABLE events AS SELECT DISTINCT
                    row->>'project' AS project, row->>'root' AS root,
                    row->>'job' AS stream, (row->>'offset')::UBIGINT AS offset,
                    (row->>'at')::TIMESTAMP AS recorded_at,
                    row->'payload'->>'trial' AS trial,
                    row->'payload'->>'topic' AS topic,
                    row->'payload'->'metadata' AS metadata,
                    row->'payload'->'data' AS data
                FROM (SELECT json AS row FROM {staged})
            """)

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
        #
        # Each step is its own table. `json_each` hands every entry the whole document it came
        # from, and parsing a receipt's artifacts in the statement that expands them did the
        # same, so cutok's 230,000 references (a receipt holds up to 1.5 MB of them) ran a
        # 24 GB machine out of memory; parsed once into a map, then expanded, they take 0.9 GiB.
        connection.execute("""
            CREATE TABLE receipt_contexts AS SELECT project, run, trial,
                json_merge_patch(to_json(t), json_object('params', t.params::JSON)) AS context
            FROM (SELECT * EXCLUDE (artifacts) FROM trials) t;
            CREATE TABLE receipt_maps AS SELECT project, run, trial,
                regexp_extract(artifacts::JSON->>'events',
                    'artifacts/([^/]+/[^/]+)/events$', 1) AS stream,
                json_transform(artifacts::JSON, '"MAP(VARCHAR, JSON)"') AS listed
            FROM trials;
            CREATE TABLE receipt_entries AS SELECT project, run, trial, stream,
                unnest(map_entries(listed)) AS entry
            FROM receipt_maps;
            DROP TABLE receipt_maps;
        """)
        connection.execute(
            """
            CREATE TABLE receipt_artifacts AS SELECT DISTINCT
                r.project, r.stream, p.row->>'root' AS root, c.context AS context,
                r.entry.key AS name, r.entry.value AS reference
            FROM receipt_entries r JOIN receipt_contexts c USING (project, run, trial),
                unnest(?::JSON[]) AS p(row)
            WHERE r.project = (p.row->>'project')
                AND json_type(r.entry.value) = 'OBJECT'
                AND (r.entry.value->>'media_type') IS NOT NULL
                AND (r.entry.value->>'path') IS NOT NULL;
            """,
            [
                [
                    json.dumps({"project": root.parents[1].name, "root": str(root.parents[1])})
                    for root in roots
                ]
            ],
        )
        connection.execute("DROP TABLE receipt_entries")
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
        jobs: list[str] = []
        lake = Lake.at(self.root)
        if lake.exists():
            jobs = [record for (record,) in lake.query(f"SELECT record FROM {ALIAS}.runs")]
        terminal = ", ".join(f"'{word}'" for word in sorted(vocabulary.TERMINAL))
        with ndjson(jobs) as staged:
            connection.execute(f"""
                CREATE TABLE jobs AS SELECT row->>'target' AS server,
                    row->>'handle' AS handle, row->>'submitted_at' AS submitted_at,
                    row->>'state' AS backend_state, row->>'verdict' AS verdict,
                    row->>'evidence' AS evidence,
                    coalesce((row->>'reported') = (row->>'verdict')
                        AND (row->>'verdict') IN ({terminal}), false) AS settled,
                    row->>'fetch_path' AS results, row AS metadata
                FROM (SELECT json AS row FROM {staged})
            """)
