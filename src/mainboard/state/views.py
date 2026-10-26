"""The results views a query reads, built from the evidence collected into the workspace: `trials`,
`events`, `runs`, `metrics`, `artifacts` and the dispatch `jobs`."""

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC
from pathlib import Path

import duckdb

from ..dispatch.jobs import vocabulary
from ..observe.files import FrameFile
from .constants import ALIAS
from .evidence import EvidenceTree
from .lake import Lake, ndjson
from .storage.parsed import names

# What each view reads besides itself.
_DEPENDENCIES: dict[str, set[str]] = {
    "runs": {"trials", "events"},
    "metrics": {"events"},
    "artifacts": {"trials", "events"},
    "trials": set(),
    "events": set(),
    "jobs": set(),
}

_RUNS = """
    CREATE VIEW runs AS SELECT DISTINCT project,
        data->>'run' AS run, data->>'host' AS host,
        data->>'card_name' AS hardware, data->>'commit' AS commit
        FROM events WHERE topic = 'started'
        UNION SELECT DISTINCT project, run, host, card_name AS hardware, commit
        FROM trials;
"""

_METRICS = """
    CREATE VIEW metrics AS SELECT e.project, s.data->>'run' AS run,
        e.trial, e.recorded_at, e.metadata, e.data
    FROM events e JOIN events s ON e.stream = s.stream AND e.project = s.project
    WHERE e.topic = 'metrics' AND s.topic = 'started';
"""


class Views:
    """The results views of the workspace at `root`, or of the experiment `experiment` names."""

    @staticmethod
    def _artifacts(connection: duckdb.DuckDBPyConnection, roots: Iterable[Path]) -> None:
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

    @staticmethod
    def _needed(sql: str) -> set[str]:
        """The views `sql` reads, with those they read in turn.

        A qualified name (`lake.runs`) is a table somewhere else, never one of these views. A text
        DuckDB does not serialize (EXPLAIN, COPY, a syntax error) builds every view, and execution
        reports any real error.
        """
        try:
            named, _ = names(sql)
        except ValueError:
            named = frozenset(_DEPENDENCIES)
        needed = {name.casefold() for name in named if "." not in name} & _DEPENDENCIES.keys()
        return needed.union(*(_DEPENDENCIES[name] for name in needed))

    def __init__(self, root: Path, *, experiment: str = "") -> None:
        self.root = root
        self.experiment = experiment

    def _events(self, connection: duckdb.DuckDBPyConnection, roots: Iterable[Path]) -> None:
        events: dict[tuple[str, str, int], str] = {}
        for identity, record in self._frames(roots):
            if events.setdefault(identity, record) != record:
                project, stream, offset = identity
                raise ValueError(
                    f"conflicting event snapshots for project {project!r}, "
                    f"stream {stream!r}, offset {offset}"
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

    def _frames(self, roots: Iterable[Path]) -> Iterator[tuple[tuple[str, str, int], str]]:
        """Every collected event frame as its identity and its row, one JSON line."""
        for root in roots:
            owner = root.parents[1]
            for path in EvidenceTree(root).files(
                f"{self.experiment or '*'}/evidence/artifacts/*/*/events/*.ndjson*"
            ):
                yield from (
                    (
                        (owner.name, frame.job, frame.offset),
                        json.dumps(
                            {
                                "project": owner.name,
                                "root": str(owner),
                                **frame.model_dump(mode="json"),
                                "at": frame.at.astimezone(UTC).replace(tzinfo=None).isoformat(),
                            },
                            sort_keys=True,
                        ),
                    )
                    for frame in FrameFile(path).frames()
                )

    def _jobs(self, connection: duckdb.DuckDBPyConnection) -> None:
        jobs: list[str] = []
        lake = Lake.at(self.root)
        if lake.exists():
            jobs = [record for (record,) in lake.query(f"SELECT record FROM {ALIAS}.runs")]
        terminal = ", ".join(f"'{word}'" for word in sorted(vocabulary.Word.settled()))
        with ndjson(jobs) as staged:
            connection.execute(f"""
                CREATE TABLE jobs AS SELECT row->>'target' AS host,
                    row->>'handle' AS handle, row->>'submitted_at' AS submitted_at,
                    row->>'state' AS backend_state, row->>'verdict' AS verdict,
                    row->>'evidence' AS evidence,
                    coalesce((row->>'reported') = (row->>'verdict')
                        AND (row->>'verdict') IN ({terminal}), false) AS settled,
                    row->>'fetch_path' AS results, row AS metadata
                FROM (SELECT json AS row FROM {staged})
            """)

    def _trials(self, connection: duckdb.DuckDBPyConnection, roots: Iterable[Path]) -> None:
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

    def load(self, connection: duckdb.DuckDBPyConnection, project: str, *, sql: str) -> None:
        """Load only the collected sources the statement depends on."""
        needed = Views._needed(sql)
        roots = self.projects(project) if needed - {"jobs"} else []
        builders: tuple[tuple[str, Callable[[], duckdb.DuckDBPyConnection | None]], ...] = (
            ("trials", lambda: self._trials(connection, roots)),
            ("events", lambda: self._events(connection, roots)),
            ("runs", lambda: connection.execute(_RUNS)),
            ("metrics", lambda: connection.execute(_METRICS)),
            ("artifacts", lambda: Views._artifacts(connection, roots)),
            ("jobs", lambda: self._jobs(connection)),
        )
        for name, build in builders:
            if name in needed:
                build()

    def projects(self, project: str) -> list[Path]:
        """The collected experiment trees of every project, or of `project` alone."""
        tree = EvidenceTree(self.root)
        candidates = [
            *tree.directories("research/*/datasets/experiments"),
            *tree.directories("datasets/experiments"),
        ]
        return [path for path in candidates if not project or path.parents[1].name == project]
