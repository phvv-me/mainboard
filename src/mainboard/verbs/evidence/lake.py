import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.project import Project
from ...dispatch.collection.collector import Collector
from ...dispatch.collection.linking import link
from ...lint.files.git import nested_projects
from ...log import logger
from ...render import progress
from ...state import DirectoryReplica, Evidence, Lake, Serving
from ...state.evidence import within
from ..base import Verbs
from ..options import Output

if TYPE_CHECKING:
    pass


# The verbs' defaults: compact output, the lake served on Quack's own port.
_COMPACT = Output()
_SERVED = Serving()


class LakeVerbs(Verbs):
    """The `lake` verbs."""

    def compact(self) -> int:
        """Compact the lake: inlined rows to Parquet, small files merged, old snapshots expired.

        Files past their retention are deleted too. Appends keep committing meanwhile; exits 1
        when another process is already compacting.
        """
        if Lake.at(self.session.workspace_root()).ready().maintain():
            logger.result("compacted")
            return 0
        logger.info("another process is compacting this lake")
        return 1

    def dedup(self, *paths: Path, dry_run: bool = False, output: Output = _COMPACT) -> None:
        """Hard-link identical evidence files to one copy, to give a host's disk back.

        Needs no lake, so a job host runs it over its own tree. Nothing is deleted: a duplicate's
        name moves onto the copy only once both hash equal, a file named by its digest must hash
        to it, and files still written or of a run that wrote within the hour are left alone.
        Bytes are distinct inodes, before and after.

        Args:
            paths: evidence directories or files inside this workspace, on one filesystem.
            dry_run: count what would be linked, linking nothing.
        """
        root = self.session.workspace_root()
        cache = Collector(root).digests
        rows: list[dict[str, str | int]] = [
            {
                "path": relative,
                **link(str(root), relative=relative, cache=cache, dry=dry_run),
            }
            for relative in (within(root, path=path) for path in paths)
        ]
        output.print_rows(rows, title="lake dedup")

    def evict(self, *paths: Path, output: Output = _COMPACT) -> None:
        """Delete tree copies of evidence the lake keeps, to give the disk back.

        A file goes only when the lake indexes that path with the same bytes and holds an
        intact object; anything else stays. Naming paths says nothing writes them any more.
        Each file is moved aside, checked against the lake once more and only then removed, so
        one changed meanwhile stays, and so do the parts of an artifact archive, whose members
        readers open by path. Readers recall evicted files from the lake, and `materialize`
        writes them back.

        Args:
            paths: evidence files or directories inside this workspace, no longer written.
        """
        keeper = Evidence(Lake.at(self.session.workspace_root()))
        with progress("evicting tree copies the lake keeps"):
            evicted = keeper.evict(keeper.finalize(paths))
        output.print_rows([{"evicted": len(evicted)}], title="lake evict", columns=("evicted",))

    def ingest(self, *paths: Path, output: Output = _COMPACT) -> None:
        """Keep evidence files in the lake byte for byte, so they can leave version control.

        Every file at or under each path lands once per content in the lake's `blobs`, and its
        workspace-relative path in `lake.evidence`; readers of receipts, artifacts and events
        then find it there once it is gone from disk. A file whose object the lake already holds
        whole is not stored again, and one it holds damaged is stored afresh, so a rerun also
        finishes an interrupted ingest. Nothing on disk is moved or changed.

        Args:
            paths: evidence files or directories inside this workspace.
        """
        with progress("keeping evidence in the lake"):
            kept = Evidence(Lake.at(self.session.workspace_root())).ingest(paths)
        output.print_rows([kept.model_dump()], title="lake ingest")

    def lake_check(self, *, output: Output = _COMPACT) -> int:
        """Compare what the lake's catalog references with what is on disk; exit 1 on a loss.

        A deleted data file breaks only its table and `count(*)` hides it, so this is what finds
        one; a catalog WAL lost after a crash is named too. Then every indexed evidence file's
        object is read back a window at a time: whole at its size, each chunk matching the
        checksum recorded at ingest, and all of it hashing to its digest. Nothing is written,
        not even a checksum for a chunk kept before checksums existed.

        Args:
        """
        lake_ = Lake.at(self.session.workspace_root()).ready()
        health = lake_.check()
        readable = all(finding.kind == "bloat" for finding in health.findings)
        findings = [*health.findings, *(Evidence(lake_).verify() if readable else ())]
        output.print_rows(
            [finding.model_dump() for finding in findings],
            title="lake check",
            columns=("table", "kind", "detail"),
        )
        return 0 if not findings else 1

    def lake_upgrade(self, *, output: Output = _COMPACT) -> None:
        """Migrate this workspace's lake catalogs to the newest DuckLake spec DuckDB writes.

        That is the workspace's own catalog and each nested project's that keeps one
        (`research/reproducibility/.mainboard`), since a reader of this workspace reads those too.

        One way: run it once every machine reading the lake runs a release that reads that spec.
        """
        root = self.session.workspace_root()
        project = Project()
        homes = [root] + [
            home
            for home in nested_projects(root)
            if any((home / state).is_dir() for state in project.out_dirs)
        ]
        rows = [
            {"lake": str(lake.catalog), "spec": lake.upgrade()}
            for lake in map(Lake.at, homes)
            if lake.exists()
        ]
        output.print_rows(rows, title="lake upgrade", columns=("lake", "spec"))

    def materialize(self, *paths: Path, output: Output = _COMPACT) -> None:
        """Write evidence the lake keeps back to where it stood in the tree.

        A file already there with the right bytes is left alone; every written file is verified
        against its digest first. A file already there with other bytes is never replaced: the
        verb refuses, naming both digests, and writes nothing.

        Args:
            paths: indexed evidence files or directories, as they stood in this workspace.
        """
        written = Evidence(Lake.at(self.session.workspace_root())).materialize(paths)
        output.print_rows(
            [{"path": str(path)} for path in written], title="lake materialize", columns=("path",)
        )

    def replicate(self, directory: Path, output: Output = _COMPACT) -> None:
        """Copy every evidence object the lake keeps, and its path index, to a second disk.

        Only objects the replica lacks are read, each verified before it takes its name, so a
        rerun after new ingests copies just those. Run it after every `ingest` whose files are
        leaving version control: until then the lake is their only copy.

        Args:
            directory: the replica's root on another disk, created when missing.
        """
        with progress("replicating the lake's evidence"):
            copied = Evidence(Lake.at(self.session.workspace_root())).replicate(
                DirectoryReplica(directory)
            )
        output.print_rows(
            [{"directory": str(directory), "copied": copied}], title="lake replicate"
        )

    def serve(self, *, served: Serving = _SERVED) -> None:
        """Serve this workspace's lake over DuckDB's Quack protocol until interrupted.

        Listens on localhost only, on two ports: the writer on `port`, and the next one for reads
        alone, where DuckLake is opened READ_ONLY. Another machine reaches them through an ssh
        tunnel (`ssh -R <port>:localhost:<port> <host>` from here, or `-L` from there) and sets
        `MB_LAKE=quack:localhost:<port>` with `MB_LAKE_TOKEN` for appends, and
        `MB_LAKE_READ=quack:localhost:<port+1>` with `MB_LAKE_READ_TOKEN` for reads. Commands
        here keep using the lake's files. Both ends must run the same DuckDB release
        (`mb self version` names it).
        """
        lake_ = Lake.at(self.session.workspace_root())
        with lake_.serving(served) as (uri, _):
            kept = "the one given" if served.token else str(lake_.token)
            said = (
                f"serving {lake_.catalog} at {uri} (token: {kept}) and read-only at "
                f"quack:localhost:{served.read_port} (token: {lake_.read_token}); Ctrl-C stops it"
            )
            logger.result(said)
            with suppress(KeyboardInterrupt):
                while True:
                    time.sleep(3600)
