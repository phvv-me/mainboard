"""What a dispatch ships: the workspace mirrored to a host, its staged scripts and listings, and
the snapshot a job runs from."""

import hashlib
import shlex
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from sqlalchemy import Column, Table, select

from ...core.project import Project
from ...engines.compile.generated import GeneratedFiles
from ...engines.compile.inputs.vendor import vendor_root
from ...engines.compile.pixi.lockfile import Lockfile
from ...log import logger
from ...state import schema
from ...state.lake import Lake
from ..agent import Agent, Scope, SshLink
from ..constants import CARD_LEASES
from ..mirroring.mirror import Mirror
from ..mirroring.snapshots import Image, Mirrored, Sealed, Snapshots, writable
from ..mirroring.sync import GitignoreFilter, SyncLock, denied, patterns
from ..provenance import Source, SourceTree
from ..shared import now, state_path
from ..state.cache import Cache
from ..transport.core import SshTransport
from ..transport.shells import Dialect

if TYPE_CHECKING:
    from ...context.plan import ExecutionPlan
    from ..jobs import JobSpec
    from ..mirroring.shipment import Shipment

# What never belongs in a vendored path dependency's copy on a host: caches and environments
# beside its sources, which no build reads and one of which is gigabytes. Its own list because
# the vendored tree is its own scope, walked through its links without the workspace's ignores.
_VENDOR_EXCLUDE = ("__pycache__/", "*.pyc", ".git", ".pixi/", ".venv/")

# The directories whose children are projects: `research/cutok` is the project `cutok`.
_PROJECT_PARENTS = frozenset({"research", "packages", "personal"})


def pin_key(source: Source, prefix: str = "") -> str:
    """The snapshot a dispatch of `source` activating the built environment `prefix` runs from.

    A snapshot freezes its generated tree and its environment link when it is first pinned, so
    two dispatches of one tree share it only while they activate one environment. The
    environment's digest (the prefix's last segment) joins the name: a re-solved lock pins a fresh
    snapshot instead of finding one frozen to the environment before it, which refused every
    later dispatch of an unchanged tree (gold, 2026-09-29).
    """
    digest = prefix.rpartition("/")[2]
    return f"{source.key}-{digest[:12]}" if digest else source.key


class Shipper:
    """Mirror the workspace to hosts and stage what a job carries there.

    Every workspace path (state database, staged scripts, the mirror's include list) resolves
    against the mirror filter's root rather than the cwd, since the mirror decides where the
    workspace begins.
    """

    @staticmethod
    def _dispatched_from(root: Path, here: Path | None = None) -> str:
        """The project a command typed in `here` (the working directory) dispatches for.

        `MB_PROJECT` names it outright; otherwise it is the directory under `research/` or
        `packages/` holding `here`, or the top-level one, and empty at the root or outside it.
        """
        if named := Project().variable("PROJECT").read():
            return named
        try:
            parts = (here or Path.cwd()).resolve().relative_to(root.resolve()).parts
        except ValueError:
            return ""
        if len(parts) > 1 and parts[0] in _PROJECT_PARENTS:
            return parts[1]
        return parts[0] if parts else ""

    @staticmethod
    def _recorded(name: str, content: bytes) -> tuple[Table, Column, str, list[dict[str, object]]]:
        """How one staged file is recorded: its table, the column and value naming it, its rows.

        A closure listing is one row per listed file, its tab-separated path, blob and status; any
        other staged file is a job script, kept whole beside the SHA-256 of its bytes.
        """
        stamp = now()
        if name.startswith("closure-"):
            closure = name.removeprefix("closure-").removesuffix(".tsv")
            listed = (line.split("\t") for line in content.decode().splitlines() if line)
            rows: list[dict[str, object]] = [
                {"ts": stamp, "closure": closure, "path": path, "blob": blob, "status": status}
                for path, blob, status in listed
            ]
            return schema.closures, schema.closures.c.closure, closure, rows
        script: dict[str, object] = {
            "ts": stamp,
            "name": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "script": content.decode("utf-8", "replace"),
        }
        return schema.job_specs, schema.job_specs.c.name, name, [script]

    def __init__(
        self,
        cache: Cache | None = None,
        sync: GitignoreFilter | None = None,
        root: Path | None = None,
    ) -> None:
        """Defaults: the store under the root, a filter rooted there, the root found upward."""
        self.sync = sync or GitignoreFilter(root or Project().workspace())
        self.root = self.sync.root
        self.cache = cache or Cache(Lake.at(self.root))
        self.project = Shipper._dispatched_from(self.root)

    def _present(self, plan: ExecutionPlan, required: Sequence[Sequence[str]]) -> list[str]:
        """The sync include paths present here, refusing a mirror with nothing to ship or a
        required group missing a file."""
        scope = plan.profile.sync
        if not scope.include:
            raise LookupError(
                f"nothing to sync to {plan.host!r}; declare [hosts.{plan.host}.sync].include "
                "(or [hosts.defaults.sync].include) before dispatching"
            )
        shipped = scope.shipped(self.root)
        include = [path for path in shipped if self.local(path).exists()]
        manifests = set(Project().manifests)
        if stale := [path for path in shipped if path not in include and path not in manifests]:
            logger.warning(
                "skipping {} stale sync include path(s) missing locally: {}",
                len(stale),
                ", ".join(stale),
            )
        if not include:
            raise LookupError(f"every sync include path for {plan.host!r} is missing locally")
        if incomplete := [
            list(group) for group in required if not all(self.local(p).is_file() for p in group)
        ]:
            raise LookupError(
                f"required path group(s) {incomplete} are incomplete; build them before "
                "dispatching"
            )
        return include

    def _protected_outputs(self, fetch: str, *, sources: Sequence[str] = ()) -> list[str]:
        """Download-only output paths, refusing any that overlaps explicitly shipped input.

        The workspace cache spans host aliases and keeps completed evidence. Paths are literal,
        so glob characters in a real filename never widen what they protect.
        """
        recorded = (run.fetch_path for run in self.cache.listed())
        outputs = []
        for path in sorted({path for path in (fetch, *recorded) if path}):
            relative = writable(path)
            if not relative or relative == ".":
                raise ValueError(
                    f"declared output must be a relative path below the workspace: {path!r}"
                )
            if any(
                PurePosixPath(source).is_relative_to(relative)
                or PurePosixPath(relative).is_relative_to(source)
                for source in sources
            ):
                raise ValueError(
                    f"explicit input overlaps declared output {relative!r}; "
                    "bind the selected data under a separate immutable input path"
                )
            outputs.append(relative)
        return outputs

    def _stage(self, name: str, content: bytes) -> str:
        """Atomically stage exact bytes and answer their workspace-relative path.

        The staged file is only what the mirror carries to the host; the record of it is the
        lake's, a job script in `job_specs` and a closure listing in `closures`, kept once per
        content-addressed name.
        """
        path = state_path(self.root) / "jobs" / name
        with GeneratedFiles(directory=path.parent).locked() as files:
            files.write(path, content)
        table, column, key, rows = Shipper._recorded(name, content)
        held = self.cache.session.rows(select(column).where(column == key).limit(1))
        if not held:
            self.cache.session.append(table, rows)
        return path.relative_to(self.root).as_posix()

    def agent(self, plan: ExecutionPlan, *, ssh: SshTransport | None = None) -> Agent:
        """The standard-library agent on `plan.host`, what every mirror and pin talks to.

        It runs on the host's own environment for `plan` under its workspace root, or before
        that exists, on the uv-managed CPython onboarding put there first.

        ssh: the policy a rental's endpoint rides; a declared host's alias answers for itself.
        """
        policy = ssh or SshTransport()
        python = Dialect().python(
            plan.prefix(plan.profile.root),
            host=plan.host,
            python=Lockfile(self.root).python_request(plan.profile.platform),
        )
        return Agent(SshLink(plan.host, policy), python=python, patience=policy.deadline)

    def image(
        self, plan: ExecutionPlan, shipment: Shipment, *, listing: str, shipped: Sequence[str]
    ) -> Image:
        """What this dispatch's snapshot copies: the closure listed, or the shipped mirror.

        listing: the staged closure listing, workspace-relative, empty for a command.
        shipped: the allowlist the mirror transfer carried, what a command's snapshot copies.
        """
        if listing:
            return Sealed(listing=listing, needs=shipment.needs, pins=shipment.pins)
        return Mirrored(scope=self.scope(plan, shipped).spec())

    def local(self, path: str) -> Path:
        """`path` as a file here, a relative one resolved against the workspace root.

        The one place a written-down path becomes a local file, so a command typed in a
        subdirectory reads and writes what it would from the root.
        """
        given = Path(path).expanduser()
        return given if given.is_absolute() else self.root / given

    def mirror(
        self,
        plan: ExecutionPlan,
        root: str,
        *,
        ssh: SshTransport | None = None,
        required: Sequence[Sequence[str]] = (),
        fetch: str = "",
    ) -> list[str]:
        """Mirror the workspace to `plan.host`, returning the include paths present locally.

        The workspace and nested `.gitignore` files bound what is sent and deleted, and pruning
        reaches only the include paths, so a host file anywhere else is never touched. Output
        paths declared by this submission or any prior job in the workspace are download-only
        whatever their names or local existence; an explicit resource under one is refused
        before transfer (bind a separate immutable input instead). Ordinary `needs` stay mutable
        mirror links, not immutable snapshots, and `plan.profile.sync.protect` also protects
        unregistered remote artifacts. A path shipped by name that vanishes before the stream
        reaches it fails the whole mirror, so a submission never continues on a partial sync.

        Vendored path dependencies (see `engines.compile.vendor`) ship as their own scope with
        each link replaced by its target: here that tree is links into the source, which a host
        could never follow, so it gets real files its compile leaves alone, and a distribution
        the manifest stopped declaring is pruned without reaching anything beside it.

        ssh: where the transfer lands. A declared host is its own alias in the user's ssh config;
            a machine rented for one job has none, so a policy bound to it names it instead.
        required: groups that must exist locally whole and ship together by name despite the
            allowlist and ignores: a compiled manifest, its lock and the state it was solved from,
            or the staged job script alone.
        """
        policy = ssh or SshTransport()
        include = self._present(plan, required)
        named = list(dict.fromkeys(path for group in required for path in group))
        self.sync.validate_sources((*named, *plan.profile.sync.include))
        with SyncLock(policy.endpoint or plan.host, self.sync.root):
            outputs = self._protected_outputs(fetch, sources=named)
            scopes = [self.scope(plan, include, hidden=outputs)]
            vendored = vendor_root(self.root)
            if self.local(vendored).is_dir():
                scopes.append(Scope([vendored], deny=patterns(_VENDOR_EXCLUDE), follow=True))
            Mirror(self.root, self.agent(plan, ssh=policy)).push(
                root,
                scopes=scopes,
                named=named,
                protected=patterns([*CARD_LEASES, *plan.profile.sync.protect], paths=outputs),
            )
        self.cache.hosts.synced(plan.host)
        return include

    def pinned(self, root: str, *, source: Source, prefix: str = "") -> str:
        """The snapshot of the mirror at `root` a job dispatched from `source` runs in.

        Path arithmetic alone, so the job script can be rendered before the connection that
        materialises the snapshot opens; the dispatch creates it from this very key.

        prefix: the built environment the snapshot activates, which names it too (`pin_key`).
        """
        return Snapshots(root).path(pin_key(source, prefix))

    def scope(
        self, plan: ExecutionPlan, roots: Sequence[str], *, hidden: Sequence[str] = ()
    ) -> Scope:
        """The tree a mirror ships from `roots`, pruned by every rule a host obeys.

        Each repository's file list decides it, then the permanent denylist, the host's excludes
        and the card leases prune it.

        hidden: declared output paths, literal, which only ever travel down.
        """
        return self.sync.scope(roots, deny=denied(plan.profile.sync.exclude, paths=hidden))

    def source(self, command: str = "", *, paths: Sequence[str] = ()) -> Source:
        """Fingerprint the mirror scope and explicit command files without version control."""
        tree = SourceTree(self.root)
        roots = [
            *(paths or [Project().manifest(self.root).name]),
            *(
                (self.root / token).relative_to(self.root).as_posix()
                for token in shlex.split(command)
                if (self.root / token).is_file() and (self.root / token).is_relative_to(self.root)
            ),
        ]
        # The directories are listed in one pass: one listing per root re-asked git about every
        # repository fourteen times a dispatch (36 s on Windows).
        directories = [path for path in roots if not (self.root / path).is_file()]
        files = [path for path in roots if (self.root / path).is_file()]
        files += tree.filter.files(directories) if directories else []
        return tree.seal(sorted(dict.fromkeys(files)))[0]

    def stage_listing(self, shipment: Shipment) -> str:
        """Stage `shipment`'s closure listing under the jobs directory, empty for a command.

        Content-addressed by the closure digest, so a wave off one closure stages one file. The
        mirror carries it beside the script, the pin copies exactly the files it names, and the
        job reads the same rows through `MB_CLOSURE`.
        """
        if not shipment.sealed:
            return ""
        SourceTree(self.root).archive(shipment.listing)
        return self._stage(shipment.listing_name, shipment.listing.encode())

    def write_job_script(self, spec: JobSpec, *, pbs: bool) -> str:
        """Render `spec` under the jobs directory and answer its workspace-relative path.

        Content-addressed, so repeated runs reuse the file rather than growing the directory.
        Workspace-relative because the mirror recreates it on the host and the scheduler runs it
        there.
        """
        text = spec.render(pbs=pbs).encode()
        return self._stage(f"job-{hashlib.sha256(text).hexdigest()}.sh", text)
