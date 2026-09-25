# THE ENVIRONMENTS A DISPATCHED JOB ACTIVATES, AND WHY THEY ARE IMMUTABLE. A host used to hold
# one prefix per environment name, mutated in place by every `install`, `sync` and `pixi run`, so a
# job's code was frozen at dispatch but its environment was not. On 2026-09-05 that killed: one
# job of five when a later dispatch reconciled a primed wave's prefix (`Failed to update PyPI
# packages`); thirty two jobs when two checkouts sharing a host root crossed a manifest and a lock
# (`CXXABI_1.3.15 not found`, the half-reconciled prefix missing its libstdc++); and a forty five
# job wave when an agent solved a new lock between queuing and start.
#
# So an environment is content, not a location: its compiled files, second-stage declarations and
# host module stack name a directory built once and never written again. A change is a new digest
# beside the old, dispatch pins the digest like the source, and the job activates that prefix
# directly. THE PRICE IS DISK (gigabytes per CUDA lock), cheaper than a wave; `prune` bounds it.

import hashlib
import json
import shutil
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING

from ...core import MissionError, Project
from .backend import Pixi
from .compiler import Compiler
from .ecosystems import SecondStage
from .generated import ActivationScript, GeneratedFiles
from .pixi_lock import canonical
from .pixi_manifest import anchored, normalized, selected_manifest
from .provisioner import environment_shard
from .state import SyncState

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from ...manifest import Manifest
    from .generated import Writer

# Beside `envs/`, so nothing walking the mutable shards descends into a built prefix.
PREFIXES = "prefixes"

# Unreferenced prefixes surviving a prune: the running wave's and the next one's.
KEEP = 2

# A finished prefix's stamp naming its digest, so an interrupted build never passes as complete.
STAMP = ".mainboard-prefix"

ACTIVATION = "activate.sh"

# Pixi's required pair, the core of an environment's identity.
MANIFEST = "pixi.toml"
LOCK = "pixi.lock"


def prefix_path(root: str, environment: str, digest: str) -> str:
    """Where the environment `digest` (from `digest_of`) lives under `root`, on any machine.

    Path arithmetic alone, so a dispatch can pin it before the host has been asked anything.
    """
    return f"{root}/{Project().out_dir}/{PREFIXES}/{environment}/{digest}"


def digest_of(source: Path, *, modules: Mapping[str, str] = {}) -> str:
    """The identity of the environment `source`'s compiled artifact describes.

    Over Pixi's pair (missing either refuses), the generated inputs, the second-stage digest and
    the ordered host modules, never tasks, which each snapshot keeps. Everything is read
    `normalized` and the lock `canonical`, since pixi re-spells both. The root is the one the
    state recorded, since a host reads the artifact out of a pinned snapshot, not the workspace.

    source: a directory named after its environment, holding `pixi.toml` and `pixi.lock`.
    modules: the target host's declared module stack, in activation order.
    """
    shard = environment_shard(source.name)
    state = SyncState.load(source)
    # The machine's own path flavor; `normalized` matches its forward-slash spelling.
    root = PurePath(state.compiled_at or _standing(source, shard))
    payload = []
    for name in (MANIFEST, LOCK):
        text = normalized(_defining(source, name), root=root, generated_dir=shard)
        content = (
            canonical(text)
            if name == LOCK
            else json.dumps(Compiler.runtime_manifest(text), separators=(",", ":"))
        )
        payload.append((name, content))
    for entry in GeneratedFiles(directory=source).inputs:
        if entry.name in (MANIFEST, LOCK):
            continue
        payload.append(
            (
                entry.name,
                normalized(entry.read_text(encoding="utf-8"), root=root, generated_dir=shard),
            )
        )
    encoded = json.dumps(
        [payload, state.runtime_from, list(modules.items())], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _standing(source: Path, shard: PurePosixPath) -> str:
    """The root of an artifact older than `compiled_at`: where its shard hangs.

    Harmless when read elsewhere, since a root matching nothing rewrites nothing.
    """
    return source.parents[len(shard.parts) - 1].as_posix()


def _defining(source: Path, name: str) -> str:
    """One of the compiled artifact's defining files, refusing a half-written artifact."""
    try:
        return (source / name).read_text(encoding="utf-8")
    except OSError as missing:
        raise MissionError(
            f"{source / name} is missing, so the environment it describes has no identity; "
            f"run `{Project().name} install --resolve` where that artifact is built"
        ) from missing


class Prefixes:
    """One workspace's addressed environments: built once, never written to again, pruned late."""

    def __init__(self, root: Path, manifest: Manifest, environment: str = "default") -> None:
        self.root = root
        self.manifest = manifest
        self.environment = environment

    @property
    def base(self) -> Path:
        return Path(prefix_path(str(self.root), self.environment, "")).parent

    def path(self, digest: str) -> Path:
        """Where the environment `digest` names is built, whether or not it has been yet."""
        return Path(prefix_path(str(self.root), self.environment, digest))

    def built(self, digest: str) -> bool:
        """Whether the environment `digest` names is finished and safe to activate."""
        target = self.path(digest)
        try:
            stamped = (target / STAMP).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return False
        return stamped == digest and all(
            (target / name).is_file() for name in (MANIFEST, LOCK, ACTIVATION)
        )

    def materialize(self, source: Path, *, modules: Mapping[str, str] = {}) -> Path:
        """Build the environment `source` describes, once, and answer where it is.

        Idempotent, and under the workspace lock so a wave starting at once builds one prefix.
        The artifact is copied in first, so prefix and description live together. The prefix
        gets its own activation script (the mirror's names the mutable environment), and on
        Windows pixi's recorded activation. Nothing writes here after the stamp.

        modules: the host's declared module stack, carried into the activation script.
        """
        digest = digest_of(source, modules=modules)
        target = self.path(digest)
        projected = selected_manifest(self.manifest, self.environment)
        pixi = Pixi(target)
        stage = SecondStage(self.root, projected, target, pixi)
        if SyncState.load(source).runtime_from != stage.digest():
            raise MissionError(
                f"{source} does not record this selected second-stage runtime; recompile and "
                "ship the matching artifact before building its addressed environment"
            )
        if self.built(digest):
            return target
        SecondStage(self.root, projected, source, Pixi(source)).frozen_inputs(self.environment)
        self.base.mkdir(parents=True, exist_ok=True)
        with GeneratedFiles(directory=self.base).locked() as files:
            if self.built(digest):
                return target
            target.mkdir(parents=True, exist_ok=True)
            self.__take(source, target, files)
            pixi.install(self.environment)
            stage.install(self.environment)
            binaries = [
                directory
                for directory in stage.binary_dirs(self.environment)
                if directory.is_dir()
            ]
            ActivationScript(
                target / ACTIVATION, pixi.shell_hook(self.environment), binaries
            ).write(modules)
            pixi.cache_windows_activation(self.environment, binaries)
            files.write(target / STAMP, f"{digest}\n")
        return target

    def __take(self, source: Path, target: Path, files: Writer) -> None:
        """Copy every defining generated file and the state into the prefix, anchored.

        The loaders and second-stage inputs travel too, or activation and `[nodejs]` would go
        missing; dotted bookkeeping and installed trees stay. Anchored at this machine's root,
        which outlives every pruned pinned tree a shared prefix serves, so an editable follows
        the mirror rather than the snapshot. Only the copy is rewritten, never the digested source.
        """
        shard = environment_shard(self.environment)
        for entry in (*GeneratedFiles(directory=source).inputs, SyncState.path(source)):
            text = entry.read_text(encoding="utf-8")
            files.write(target / entry.name, anchored(text, root=self.root, generated_dir=shard))

    def referenced(self, sources: Path) -> set[str]:
        """Every environment digest a pinned tree under `sources` still activates.

        Read off the trees, which queued jobs actually name, not the run registry.
        """
        found: set[str] = set()
        for link in sources.glob(f"*/{Project().out_dir}/envs/*/.pixi"):
            try:
                resolved = link.readlink()
            except OSError:
                continue
            found.add(resolved.parent.name)
        return found

    def prune(self, *, live: Collection[str]) -> list[str]:
        """Remove and name every built environment nothing still names, the newest `KEEP` kept.

        live: digests a snapshot or unsettled run activates, never removed whatever their age.
        """
        try:
            found = sorted(
                (entry for entry in self.base.iterdir() if entry.is_dir()),
                key=lambda entry: entry.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []
        doomed = [entry for entry in found[KEEP:] if entry.name not in live]
        for entry in doomed:
            shutil.rmtree(entry, ignore_errors=True)
        return [entry.name for entry in doomed]
