# THE ENVIRONMENTS A DISPATCHED JOB ACTIVATES, AND WHY THEY ARE IMMUTABLE.
#
# A host used to hold exactly one installed environment per name: `<root>/.mainboard/envs/<env>/
# .pixi/envs/<env>`, mutated in place by every `install`, every `sync`, and by pixi itself on the
# way into any command. Every pinned source tree symlinked its environment back to that one
# prefix, so a job's code was frozen at dispatch while the environment it would run in was not.
#
# A queued job therefore had no environment of its own. It got whatever the prefix happened to be
# at the moment it started, and three failures on 2026-09-05 were all that same fact:
#
#   * a wave primed at one commit whose prefix a later dispatch from another commit reconciled
#     under it, killing one job of five with `Failed to update PyPI packages`;
#   * two checkouts of one workspace sharing a host root, so a manifest from one landed over a
#     prefix installed from the other's lock and thirty two jobs died importing sqlite3 against a
#     half-reconciled prefix (`CXXABI_1.3.15 not found`, the loader falling through to the
#     system libstdc++ because the prefix's own was not there at that instant);
#   * a forty five job wave that died the same way after an agent solved a new lock in the same
#     workspace between its queuing and its start.
#
# So an environment is content now, not a location. The digest of the compiled manifest and the
# lock beside it names a directory that is built once and never written to again; a new lock is a
# new digest and a new directory beside the old one, and a job that was queued against the old
# one keeps it. Dispatch pins the digest into the snapshot the same way it pins the source, the
# job activates that prefix directly rather than asking pixi to reconcile anything, and the sweep
# that prunes unused source trees prunes unreferenced prefixes with them.
#
# THE PRICE IS DISK. Every distinct lock a host has been dispatched from keeps a full
# environment, which for a CUDA workspace is gigabytes. That is the trade: an environment that
# two waves fight over costs a whole wave, and a wave is worth more than a copy of an
# environment. `prune` is what keeps the bill finite, and it is the same pass that already keeps
# pinned trees finite.

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import MissionError, Project
from .backend import Pixi
from .ecosystems import SecondStage
from .generated import ActivationScript, GeneratedFiles
from .pixi_lock import canonical
from .pixi_manifest import anchored, normalized, selected_manifest
from .provisioner import environment_shard

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from ...manifest import Manifest
    from .generated import Writer

# Where a workspace keeps its addressed environments, beside the generated tree rather than
# inside `envs/`, so nothing that walks the mutable shards ever descends into a built prefix.
PREFIXES = "prefixes"

# How many unreferenced prefixes survive a prune. Two, because the one a running wave activates
# and the one the next wave was just dispatched against are both live for as long as the first
# wave lasts, and anything older than that pair is a bill nobody is paying for.
KEEP = 2

# What a finished prefix carries, naming the artifact it was built from. Its presence is what
# makes a second dispatch of the same lock free, and what keeps an interrupted build from being
# mistaken for a complete one.
STAMP = ".mainboard-prefix"

# What the activation a dispatched job sources is called inside a built prefix.
ACTIVATION = "activate.sh"

# The two files that define an environment, and therefore the ones its digest is taken over. The
# rest of the generated shard is copied in beside them (see `Prefixes.__take`) but not hashed:
# the state file names digests rather than dependencies, and an activation script or a second
# stage's own manifest cannot change which packages the lock says land here.
MANIFEST = "pixi.toml"
LOCK = "pixi.lock"


def prefix_path(root: str, environment: str, digest: str) -> str:
    """Where the environment `digest` names lives under `root`, on this machine or another.

    Path arithmetic alone, so a dispatch can pin a digest into a snapshot on a host it has not
    asked for anything yet, and the host arrives at the same directory when it builds.

    root: the workspace root the prefix lives under.
    environment: the logical environment being addressed.
    digest: the environment identity, from `digest_of`.
    """
    return f"{root}/{Project().out_dir}/{PREFIXES}/{environment}/{digest}"


def digest_of(source: Path) -> str:
    """The identity of the environment `source`'s compiled artifact describes.

    Over the compiled manifest and the lock beside it, because those two decide every package
    that lands in a prefix and nothing else does. A source missing either has no identity at
    all and says so, since building from half an artifact is how a prefix ends up describing
    one lock and containing another.

    Both files are read normalized, and the lock canonically on top of that, since pixi writes
    the lock and each version writes some of it differently: see `pixi_lock.canonical` for the
    platform relabelling that split one artifact into two addresses and killed a whole wave, and
    `pixi_manifest.normalized` for the local locations pixi respells, which split one artifact
    the same way the day a vendored dependency first named a directory under the root.

    The shard is named by the directory the artifact sits in, whose last segment is the
    environment it was compiled for and whose depth says which directory above it is the
    workspace root. Both are read off the artifact's own location rather than passed in, so
    every caller addresses one artifact the one way, including the host reading a snapshot's
    copy of it.

    source: a directory holding a compiled `pixi.toml` and the `pixi.lock` solved from it.
    """
    shard = environment_shard(source.name)
    root = source.parents[len(shard.parts) - 1]
    fingerprint = hashlib.sha256()
    for name in (MANIFEST, LOCK):
        text = normalized(_defining(source, name), root=root, generated_dir=shard)
        fingerprint.update((canonical(text) if name == LOCK else text).encode("utf-8"))
    return fingerprint.hexdigest()[:16]


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
    """One workspace's addressed environments: built once, never written to again, pruned late.

    root: the workspace root the prefixes live under.
    manifest: the workspace manifest, whose second-stage ecosystems install beside pixi's own.
    environment: the logical environment being addressed.
    """

    def __init__(self, root: Path, manifest: Manifest, environment: str = "default") -> None:
        self.root = root
        self.manifest = manifest
        self.environment = environment

    @property
    def base(self) -> Path:
        """The directory every addressed prefix for this environment lives in."""
        return Path(prefix_path(str(self.root), self.environment, "")).parent

    def path(self, digest: str) -> Path:
        """Where the environment `digest` names is built, whether or not it has been yet.

        Pure path arithmetic, so a dispatch can pin the digest into a snapshot before the
        machine that will build it has been asked for anything.

        digest: the environment identity, from `digest_of`.
        """
        return Path(prefix_path(str(self.root), self.environment, digest))

    def built(self, digest: str) -> bool:
        """Whether the environment `digest` names is finished and safe to activate."""
        return (self.path(digest) / STAMP).is_file()

    def materialize(self, source: Path, *, modules: Mapping[str, str] = {}) -> Path:
        """Build the environment `source` describes, once, and answer where it is.

        Idempotent by construction: an environment already built is answered without touching
        it, which is what makes this safe to call from every dispatch and from every job that
        finds its prefix missing. The build itself happens under the workspace lock, so a wave
        whose jobs all start at once builds one environment rather than nine.

        The artifact is copied in before pixi is asked for anything, so the prefix and the
        description it was built from live together and neither can be replaced without the
        other. Nothing after the stamp ever writes here again.

        The activation script is written here too, for this prefix and no other, because that is
        what a dispatched job sources: the mirror's own script names the mutable environment and
        would send every job back to the one thing this exists to stop sharing.

        source: the directory holding the compiled artifact to build from.
        modules: the host's declared module stack, carried into the activation script.
        """
        digest = digest_of(source)
        target = self.path(digest)
        if self.built(digest):
            return target
        self.base.mkdir(parents=True, exist_ok=True)
        with GeneratedFiles(directory=self.base).locked() as files:
            if self.built(digest):
                return target
            target.mkdir(parents=True, exist_ok=True)
            self.__take(source, target, files)
            pixi = Pixi(target)
            pixi.install(self.environment)
            projected = selected_manifest(self.manifest, self.environment)
            stage = SecondStage(self.root, projected, target, pixi)
            stage.install(self.environment)
            binaries = [
                directory
                for directory in stage.binary_dirs(self.environment)
                if directory.is_dir()
            ]
            ActivationScript(
                target / ACTIVATION, pixi.shell_hook(self.environment), binaries
            ).write(modules)
            files.write(target / STAMP, f"{digest}\n")
        return target

    def __take(self, source: Path, target: Path, files: Writer) -> None:
        """Copy the compiled artifact into the prefix, anchored so the copy reads as it did.

        EVERY generated file travels, not only the pair the digest is taken over. The manifest's
        activation sources the dotenv loader and the unset script by name, and the second stage
        installs from whatever its own managers read, so a prefix holding only the manifest and
        the lock is activated without the variables the workspace declares and installed without
        the packages a `[nodejs]` table asks for. What stays behind are the dotted entries: the
        sync lock and the mutable environment's own stamps, which describe that environment
        rather than this one, and the installed trees, which are directories and are what this
        build is about to create for itself.

        Each file is anchored on the way in, because a compile spells every declared location
        relative to the environment shard it was written for and a prefix is not that directory.
        The anchor is this machine's own workspace root, which is where those paths meant to
        point and, on a host, the one root that outlives any pinned tree: a prefix is addressed
        by content, so one of them serves every tree whose manifest and lock agree, while the
        trees themselves are pruned a few deep. An editable install anchored into a swept tree
        would be an import error in every later wave. The price is that such a package's source
        follows the mirror rather than the snapshot, which is the one thing about a dispatched
        job that a shared prefix cannot freeze.

        Only the copy is rewritten. The source keeps the bytes its digest was taken over, so the
        environment this prefix answers to is still addressed by content and not by where it
        happens to have been built.

        source: the directory holding the compiled artifact.
        target: the prefix being built from it.
        files: the writer holding the lock on the prefixes directory.
        """
        shard = environment_shard(self.environment)
        for entry in sorted(source.iterdir()):
            if entry.name.startswith(".") or not entry.is_file():
                continue
            text = entry.read_text(encoding="utf-8")
            files.write(target / entry.name, anchored(text, root=self.root, generated_dir=shard))

    def referenced(self, sources: Path) -> set[str]:
        """Every environment digest a pinned tree under `sources` still activates.

        Read off the trees themselves rather than from a record of what was dispatched, because
        the tree is what a queued job actually names: as long as one stands, the environment it
        points at is in use, whatever the run registry says about the jobs that own it.

        sources: the directory the host keeps its pinned trees in.
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
        """Remove every built environment nothing still names, newest `KEEP` kept, and name them.

        The other half of building one prefix per lock. `live` is every digest a snapshot or a
        run still owed an outcome activates, so what goes is what nothing can reach: an
        environment from a lock this workspace has moved past and no queued job remembers.

        live: the digests still referenced, never removed whatever their age.
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
