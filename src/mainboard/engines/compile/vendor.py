# HOW A WORKSPACE DEPENDS ON A PACKAGE THAT LIVES OUTSIDE ITS ROOT.
#
# A path dependency inside the workspace is easy: the manifest spells it relative to the root,
# the compiled artifact spells it relative to the environment shard, the mirror carries the
# directory it names, and the same three files mean the same thing on every machine. A path that
# leaves the root is none of that. `../../packages/sample_lib` resolves on the workstation, where
# the workspace sits inside the monorepo; on Miyabi the mirror is
# `/work/xg25g007/x10537/reproducibility`, a SIBLING of the monorepo's own mirror, so the same
# spelling names a directory nothing ever put there, the lock records a location that exists on
# one machine in the world, and the digest taken over that lock spells that machine's tree.
#
# Two unpublished house packages are exactly this dependency and both were reached by hand until
# now: sample-lib commented out of the reproducibility manifest and its figures rendered
# under a hand-written `PYTHONPATH`, and atpx pinned back to the last published version while
# the ledger ran the unpublished one out of its source tree the same way. Neither hack survives
# a dispatch, because neither reaches a host at all.
#
# THE RULE, AND IT HAS NO KNOB: a path dependency that leaves the workspace root is vendored.
# Every one of them is declared to pixi at `<generated>/vendor/<distribution>` instead of where
# the manifest says it lives, one location, inside the root, the same distance from the root on
# every machine. The compiled manifest therefore spells `../../../.mainboard/vendor/sample_lib-
# tsukuba` wherever it is compiled, the lock pixi solves from it relativises back to that same
# spelling, and the environment digest taken over the pair is one number on the workstation and
# on the host. Nothing downstream needs to learn a new shape: `pixi_manifest.self_installed`
# already reports every editable path inside the root, so the vendored import roots reach a
# dispatched job's `PYTHONPATH` with the workspace's own, and `pixi_manifest.anchored` already
# rewrites a workspace-relative spelling into the machine's root on the way into a prefix.
#
# WHAT STANDS AT THAT LOCATION IS A REAL DIRECTORY WHOSE ENTRIES ARE SYMLINKS INTO THE SOURCE.
# Real, because a resolver handed a symlinked project root is free to record where it really
# went, and one canonicalised path would put the machine's own tree back into the lock the whole
# arrangement exists to keep machine-independent. Symlinked entries, because that is what keeps
# the editable install editable: an edit under `packages/sample_lib/src` is seen by the very next
# import, with nothing to re-vendor and nothing to reinstall, exactly as it was when the
# manifest named the source directly.
#
# A HOST HAS NO SOURCE TO LINK TO, and does not need one. The mirror carries the vendored tree
# with the referent of every link in place of the link (see `Dispatcher.rsync_up`), so what
# lands there is the ordinary directory of real files that the manifest and the lock already
# name, and a compile on the host leaves it exactly as the mirror left it. Vendoring is
# therefore a workstation act that a host inherits, and a workspace whose source is gone and
# whose vendored copy never arrived is refused by name rather than by pixi's report of a
# directory that is not a Python project.
#
# THE SOURCE IS NOT PART OF THE ENVIRONMENT'S IDENTITY. An editable install contributes
# dependency metadata to a prefix and no code, so a lock records a vendored distribution by name
# and location, never by a hash of its files: editing `sample_lib/style.py` cannot move a digest,
# cannot invalidate a lock, and cannot strand a queued wave. Editing that package's
# `pyproject.toml` moves `Compiler.resolution_digest`, which is the point, since that is the
# file a solve reads. That digest reads the vendored location too, so the number the workstation
# blessed a lock with is the number the host recomputes from the copy the mirror carried.

from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from ...core import MissionError, Project

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ...manifest import Manifest, Scope
    from .generated import Writer

# The generated subdirectory every vendored distribution is materialized in. Beside the
# environment shards rather than inside one, because a distribution is vendored for the
# workspace and several environments may declare the same one.
VENDOR = "vendor"


def vendor_root() -> str:
    """The workspace-relative directory every vendored path dependency is materialized in."""
    return f"{Project().out_dir}/{VENDOR}"


def outside(path: str) -> bool:
    """Whether a declared path dependency names something the workspace root does not contain.

    Pure arithmetic over the spelling, never a question about this machine's filesystem, so a
    compile answers it the same way on a host where neither location exists.

    path: the location a dependency spec declares, workspace-relative as the manifest writes it.
    """
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return True
    depth = 0
    for part in PurePosixPath(path).parts:
        depth += {".": 0, "..": -1}.get(part, 1)
        if depth < 0:
            return True
    return False


def relocated(name: str, path: str) -> str:
    """Where a compiled artifact spells `name`'s local source, vendoring it when it leaves.

    name: the distribution the spec was declared under, which names its vendored directory.
    path: the location the manifest declared, workspace-relative.
    """
    return f"{vendor_root()}/{name}" if outside(path) else path


def path_deps(manifest: Manifest) -> dict[str, str]:
    """Every Python path dependency `manifest` declares anywhere, distribution against location.

    Every scope, because a path dependency stands under the workspace, under `dev`, under a
    platform target and under a named environment's own tables, and a dependency declared for
    one platform is still a dependency.
    """
    declared: dict[str, str] = {}
    for scope in _scopes(manifest):
        python = scope.toolchains().get("python")
        if not python:
            continue
        for name, spec in python.all_deps().items():
            if spec.is_path and isinstance(path := (spec.model_extra or {})["path"], str):
                declared[name] = path
    return declared


def _scopes(manifest: Manifest) -> Iterator[Scope]:
    """Every scope of a manifest whose dependency tables a solve can reach."""
    yield from (manifest, manifest.dev, *manifest.on.values())
    for env in manifest.envs.values():
        yield from (env, *env.on.values())


class Vendor:
    """One workspace's copies of the path dependencies that live outside its root.

    root: the workspace root the vendored tree is generated under.
    manifest: the whole workspace manifest, never one environment's projection of it, so a
        compile of the serving environment never retires what the default one declares.
    """

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    @property
    def base(self) -> Path:
        """The directory every vendored distribution is materialized in."""
        return self.root / vendor_root()

    def roster(self) -> dict[str, str]:
        """Every distribution vendored here, against the declared location it comes from."""
        return {name: path for name, path in path_deps(self.manifest).items() if outside(path)}

    def refresh(self, files: Writer) -> None:
        """Bring the vendored tree in line with what the manifest declares, and nothing else.

        Runs inside `Compiler.write`, under the workspace lock and before the manifest that
        names these directories is written, so a solve never reads a location this compile was
        about to fill. Idempotent: an unchanged roster relinks nothing and removes nothing.
        """
        roster = self.roster()
        for name, declared in roster.items():
            self.__materialize(files, name, declared)
        for stray in self.__strays(roster):
            files.remove(stray)

    def __materialize(self, files: Writer, name: str, declared: str) -> None:
        """Point `name`'s vendored directory at the source, or keep the copy a mirror landed.

        A machine with the source rebuilds the directory of links from it, which is how a file
        added to or removed from the package reaches the vendored spelling. A machine without
        one keeps what it has, since that is the mirror's dereferenced copy and this compile has
        nothing better to say about it. A machine with neither is refused here, naming the
        location the manifest declared, rather than several layers down in pixi's report that
        some directory is not a Python project.
        """
        target = self.base / name
        source = self.root / declared
        if not source.is_dir():
            if target.is_dir():
                return
            raise MissionError(
                f"{name} is declared at {declared}, which is outside {self.root} and is not "
                f"there either, so there is nothing to vendor into {target}. Run "
                f"`{Project().name} install` where that source exists, or correct the path."
            )
        target.mkdir(parents=True, exist_ok=True)
        entries = [entry for entry in sorted(source.iterdir()) if not entry.name.startswith(".")]
        for entry in entries:
            files.link(target / entry.name, entry.resolve())
        kept = {entry.name for entry in entries}
        for gone in sorted(target.iterdir()):
            if gone.name not in kept:
                files.remove(gone)

    def __strays(self, roster: dict[str, str]) -> list[Path]:
        """Every vendored directory the manifest no longer declares anywhere."""
        try:
            entries = sorted(self.base.iterdir())
        except OSError:
            return []
        return [entry for entry in entries if entry.name not in roster]
