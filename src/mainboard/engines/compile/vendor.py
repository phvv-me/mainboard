# HOW A WORKSPACE DEPENDS ON A PACKAGE OUTSIDE ITS ROOT. `../../packages/sample_lib` resolves on
# the workstation, but on Miyabi the mirror (`/work/xg25g007/x10537/reproducibility`) is a
# sibling of the monorepo's, so the spelling names nothing there and the lock and digest spell
# one machine's tree. (Before this, sample-lib and atpx were reached by hand-written `PYTHONPATH`
# and version pins that never reached a host.)
#
# THE RULE, WITH NO KNOB: such a dependency is declared to pixi at `<generated>/vendor/<name>`,
# inside the root at the same distance on every machine, so the compiled manifest, the lock and
# the digest agree everywhere. `pixi_manifest.self_installed` and `anchored` already handle any
# editable path inside the root.
#
# THE VENDORED DIRECTORY IS REAL, ITS ENTRIES SYMLINKS INTO THE SOURCE: real, so a resolver cannot
# record the canonical machine path; symlinked, so the next import sees an edit with nothing to
# re-vendor. A host has no source: the mirror carries each link's referent (see
# `Dispatcher.mirror`), and a compile there leaves that copy alone, refusing by name when neither
# exists.
#
# THE SOURCE IS NOT PART OF THE ENVIRONMENT'S IDENTITY: an editable contributes metadata, not
# code, so editing its files moves no digest, while editing its `pyproject.toml` moves
# `Compiler.resolution_digest`, read at the vendored location on both machines alike.

from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from ...core import MissionError, Project

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ...manifest import Manifest, Scope
    from .generated import Writer

# Beside the environment shards, since several environments may declare one distribution.
VENDOR = "vendor"


def vendor_root() -> str:
    """The workspace-relative directory every vendored path dependency is materialized in."""
    return f"{Project().out_dir}/{VENDOR}"


def outside(path: str) -> bool:
    """Whether a declared workspace-relative path leaves the root, by spelling alone."""
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        return True
    depth = 0
    for part in PurePosixPath(path).parts:
        depth += {".": 0, "..": -1}.get(part, 1)
        if depth < 0:
            return True
    return False


def relocated(name: str, path: str) -> str:
    """Where a compiled artifact spells `name`'s local source, vendoring it when it leaves."""
    return f"{vendor_root()}/{name}" if outside(path) else path


def path_deps(manifest: Manifest) -> dict[str, str]:
    """Every Python path dependency `manifest` declares in any scope, name to location."""
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

    manifest: the whole manifest, never one environment's projection, so compiling one
        environment never retires what another declares.
    """

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    @property
    def base(self) -> Path:
        return self.root / vendor_root()

    def roster(self) -> dict[str, str]:
        """Every distribution vendored here, against the declared location it comes from."""
        return {name: path for name, path in path_deps(self.manifest).items() if outside(path)}

    def refresh(self, files: Writer) -> None:
        """Bring the vendored tree in line with the manifest, idempotently, under the sync lock."""
        roster = self.roster()
        for name, declared in roster.items():
            self.__materialize(files, name, declared)
        for stray in self.__strays(roster):
            files.remove(stray)

    def __materialize(self, files: Writer, name: str, declared: str) -> None:
        """Relink `name`'s vendored directory from the source, or keep the copy a mirror landed.

        With neither, refuse here by name rather than through pixi's "not a Python project".
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
