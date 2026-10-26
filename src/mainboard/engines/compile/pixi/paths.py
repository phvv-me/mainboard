"""Workspace locations as a generated Pixi file spells them, and back.

A generated manifest sits in a state directory below the workspace, so every workspace-relative
location it names climbs back out of that directory; a file copied elsewhere, or digested across
machines, has those spellings resolved or made uniform here.
"""

import re
from pathlib import PurePath, PurePosixPath
from typing import TYPE_CHECKING

import tomlkit

from ..constants import GENERATED_DIR
from ..inputs.vendor import relocated

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from ..inputs.toml import Toml

# pixi tables whose values are dependency specs, where a `path` source lives.
_DEP_TABLES = ("dependencies", "pypi-dependencies", "dependency-overrides")

# One relative path token (leading parents and the segments under them), guarded against biting
# into a longer path (`/opt/a/../b`) or prose (`...`).
_RELATIVE = re.compile(r"(?<![\w./+~@-])\.\.(?:/[\w.+~@-]+)*(?![\w./+~@-])")


def rerooted(path: str, *, generated_dir: PurePath = GENERATED_DIR) -> str:
    """A workspace-relative path as a generated Pixi manifest must spell it.

    The parents are counted from `generated_dir`'s depth, never by hand. An absolute path rides
    through, and an empty one names the workspace root.
    """
    if PurePosixPath(path).is_absolute():
        return path
    parents = ("..",) * len(generated_dir.parts)
    root = PurePosixPath(*parents)
    return (root / PurePosixPath(path)).as_posix() if path else root.as_posix()


def anchored(text: str, *, root: PurePath, generated_dir: PurePath = GENERATED_DIR) -> str:
    """One generated file's text, with every workspace-relative spelling resolved against `root`.

    The inverse of `rerooted`, for a file copied away from where it was compiled (a prefix sits
    one directory deeper, so `"../../.."` meant the generated directory and pixi refused it).
    Textual, since the lock and dotenv loader spell the same locations. Every spelling landing
    inside the workspace is rewritten, not only the one `rerooted` writes: pixi 0.79 re-spelled
    `../../../.mainboard/vendor/patos` as `../../vendor/patos`, which escaped an exact-prefix match
    and left a Miyabi wave importing no torch (2026-09-06).

    generated_dir: the directory it was compiled into, whose depth decides which relative
        spellings reach the workspace.
    """
    return _rewritten(
        text,
        generated_dir=generated_dir,
        spell=lambda inside: (root / inside).as_posix() if inside else root.as_posix(),
    )


def normalized(text: str, *, root: PurePath, generated_dir: PurePath = GENERATED_DIR) -> str:
    """One generated file's text with every workspace location spelled one way, for a digest.

    pixi re-spells locations it was handed (as `lock.canonical` handles one level up), and
    `{{ config_root }}` renders each machine's own root into `[activation.env]`: the workstation
    pinned a4c06131efc5808c, the host read fc4975ef2096b9ac, and four Miyabi jobs died with no
    prefix built (2026-09-06). The root is written back out rather than activation stripped,
    so a real change to what a workspace exports still moves the address.

    root: the workspace root the file was compiled for, the one machine-specific spelling.
    """
    spelled = _unrooted(text, root=root, generated_dir=generated_dir)
    return _rewritten(
        spelled,
        generated_dir=generated_dir,
        spell=lambda inside: rerooted(inside, generated_dir=generated_dir),
    )


def _unrooted(text: str, *, root: PurePath, generated_dir: PurePath) -> str:
    """`text` with every absolute location under `root` written the way `rerooted` writes one."""
    here = re.escape(root.as_posix())
    anchored_at = re.compile(rf"(?<![\w./+~@-]){here}((?:/[\w.+~@-]+)*)(?![\w./+~@-])")
    return anchored_at.sub(
        lambda match: rerooted(match[1].lstrip("/"), generated_dir=generated_dir), text
    )


def _rewritten(text: str, *, generated_dir: PurePath, spell: Callable[[str], str]) -> str:
    """Every relative path token in `text` that reaches inside the workspace, respelled.

    A token climbing past the root is left as it stands: no mirror carries what it names.
    """

    def replace(match: re.Match[str]) -> str:
        inside = _inside(match[0], generated_dir=generated_dir)
        return match[0] if inside is None else spell(inside)

    return _RELATIVE.sub(replace, text)


def _inside(token: str, *, generated_dir: PurePath) -> str | None:
    """Where a relative token resolves under the workspace root (`""` for the root), else None.

    Pure arithmetic over the spelling; pathlib already drops every `.` component.
    """
    parts: list[str] = []
    for part in (*generated_dir.parts, *PurePosixPath(token).parts):
        if part != "..":
            parts.append(part)
        elif parts:
            parts.pop()
        else:
            return None
    return PurePosixPath(*parts).as_posix() if parts else ""


def self_installed(manifest: str, *, generated_dir: PurePath = GENERATED_DIR) -> list[str]:
    """Every workspace-relative directory a compiled manifest installs as an editable package.

    An editable's directory decides what a job imports, as when a workspace installs its own
    root (`path = "."`). Read from the compiled manifest the prefix is built and digested from,
    so the two rosters never differ. Paths outside the workspace never travel and are left out.

    manifest: the compiled `pixi.toml`'s text.
    """
    parents = rerooted("", generated_dir=generated_dir)
    declared = [
        path
        for spec in _source_specs(tomlkit.parse(manifest).unwrap())
        if spec.get("editable")
        if isinstance(path := spec.get("path"), str)
    ]
    inside = [path for path in declared if path == parents or path.startswith(f"{parents}/")]
    return list(dict.fromkeys(path.removeprefix(parents).lstrip("/") for path in inside))


def local_sources(manifest: str, *, generated_dir: PurePath = GENERATED_DIR) -> list[str]:
    """Workspace-relative dependency sources, including local wheel files."""
    parents = rerooted("", generated_dir=generated_dir)
    declared = [spec["path"] for spec in _source_specs(tomlkit.parse(manifest).unwrap())]
    inside = [
        path
        for path in declared
        if isinstance(path, str) and (path == parents or path.startswith(f"{parents}/"))
    ]
    return list(dict.fromkeys(path.removeprefix(parents).lstrip("/") for path in inside))


def _source_specs(value: Toml) -> Iterator[dict[str, Toml]]:
    """Every local dependency spec in the workspace, feature and platform-target tables."""
    items: Iterable[tuple[str | int, Toml]] = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, list)
        else ()
    )
    for key, item in items:
        if key in _DEP_TABLES and isinstance(item, dict):
            yield from (
                spec
                for spec in item.values()
                if isinstance(spec, dict) and isinstance(spec.get("path"), str)
            )
        else:
            yield from _source_specs(item)


def _reroot_source(name: str, spec: Toml, *, generated_dir: PurePath) -> Toml:
    """A dep spec with its local `path` source rerooted, at its vendored location if outside.

    See `vendor` for why a path leaving the root resolves nowhere on a host.
    """
    if isinstance(spec, dict) and isinstance(path := spec.get("path"), str):
        return {**spec, "path": rerooted(relocated(name, path=path), generated_dir=generated_dir)}
    return spec


def reparented(value: Toml, *, generated_dir: PurePath) -> Toml:
    """`value` with the `path` sources of its dependency tables rerooted.

    Only a dependency table's specs are read, so a dependency named `path` is kept.
    """
    if isinstance(value, dict):
        return {
            key: {
                name: _reroot_source(name, spec, generated_dir=generated_dir)
                for name, spec in item.items()
            }
            if key in _DEP_TABLES and isinstance(item, dict)
            else reparented(item, generated_dir=generated_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [reparented(item, generated_dir=generated_dir) for item in value]
    return value
