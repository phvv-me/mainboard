# THE LOCK IS THE ONE DEFINING FILE MAINBOARD DOES NOT WRITE. A prefix is addressed by the
# compiled manifest (byte-stable) and the lock, which each pixi version re-serializes differently.
# 2026-09-05: pixi 0.77 labelled a named platform variant `p1`, 0.79 `linux-64-system`, and both
# sort blocks by label; with no package, version or hash moved the workstation pinned
# 4950b228a3eaf208, the host built 4e0f0670076776b1, and the wave died with `mainboard found no
# built environment`. So the digest reads a canonical lock: a platform is the subdirectory it
# solves for plus the virtual packages it vouches, never its label or the order labels impose.
# Everything else is left as is, so a moved package, version or hash still moves the address.

import re
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The platform roster and one entry in it. Pixi writes `subdir` only when the label is not
# already the platform string.
_PLATFORMS = "platforms:"
_ENTRY = re.compile(r"- name: (\S+)")
_SUBDIR = re.compile(r"  subdir: (\S+)")

# The per-environment mapping from a platform label to its packages, the only other place a lock
# spells a label, and one of its keys.
_MAPPING = "    packages:"
_INDENT = "      "
_KEY = re.compile(rf"{_INDENT}(\S+):")

# One package a platform key lists (`- conda: <url>`, `- pypi: <url>`); deeper lines continue it.
_LOCATION = re.compile(rf"{_INDENT}- [a-z]+: (\S+)")

# A left-margin key closing the platform roster, unlike a flush `- name:` entry or a comment.
_TOP_LEVEL = re.compile(r"[A-Za-z_]")

# Separates a subdirectory from the rank of entries sharing it; only a digest reads this text.
_RANKED = "#"


def canonical(lock: str) -> str:
    """`lock` with every platform label content-derived and every labelled block in that order."""
    lines = lock.splitlines()
    naming = _naming(lines)
    if not naming:
        return lock
    return "\n".join(_relabelled(lines, naming)) + "\n"


def _naming(lines: Sequence[str]) -> dict[str, str]:
    """Every platform label in `lines` against its content-derived replacement.

    An entry is named after its subdirectory, ranked by its body (the virtual packages it
    vouches) when several share one.
    """
    naming: dict[str, str] = {}
    entries = sorted(_entries(lines), key=itemgetter(1))
    for subdir, sharing in groupby(entries, key=itemgetter(1)):
        ranked = sorted(sharing, key=itemgetter(2))
        for rank, (label, _, _) in enumerate(ranked):
            naming[label] = subdir if len(ranked) == 1 else f"{subdir}{_RANKED}{rank}"
    return naming


def _entries(lines: Sequence[str]) -> list[tuple[str, str, str]]:
    """Every platform the lock declares, as its label, the subdirectory it solves for, its body."""
    start, stop = _roster(lines)
    declared: list[tuple[str, list[str]]] = []
    for line in lines[start:stop]:
        if named := _ENTRY.fullmatch(line):
            declared.append((named[1], []))
        elif declared:
            declared[-1][1].append(line)
    return [(label, _subdir(label, body), "\n".join(body)) for label, body in declared]


def _subdir(label: str, body: Sequence[str]) -> str:
    """The subdirectory one entry solves for: the one it names, else its own label."""
    return next((named[1] for line in body if (named := _SUBDIR.fullmatch(line))), label)


def _roster(lines: Sequence[str]) -> tuple[int, int]:
    """Where the platform roster's entries begin and end, an empty span when there is none."""
    try:
        start = lines.index(_PLATFORMS) + 1
    except ValueError:
        return (0, 0)
    stop = start
    while stop < len(lines) and not _TOP_LEVEL.match(lines[stop]):
        stop += 1
    return (start, stop)


def _mappings(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Where each environment's platform-keyed packages mapping begins and ends."""
    spans: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if line != _MAPPING:
            continue
        stop = index + 1
        while stop < len(lines) and lines[stop].startswith(_INDENT):
            stop += 1
        spans.append((index + 1, stop))
    return spans


def _relabelled(lines: Sequence[str], naming: Mapping[str, str]) -> list[str]:
    """`lines` with every span a platform label introduces relabelled and put in that order."""
    headers = {_roster(lines): _ENTRY} | {span: _KEY for span in _mappings(lines)}
    rewritten = {
        start: (stop, _ordered(lines[start:stop], header, naming))
        for (start, stop), header in headers.items()
    }
    out: list[str] = []
    index = 0
    while index < len(lines):
        if index in rewritten:
            stop, body = rewritten[index]
            out += body
            index = stop
            continue
        out.append(lines[index])
        index += 1
    return out


def _ordered(body: Sequence[str], header: re.Pattern[str], naming: Mapping[str, str]) -> list[str]:
    """`body`'s labelled blocks, each renamed through `naming`, in canonical label order.

    A line naming no label continues the block above it; lines before the first label stay on top.
    """
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in body:
        named = header.fullmatch(line)
        label = naming.get(named[1], "") if named else ""
        if named and label:
            blocks.append((label, [f"{line[: named.start(1)]}{label}{line[named.end(1) :]}"]))
        else:
            blocks[-1][1].append(line)
    blocks.sort(key=itemgetter(0))
    return [line for _, block in blocks for line in block]


def packages(lock: str) -> dict[str, list[str]]:
    """Every package location the lock installs, by subdirectory (`win-64-system` as `win-64`)."""
    lines = lock.splitlines()
    subdirs = {label: subdir for label, subdir, _ in _entries(lines)}
    found: dict[str, list[str]] = {}
    for start, stop in _mappings(lines):
        label = ""
        for line in lines[start:stop]:
            if named := _KEY.fullmatch(line):
                label = named[1]
                found.setdefault(subdirs.get(label, label), [])
                continue
            location = _LOCATION.fullmatch(line)
            if label and location:
                found[subdirs.get(label, label)].append(location[1])
    return found
