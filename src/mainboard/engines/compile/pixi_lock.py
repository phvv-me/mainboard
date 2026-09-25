# THE LOCK IS THE ONE DEFINING FILE MAINBOARD DOES NOT WRITE.
#
# A prefix is addressed by the content of the compiled manifest and the lock beside it, and the
# manifest is this package's own output, byte-stable wherever it is compiled from the same
# workspace. The lock is pixi's, and pixi rewrites it: it reads the file, re-serializes what it
# understood, and every version spells some of that differently from the last one.
#
# 2026-09-05. The workstation solved with pixi 0.77, which labels a named platform variant `p1`,
# `p2`, ..., and Miyabi provisioned with pixi 0.79, which labels the same variant with the name
# the manifest gave it (`linux-64-system`). Both locks then sort their blocks by that label, so
# renaming reordered them as well. Not one package, version or hash moved. The workstation pinned
# 4950b228a3eaf208, the host built 4e0f0670076776b1, and every job of the wave died with
# `mainboard found no built environment`.
#
# So the digest is taken over a canonical lock, not over the bytes pixi last happened to write:
# a platform's identity in a lock is the subdirectory it solves for and the virtual packages it
# vouches, never the label chosen for it, and the order labelled blocks stand in is that label's
# accident. Everything else is left exactly as it is, so a lock that moved a package, a version
# or a hash still moves the address, which is the whole point of addressing one this way.

import re
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The top-level key introducing the platform roster, and the shape of one entry in it. Pixi
# writes `subdir` only when the label is not already the platform string, so an entry without
# one solves for the subdirectory its own label spells.
_PLATFORMS = "platforms:"
_ENTRY = re.compile(r"- name: (\S+)")
_SUBDIR = re.compile(r"  subdir: (\S+)")

# The per-environment mapping from a platform label to the packages that land on it, and the
# shape of one of its keys. Its entries are the only other place a lock spells a platform label.
_MAPPING = "    packages:"
_INDENT = "      "
_KEY = re.compile(rf"{_INDENT}(\S+):")

# One package a platform key lists, `- conda: <url>` or `- pypi: <url>`, at the key's own margin.
# Anything indented deeper, such as the `extras:` a PyPI entry carries, continues that entry.
_LOCATION = re.compile(rf"{_INDENT}- [a-z]+: (\S+)")

# A line that closes the platform roster: a key of its own at the left margin, which neither a
# sequence entry (`- name: ...`, written flush at that same margin) nor a comment is.
_TOP_LEVEL = re.compile(r"[A-Za-z_]")

# What separates a subdirectory from the rank telling two entries that solve for it apart. Only
# a digest ever reads this text, so it is free to spell a label in a way pixi never would.
_RANKED = "#"


def canonical(lock: str) -> str:
    """`lock` with every platform label content-derived and every labelled block in that order.

    The form an environment's digest is taken over. A rewrite that renames the platforms and
    reorders the blocks that name them lands on the same text, and a rewrite that changes which
    packages land anywhere does not.

    lock: the lock file's text, as pixi last wrote it.
    """
    lines = lock.splitlines()
    naming = _naming(lines)
    if not naming:
        return lock
    return "\n".join(_relabelled(lines, naming)) + "\n"


def _naming(lines: Sequence[str]) -> dict[str, str]:
    """Every platform label in `lines` against the content-derived label that replaces it.

    An entry is named after the subdirectory it solves for. Two entries may solve for one
    subdirectory under different floors, and then the virtual packages they vouch are the only
    thing that tells them apart, so each is ranked by the body it carries.
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
        named = _ENTRY.fullmatch(line)
        if named:
            declared.append((named[1], []))
        elif declared:
            declared[-1][1].append(line)
    return [(label, _subdir(label, body), "\n".join(body)) for label, body in declared]


def _subdir(label: str, body: Sequence[str]) -> str:
    """The subdirectory one entry solves for: the one it names, else its own label."""
    for line in body:
        named = _SUBDIR.fullmatch(line)
        if named:
            return named[1]
    return label


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

    A line naming no label continues the block above it, so a block travels whole and anything
    standing before the first label keeps its place at the top.
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
    """Every package location the lock installs, keyed by the subdirectory it lands on.

    The question a machine finding asks of a lock: does it hold anything at all for this
    machine's platform, and what were those builds compiled against. A labelled platform is read
    back to the subdirectory it solves for, so `win-64-system` answers as `win-64`.

    lock: the lock file's text.
    """
    lines = lock.splitlines()
    subdirs = {label: subdir for label, subdir, _ in _entries(lines)}
    found: dict[str, list[str]] = {}
    for start, stop in _mappings(lines):
        label = ""
        for line in lines[start:stop]:
            named = _KEY.fullmatch(line)
            if named:
                label = named[1]
                found.setdefault(subdirs.get(label, label), [])
                continue
            location = _LOCATION.fullmatch(line)
            if label and location:
                found[subdirs.get(label, label)].append(location[1])
    return found
