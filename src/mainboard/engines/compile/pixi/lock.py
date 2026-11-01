# THE LOCK IS THE ONE DEFINING FILE MAINBOARD DOES NOT WRITE. A prefix is addressed by the
# compiled manifest (byte-stable) and the lock, which each pixi version re-serializes differently.
# 2026-09-05: pixi 0.77 labelled a named platform variant `p1`, 0.79 `linux-64-system`, and both
# sort blocks by label; with no package, version or hash moved the workstation pinned
# 4950b228a3eaf208, the host built 4e0f0670076776b1, and the wave died with `mainboard found no
# built environment`. So the digest reads a canonical lock: a platform is the subdirectory it
# solves for plus the virtual packages it vouches, never its label or the order labels impose.
# Everything else is left as is, so a moved package, version or hash still moves the address.
#
# The address hashes the lock's text as pixi writes it (block style, no blank lines), so the text
# is edited line by line and the YAML parse only says which lines spell a label and where each
# run of labelled blocks ends.

from functools import lru_cache
from itertools import groupby, pairwise
from operator import itemgetter
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# The nodes spelling a run of labels, and the line the run ends at.
type _Run = tuple[list[yaml.Node], int]


@lru_cache(maxsize=4)
def canonical(lock: str) -> str:
    """`lock` with every platform label content-derived and every labelled block in that order."""
    lines = lock.splitlines()
    tree = _tree(lock)
    if not (platforms := _platforms(tree)):
        return lock
    roster: _Run = ([label for label, _ in platforms], tree["platforms"].end_mark.line)
    naming = _naming(platforms, _blocks(roster, lines))
    out: list[str] = []
    index = 0
    for run in sorted([roster, *_mappings(tree, naming)], key=itemgetter(1)):
        out += lines[index : run[0][0].start_mark.line] + _ordered(run, lines, naming)
        index = run[1]
    return "\n".join(out + lines[index:]) + "\n"


def _tree(lock: str) -> dict[str, yaml.Node]:
    """The lock's top-level nodes by key, each knowing its lines; none unless it is YAML."""
    try:
        return _fields(yaml.compose(lock, Loader=yaml.CSafeLoader))
    except yaml.YAMLError:
        return {}


def _fields(node: yaml.Node | None) -> dict[str, yaml.Node]:
    """A mapping node's values by key, none for any other node."""
    pairs = node.value if isinstance(node, yaml.MappingNode) else []
    return {key.value: value for key, value in pairs}


def _items(node: yaml.Node | None) -> list[yaml.Node]:
    """A sequence node's items, none for any other node."""
    return node.value if isinstance(node, yaml.SequenceNode) else []


def _platforms(tree: Mapping[str, yaml.Node]) -> list[tuple[yaml.Node, str]]:
    """Each platform the roster declares: the node spelling its label, its subdirectory.

    Pixi writes `subdir` only when the label is not already the platform string.
    """
    entries = [_fields(entry) for entry in _items(tree.get("platforms"))]
    return [(e["name"], e.get("subdir", e["name"]).value) for e in entries if "name" in e]


def _mappings(tree: Mapping[str, yaml.Node], naming: Mapping[str, str]) -> list[_Run]:
    """Each environment's packages mapping, as its keys naming a platform and where it ends."""
    runs: list[_Run] = []
    for environment in _fields(tree.get("environments")).values():
        if mapping := _fields(environment).get("packages"):
            keys = [key for key, _ in mapping.value if key.value in naming]
            runs += [(keys, mapping.end_mark.line)] if keys else []
    return runs


def _blocks(run: _Run, lines: Sequence[str]) -> list[Sequence[str]]:
    """The lines of each label's block: from the line spelling it to the line spelling the next."""
    labels, stop = run
    starts = [label.start_mark.line for label in labels] + [stop]
    return [lines[start:end] for start, end in pairwise(starts)]


def _naming(
    platforms: Sequence[tuple[yaml.Node, str]], blocks: Sequence[Sequence[str]]
) -> dict[str, str]:
    """Every platform label against its content-derived replacement `subdir` or `subdir#rank`.

    A platform is named after its subdirectory, ranked by its body (the virtual packages it
    vouches) when several share one.
    """
    naming: dict[str, str] = {}
    bodies = (
        (label.value, subdir, "\n".join(block[1:]))
        for (label, subdir), block in zip(platforms, blocks, strict=True)
    )
    for subdir, sharing in groupby(sorted(bodies, key=itemgetter(1)), key=itemgetter(1)):
        ranked = sorted(sharing, key=itemgetter(2))
        for rank, (label, _, _) in enumerate(ranked):
            naming[label] = subdir if len(ranked) == 1 else f"{subdir}#{rank}"
    return naming


def _ordered(run: _Run, lines: Sequence[str], naming: Mapping[str, str]) -> list[str]:
    """The run's blocks, each label renamed through `naming`, in canonical label order."""
    ordered: list[tuple[str, list[str]]] = []
    for label, block in zip(run[0], _blocks(run, lines), strict=True):
        name, start, end = naming[label.value], label.start_mark.column, label.end_mark.column
        ordered.append((name, [block[0][:start] + name + block[0][end:], *block[1:]]))
    return [line for _, block in sorted(ordered, key=itemgetter(0)) for line in block]


def packages(lock: str, *, environment: str = "") -> dict[str, list[str]]:
    """Every package location the lock installs, by subdirectory.

    `linux-64-system` counts as `linux-64`, and only `environment`'s locations count when it is
    named.
    """
    tree = _tree(lock)
    subdirs = {label.value: subdir for label, subdir in _platforms(tree)}
    found: dict[str, list[str]] = {}
    for name, body in _fields(tree.get("environments")).items():
        if environment in ("", name):
            for label, entries in _fields(_fields(body).get("packages")).items():
                found.setdefault(subdirs.get(label, label), []).extend(
                    location.value
                    for entry in _items(entries)
                    for location in _fields(entry).values()
                )
    return found
