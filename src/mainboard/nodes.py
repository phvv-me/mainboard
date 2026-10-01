# WHERE A LEDGER NODE KEEPS ITS EVIDENCE, FOUND IN THE WORKSPACE RATHER THAN DECLARED IN IT.
#
# A dispatch that named its node but no results path left the receipts in the pinned snapshot
# on the cluster, rsynced back by hand hours later (`--node recovery_cost_cards`, no `--fetch`,
# miyabi-g 2026-09-05). A node IS a directory and its evidence is the directory inside it, so
# the fetch path is a fact about the tree rather than something to retype per dispatch.
#
# The shallowest directory named for the node wins, because a node's own directory sits nearer
# the root than any store mirroring it: a workspace keeping a central copy under
# `datasets/experiments/<node>/evidence` writes its runs to `experiments/<node>/evidence`, and
# the run is what a dispatch pulls back. No layout is declared, so nodes anywhere answer, and a
# tree without the node answers empty rather than inventing a path. Evidence a lake keeps counts
# as on disk, so a node whose evidence left the tree still answers where it did.

from typing import TYPE_CHECKING

from .core.project import Project
from .state.evidence import EvidenceTree

if TYPE_CHECKING:
    from pathlib import Path

EVIDENCE = "evidence"

# Deep enough for the `experiments/<node>` and `datasets/experiments/<node>` shapes a program
# uses, shallow enough that the search never walks a data tree.
DEPTH = 3

# Never searched, beside every hidden directory and the generated tree: a cache is not a claim,
# however many directories named like nodes it holds.
_SKIPPED = frozenset({"__pycache__", "node_modules"})


def evidence_of(root: Path, node: str) -> str:
    """Where `node` writes its evidence, workspace-relative, empty when the tree has no such node.

    Named whether or not the evidence directory exists yet, since a node's first run is the
    dispatch whose receipts most need to come home, and the directory it creates is the one to
    pull.
    """
    if not node:
        return ""
    held = _kept(root, node)
    found = sorted(
        {*_candidates(root, node), *(path.parent for path in held)},
        key=lambda path: (len(path.parts), path.as_posix()),
    )
    settled = [path for path in found if (path / EVIDENCE).is_dir() or path / EVIDENCE in held]
    settled = settled or found
    if not settled:
        return ""
    return f"{settled[0].relative_to(root).as_posix()}/{EVIDENCE}"


def _kept(root: Path, node: str) -> set[Path]:
    """`node`'s evidence directories within `DEPTH` of `root` that a lake keeps, so a node whose
    evidence left the tree still settles where it did; generated trees left out."""
    tree = EvidenceTree(root)
    kept = {
        path.relative_to(tree.base)
        for depth in range(DEPTH)
        for path in tree.directories("/".join(["*"] * depth + [node, EVIDENCE]))
    }
    return {root / path for path in kept if not any(_hidden(part) for part in path.parts)}


def _hidden(name: str) -> bool:
    """Whether a directory named `name` is never a node's: hidden, a cache or generated."""
    return name.startswith(".") or name in _SKIPPED or name in Project().out_dirs


def _candidates(root: Path, node: str) -> list[Path]:
    """Every directory named `node` within `DEPTH` of `root`, generated trees left out."""
    found: list[Path] = []
    frontier = [root]
    for _ in range(DEPTH):
        children = [entry for directory in frontier for entry in _children(directory)]
        found += [entry for entry in children if entry.name == node]
        frontier = [entry for entry in children if entry.name != node]
    return found


def _children(directory: Path) -> list[Path]:
    """The subdirectories of `directory` a node could be, in name order; none if unreadable."""
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [entry for entry in entries if entry.is_dir() and not _hidden(entry.name)]
