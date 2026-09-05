# WHERE A LEDGER NODE KEEPS ITS EVIDENCE, FOUND IN THE WORKSPACE RATHER THAN DECLARED IN IT.
#
# A dispatched job's receipts come home only if the dispatch declared a results path, and a
# dispatch that named the node it serves and nothing else got no path at all: the receipts sat in
# the pinned snapshot on the cluster and were rsynced back by hand hours later
# (`--node recovery_cost_cards`, no `--fetch`, miyabi-g 2026-09-05). The node already says where
# its evidence goes, though, because a node IS a directory and its evidence is the directory
# inside it, so the fetch path is a fact about the workspace rather than something a caller
# should have to retype per dispatch.
#
# So it is read off the tree. The node's slug names a directory somewhere under the workspace,
# and the evidence directory is the one inside it. The shallowest match wins, because a node's
# own directory sits nearer the root than any store that mirrors it: a workspace keeping a
# central copy under `datasets/experiments/<node>/evidence` writes its runs to
# `experiments/<node>/evidence`, and the run is what a dispatch pulls back.
#
# Nothing here declares a layout, which is the point: a workspace whose nodes are somewhere else
# entirely answers correctly as long as its node is a directory, and one that has no such
# directory answers empty and the dispatch says so rather than inventing a path.

from typing import TYPE_CHECKING

from .core.project import Project

if TYPE_CHECKING:
    from pathlib import Path

# What a node's evidence directory is called inside it.
EVIDENCE = "evidence"

# How far below the workspace root a node directory is looked for. Deep enough for the
# `experiments/<node>` and `datasets/experiments/<node>` shapes a program actually uses, shallow
# enough that the search never walks a data tree.
DEPTH = 3

# Directories a node is never looked for inside, beside every hidden one and the generated
# tree: a cache is not a claim, however many directories named like nodes it holds.
_SKIPPED = frozenset({"__pycache__", "node_modules"})


def evidence_of(root: Path, node: str) -> str:
    """Where `node` writes its evidence, workspace-relative, empty when the tree holds no such
    node.

    The answer names the evidence directory whether or not it exists yet, since the first run of
    a node is exactly the dispatch whose receipts most need to come home, and the directory the
    job will create is the one to pull.

    root: the workspace root the dispatch is staged from.
    node: the ledger slug the run serves.
    """
    if not node:
        return ""
    found = sorted(_candidates(root, node), key=lambda path: (len(path.parts), path.as_posix()))
    settled = [path for path in found if (path / EVIDENCE).is_dir()] or found
    if not settled:
        return ""
    return f"{settled[0].relative_to(root).as_posix()}/{EVIDENCE}"


def _candidates(root: Path, node: str, depth: int = DEPTH) -> list[Path]:
    """Every directory named `node` within `depth` of `root`, generated trees left out."""
    found: list[Path] = []
    frontier = [root]
    for _ in range(depth):
        walking: list[Path] = []
        for directory in frontier:
            for entry in _children(directory):
                (found if entry.name == node else walking).append(entry)
        frontier = walking
    return found


def _children(directory: Path) -> list[Path]:
    """The subdirectories of `directory` a node could be, in name order."""
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name not in _SKIPPED
        and entry.name != Project().out_dir
    ]
