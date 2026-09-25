from collections.abc import Iterator
from pathlib import Path

import pytest

from mainboard.nodes import evidence_of


def _tree(root: Path, *paths: str) -> None:
    """Create every directory `paths` names under `root`."""
    for path in paths:
        (root / path).mkdir(parents=True, exist_ok=True)


def test_a_node_names_its_own_evidence_directory(tmp_path: Path) -> None:
    """The fetch path a `--node` dispatch was missing is a fact about the tree, not a flag."""
    _tree(tmp_path, "experiments/recovery_cost_cards/evidence")

    assert evidence_of(tmp_path, "recovery_cost_cards") == (
        "experiments/recovery_cost_cards/evidence"
    )


def test_a_node_that_has_never_run_still_names_where_its_evidence_will_land(
    tmp_path: Path,
) -> None:
    """The first run of a node is exactly the dispatch whose receipts most need to come home."""
    _tree(tmp_path, "experiments/fresh_claim")

    assert evidence_of(tmp_path, "fresh_claim") == "experiments/fresh_claim/evidence"


def test_the_nodes_own_directory_wins_over_a_store_that_mirrors_it(tmp_path: Path) -> None:
    """A workspace keeping a central copy writes its runs to the node, and the run is what a
    dispatch pulls back."""
    _tree(
        tmp_path,
        "experiments/recovery_cost_cards/evidence",
        "datasets/experiments/recovery_cost_cards/evidence/receipts",
    )

    assert evidence_of(tmp_path, "recovery_cost_cards") == (
        "experiments/recovery_cost_cards/evidence"
    )


def test_a_workspace_with_no_such_node_names_no_path_at_all(tmp_path: Path) -> None:
    """Inventing one would send a pull at a directory nothing ever writes."""
    _tree(tmp_path, "experiments/something_else/evidence", ".mainboard/dispatch/sources/abc")

    assert evidence_of(tmp_path, "recovery_cost_cards") == ""
    assert evidence_of(tmp_path, "") == ""
    # The generated tree is not searched, so a pinned snapshot of this workspace cannot answer
    # for a node of it.
    assert evidence_of(tmp_path, "sources") == ""


def test_a_directory_the_walk_cannot_read_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another user's scratch or a stale mount beside the node must not cost the node its path."""
    _tree(tmp_path, "locked/inner", "experiments/fresh_claim")
    listed = Path.iterdir

    def guarded(directory: Path) -> Iterator[Path]:
        if directory.name == "locked":
            raise PermissionError(13, "Permission denied", str(directory))
        return listed(directory)

    monkeypatch.setattr(Path, "iterdir", guarded)

    assert evidence_of(tmp_path, "fresh_claim") == "experiments/fresh_claim/evidence"
