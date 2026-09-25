from collections.abc import Iterator
from pathlib import Path

import pytest

from mainboard.nodes import evidence_of

_CARDS = "experiments/recovery_cost_cards/evidence"


@pytest.mark.parametrize(
    ("tree", "node", "expected"),
    [
        pytest.param(
            (_CARDS,),
            "recovery_cost_cards",
            _CARDS,
            id="the fetch path a --node dispatch was missing is a fact about the tree",
        ),
        pytest.param(
            ("experiments/fresh_claim",),
            "fresh_claim",
            "experiments/fresh_claim/evidence",
            id="a node that has never run still names where its evidence will land",
        ),
        pytest.param(
            (_CARDS, "datasets/experiments/recovery_cost_cards/evidence/receipts"),
            "recovery_cost_cards",
            _CARDS,
            id="the node's own directory wins over a store that mirrors it",
        ),
        pytest.param(
            ("experiments/something_else/evidence",),
            "recovery_cost_cards",
            "",
            id="a workspace with no such node names no path rather than inventing one",
        ),
        pytest.param((_CARDS,), "", "", id="a dispatch serving no node names no path"),
        pytest.param(
            (".mainboard/dispatch/sources/abc",),
            "sources",
            "",
            id="a pinned snapshot in the generated tree cannot answer for a node",
        ),
    ],
)
def test_a_node_names_its_evidence_directory_off_the_workspace_tree(
    tmp_path: Path, tree: tuple[str, ...], node: str, expected: str
) -> None:
    for path in tree:
        (tmp_path / path).mkdir(parents=True)
    assert evidence_of(tmp_path, node) == expected


def test_a_directory_the_walk_cannot_read_is_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another user's scratch or a stale mount beside the node must not cost the node its path."""
    (tmp_path / "locked/inner").mkdir(parents=True)
    (tmp_path / "experiments/fresh_claim").mkdir(parents=True)
    listed = Path.iterdir

    def guarded(directory: Path) -> Iterator[Path]:
        if directory.name == "locked":
            raise PermissionError(13, "Permission denied", str(directory))
        return listed(directory)

    monkeypatch.setattr(Path, "iterdir", guarded)

    assert evidence_of(tmp_path, "fresh_claim") == "experiments/fresh_claim/evidence"
