import tempfile
from collections.abc import Mapping
from pathlib import Path

from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.operations import MoveDefinitions

from ..engine.builders import written
from .builders import Move, row


def moved(sources: Mapping[str, str], *moves: Move) -> dict[str, str]:
    """`sources` once every move is made, the way the fixpoint makes them.

    Each pass renders the waiting moves over the files as they stand, and a move whose edits
    collide with one already taken waits for the next pass. A pass that takes none is an error.
    """
    files, waiting = dict(sources), list(moves)
    while waiting:
        with tempfile.TemporaryDirectory() as folder:
            plan = Plan(written(Path(folder), files))
            taken = [move for move in waiting if _is_taken(plan, files, move)]
            files |= {path: data.decode() for path, data in plan.written().items()}
        assert taken, f"no move was taken of {waiting}"
        waiting = [move for move in waiting if move not in taken]
    return files


def _is_taken(plan: Plan, files: Mapping[str, str], move: Move) -> bool:
    """Whether `plan` takes the edits of `move` over `files`."""
    edits = MoveDefinitions().edits(row(files, move))
    return edits is not None and plan.add(edits)
