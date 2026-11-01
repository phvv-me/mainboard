"""A settlement pass keeps the center clean of what a killed collection left behind."""

from mainboard import Board
from mainboard.upkeep.disk.transient import Transient

from ...builders import survivors
from ...upkeep.disk.builders import streaming_into

# A process id no process has.
_DEAD = 999_999_999


def test_a_settlement_pass_clears_the_staging_a_killed_collection_left_behind(
    board: Board,
) -> None:
    left = Transient(board.root).place / f"collect-{_DEAD}-a"
    with streaming_into(left) as stream:
        board.monitor().once()

        assert (left.exists(), survivors([stream.pid])) == (False, [])
