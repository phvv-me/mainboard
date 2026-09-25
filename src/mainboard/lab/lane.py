import itertools
import math
from typing import TYPE_CHECKING

from patos import FlexModel
from pydantic import ConfigDict

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Sequence


class Lane(FlexModel):
    """One counterbalanced condition an experiment measures each model or trial under.

    name: the label a trial's run id hashes and a report facets by. Extra keyword arguments
        become the lane's own arbitrary-typed fields (a warmup flag, a prompt variant, a fixture).
    """

    model_config = ConfigDict(extra="allow")

    name: str


def orders(lanes: Sequence[Lane], block: int) -> tuple[Lane, ...]:
    """The permutation of `lanes` for 0-based trial `block`, cycling all `len(lanes)!` before
    repeating so lane order is balanced across a study rather than fixed in declaration order."""
    permutations = tuple(itertools.permutations(lanes))
    return permutations[block % len(permutations)]


def validates(blocks: int, lanes: Sequence[Lane]) -> None:
    """Refuse a block count that is not a multiple of `len(lanes)!`, which would run some lane
    ordering more often than another."""
    cycle = math.factorial(len(lanes))
    if blocks % cycle != 0:
        raise MissionError(
            f"{blocks} blocks is not a multiple of {len(lanes)}! ({cycle}) lane permutations, "
            f"counterbalancing would be uneven"
        )
