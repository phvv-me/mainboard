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

    name: the lane's identity, the label a trial's run id hashes and a report facets by.
    Extra keyword arguments become the lane's own arbitrary-typed fields (a warmup flag, a
    prompt variant, a fixture object), exactly as declared at the call site.
    """

    model_config = ConfigDict(extra="allow")

    name: str


def orders(lanes: Sequence[Lane], block: int) -> tuple[Lane, ...]:
    """One counterbalanced ordering of `lanes` for trial block `block`.

    Declaring `lanes` on an experiment hands counterbalancing entirely to this function: every
    block cycles through all `len(lanes)!` permutations before repeating, so lane order is
    balanced across a study's trials instead of always running in declaration order.

    lanes: the lanes to order, in their declared sequence.
    block: the 0-based trial index selecting which permutation to run.
    """
    permutations = tuple(itertools.permutations(lanes))
    return permutations[block % len(permutations)]


def validates(blocks: int, lanes: Sequence[Lane]) -> None:
    """Refuse a block count that would not complete whole counterbalancing cycles.

    blocks: the planned number of trial blocks.
    lanes: the lanes each block orders through `orders`; `blocks` must be a multiple of
        `len(lanes)!` so no lane ordering runs more often than another across the study.
    """
    cycle = math.factorial(len(lanes))
    if blocks % cycle != 0:
        raise MissionError(
            f"{blocks} blocks is not a multiple of {len(lanes)}! ({cycle}) lane permutations, "
            f"counterbalancing would be uneven"
        )
