"""What a card must look like before an acquisition measures on it, declared per card name."""

from typing import Literal

from ...core.base import Declared


class Admission(Declared):
    """The idle standard one card is held to, keyed in `[admission.<card name>]`.

    utilization_pct: the utilization a card must stay below to count as idle; a desktop card
        whose compositor holds it near a quarter declares what it can reach.
    holders: what other compute processes on the card mean: `refuse` the acquisition, or
        `record` them beside it because the card is shared and the study says so.
    """

    utilization_pct: int = 10
    holders: Literal["refuse", "record"] = "refuse"
