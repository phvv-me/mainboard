"""A group's hold on a range, which a plan keeps beside the edits it takes."""

from typing import Literal

from .edits import SpanEdit


class Claim(SpanEdit):
    """Hold `span` for this finding's group: a later group claiming an overlapping range waits a
    pass. It changes no byte; an operation claims what its edits were judged against (a block a
    deletion would empty), so two groups never act on one block together."""

    kind: Literal["claim"] = "claim"
