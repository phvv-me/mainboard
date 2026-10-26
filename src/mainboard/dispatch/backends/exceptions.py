"""The exceptions of `backends` other modules share."""

from ...core.errors import MissionError


class CapacityGone(MissionError):
    """The offer chosen was taken or has no capacity left; another one may still rent."""
