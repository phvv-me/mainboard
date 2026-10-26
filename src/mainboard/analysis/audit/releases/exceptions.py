"""The exceptions of `releases` other modules share."""


class Unreachable(OSError):
    """The network did not answer and no cached answer exists."""
