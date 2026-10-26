"""The one way the agent declines a request."""


class Refusal(Exception):
    """A request this end will not carry out, reported as one line rather than a traceback."""
