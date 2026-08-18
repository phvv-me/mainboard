from ...core.base import Declared


class Observe(Declared):
    """A host's live-observability posture: how eagerly its dispatched jobs get watched.

    level: `off` never spools, `poll` checks in every `poll-seconds`, `stream` keeps a live
        connection open, `interactive` additionally allows attaching a shell.
    channel: the named transport a poll or stream reads through, `auto` cascading `ssh-poll`
        then `stream`.
    poll_seconds: how often a poll channel re-checks the job; ignored under `stream` or
        `interactive`.
    """

    level: str = "poll"
    channel: str = "auto"
    poll_seconds: float = 30.0
