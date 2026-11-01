class MissionError(RuntimeError):
    """A user-facing failure printed without a traceback at the CLI boundary."""


class CorruptFile(MissionError):
    """A file holds what its reader refuses, so the fix is to move it aside, never to repair it."""


class ExposedFile(MissionError):
    """A file only its owner may touch is open to others, and `chmod 600` is the whole fix."""


class Refused(MissionError):
    """A remote answered and refused what it was sent, so sending it again is refused again."""


class AgentRefused(MissionError):
    """The agent ran and declined, or failed, and said why on its last line."""


class Unavailable(MissionError):
    """A remote cannot be used on this machine: its client package or its login is missing."""


class Unreachable(MissionError):
    """A remote did not answer or answered with a server error, so what it was sent can wait."""


class NoWorkspace(FileNotFoundError):
    """No manifest from here upward: a command that needs a workspace was run outside one.

    A `FileNotFoundError` for the callers that already catch that, and printed like a
    `MissionError` at the CLI boundary, since the fix is where to run, never a traceback.
    """
