class MissionError(RuntimeError):
    """A user-facing failure printed without a traceback at the CLI boundary."""


class NoWorkspace(FileNotFoundError):
    """No manifest from here upward: a command that needs a workspace was run outside one.

    A `FileNotFoundError` for the callers that already catch that, and printed like a
    `MissionError` at the CLI boundary, since the fix is where to run, never a traceback.
    """
