"""The exceptions of `runtime` other modules share."""


class Refusal(Exception):
    """An environment that could not be entered, with the exit status the job ends on.

    message: what the job says about it, empty when the activation already said it.
    """

    def __init__(self, message: str, status: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
