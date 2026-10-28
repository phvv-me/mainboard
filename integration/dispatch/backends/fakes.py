import json
from urllib.error import HTTPError
from urllib.request import Request


class Recorded:
    """A transport answering each (method, path suffix) from a table, keeping every request."""

    def __call__(self, request: Request) -> bytes:
        body = json.loads(request.data) if isinstance(request.data, bytes) else {}
        self.sent.append((request.get_method(), request.full_url, body))
        for (method, suffix), answer in self.answers.items():
            if request.get_method() == method and request.full_url.endswith(suffix):
                if isinstance(answer, HTTPError):
                    raise answer
                return json.dumps(answer).encode()
        raise AssertionError(f"unexpected {request.get_method()} {request.full_url}")

    def __init__(self, answers: dict[tuple[str, str], object]) -> None:
        self.answers = answers
        self.sent: list[tuple[str, str, dict]] = []
