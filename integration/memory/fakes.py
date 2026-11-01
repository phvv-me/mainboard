from collections.abc import Collection

from mainboard.core import Refused, Unreachable
from mainboard.memory.keeps import Keep
from mainboard.memory.upstream import Remote, Upstream


class Recorder(Upstream):
    """An upstream that records what it is sent, answers `Unreachable` while it is down, and
    refuses the contents named in `refusing`."""

    @property
    def server(self) -> str:
        return "http://recorder.invalid/mcp"

    def __init__(self, *, down: bool = False, refusing: Collection[str] = ()) -> None:
        self.down = down
        self.refusing = refusing
        self.calls: list[tuple[str, str]] = []
        self.kept: list[Keep] = []

    def _reached(self) -> None:
        if self.down:
            raise Unreachable("the recorder is down")

    def find(self, question: str, *, budget: int | None = None) -> str:
        self._reached()
        self.calls.append(("find", question))
        return f"answer to {question}"

    def keep(self, keep: Keep) -> str:
        self._reached()
        if keep.content in self.refusing:
            raise Refused(f"the recorder refuses {keep.content}")
        self.calls.append(("keep", keep.content))
        self.kept.append(keep)
        return f"document-{len(self.kept)}"

    def status(self) -> Remote:
        self._reached()
        return Remote(login=True, caller="owner", finds=2, keeps=len(self.kept), processing="idle")
