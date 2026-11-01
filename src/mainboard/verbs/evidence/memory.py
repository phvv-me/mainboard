"""The `memory` verbs: find and keep in AIZK, with a keep that waits while AIZK is down."""

from pydantic import ConfigDict

from ...core.errors import MissionError, Unavailable
from ...log import logger
from ...memory.health import MemoryHealth
from ...memory.keeps import Keep
from ...memory.outbox import Memory, Outbox
from ...memory.upstream import Upstream
from ...render import Output
from ...render.values import visible
from ..constants import TREE
from ..session import Verbs

_COMPACT = Output()
_KEPT = ("key", "state", "document", "item")
_SHORT = 8


class MemoryVerbs(Verbs):
    """Ask the memory, put things in it, and see how it is.

    upstream: the memory to talk to, AIZK's client when None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    upstream: Upstream | None = None

    def _memory(self) -> Memory:
        return Memory(Outbox(self.session.workspace_root()), self._upstream())

    def _upstream(self) -> Upstream:
        return self.upstream or Memory.aizk()

    def find(self, question: str, *, budget: int | None = None, output: Output = _COMPACT) -> None:
        """Answer a question from memory, with its sources, after delivering what waits.

        Args:
            question: what to ask, in words.
            budget: the most tokens the answer may take, the server's default when unset.
        """
        answer = self._memory().find(question, budget=budget)
        if output.mode == "json":
            output.print_record({"answer": answer}, title="find")
        else:
            logger.result(visible(answer))

    def keep(
        self, *items: str, note: str = "", scope: str = "", output: Output = _COMPACT
    ) -> None:
        """Keep text, files or URIs in memory; they wait and replay while AIZK is down.

        Each item goes to the local spool first and is delivered once; whatever AIZK cannot take
        yet stays `queued` and is delivered by the next call that reaches AIZK or by the drain
        pass. The same item in the same scope keeps once however often it is given, with any note.

        Args:
            items: text, the path of a file, or a URI.
            note: companion text kept with each item, such as what an image shows.
            scope: the AIZK scope to keep them in, the caller's default when unset.
        """
        if not items:
            raise MissionError("name what to keep: text, the path of a file, or a URI")
        keeps = [Keep.of(item, note=note, scope=scope) for item in items]
        rows = [
            {**outcome.model_dump(mode="json"), "key": str(outcome.key)[:_SHORT]}
            for outcome in self._memory().keep(keeps)
        ]
        output.print_rows(rows, title="keep", columns=_KEPT)

    def status(self, *, output: Output = _COMPACT) -> None:
        """Show the upstream, the login, what waits in the spool and what AIZK counts.

        Delivers what waits first when AIZK answers; when it does not, says what is local and why
        the rest is missing.
        """
        outbox = Outbox(self.session.workspace_root())
        try:
            health = MemoryHealth.of(outbox, self._upstream())
        except Unavailable as fault:
            health = MemoryHealth.of(outbox, reason=str(fault))
        output.print_record(health.row(), title="memory")


_MEMORY = MemoryVerbs()
memory = TREE.node("memory").app(_MEMORY.find, _MEMORY.keep, _MEMORY.status)
