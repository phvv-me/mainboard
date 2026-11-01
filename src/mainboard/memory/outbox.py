"""The spool of keeps that no pass has delivered, the receipts of those that were, and `Memory`,
the one front that spools a keep before it delivers it.

A keep waits as one spool line (`core.events`, event id = the keep's key) in
`.mb/memory/spool/<key>.jsonl`, with a copy of a binary file beside it as `<key>.blob`; delivering
it replaces the line by a receipt in `.mb/memory/kept/<key>.json`. It sits beside the telemetry
spool rather than in it, because the collect pass sends every line there to the sink and deletes
the file, and a keep is for the memory, not for telemetry. One pass delivers at a time, under a
lock beside the spool. An entry that is no keep, whose id is not its keep's key, or whose file's
copy is gone, and a keep the upstream refused, are set aside as `.refused` and never retried,
since skipping them would hide the bug that wrote them and retrying one would stop every pass.

A call that reaches the upstream first delivers what the spool holds, so a keep made while the
upstream was down is there before the next question; the drain pass delivers the rest through the
same `Memory.drain`.
"""

import shutil
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path
from uuid import UUID

from filelock import FileLock, Timeout
from patos import FrozenModel
from pydantic import AwareDatetime, ValidationError

from ..core import CorruptFile, MissionError, Refused, Unavailable, Unreachable, write_atomic
from ..core.events import EventSpool
from ..core.project import Project
from ..log import logger
from .enums import Kind, State
from .keeps import Keep, Outcome
from .upstream import Upstream

_EXTRA = "mainboard[memory]"
_LOCK = "drain.lock"
_PRIVATE = 0o600


class Receipt(FrozenModel):
    """A keep delivered: the upstream's id for it and when."""

    key: UUID
    document: str
    at: AwareDatetime


class Waiting(FrozenModel):
    """A keep in the spool, and when it was made."""

    keep: Keep
    since: AwareDatetime


class Outbox:
    """The spool and receipts of one workspace's keeps."""

    @contextmanager
    def delivering(self) -> Generator[bool]:
        """Hold the right to deliver; False, holding nothing, while another pass has it."""
        self.lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            held = FileLock(self.lock, timeout=0, mode=_PRIVATE).acquire()
        except Timeout:
            yield False
            return
        with held:
            yield True

    def __init__(self, root: Path) -> None:
        home = Project().out(root) / "memory"
        self.spool = home / "spool"
        self.kept = home / "kept"
        self.lock = home / _LOCK

    def _blob(self, key: UUID) -> Path:
        return self.spool / f"{key}.blob"

    def _copy(self, keep: Keep) -> None:
        """Copy the file `keep` names beside its line; refused if it is not the file digested."""
        if keep.source is None:
            raise MissionError(f"the file keep {keep.key} names no file")
        blob = self._blob(keep.key)
        shutil.copyfile(keep.source, blob)
        blob.chmod(_PRIVATE)
        with blob.open("rb") as handle:
            digest = file_digest(handle, "sha256").hexdigest()
        if digest != keep.digest:
            blob.unlink()
            raise MissionError(f"{keep.source} changed while it was being kept; keep it again")

    def _line(self, key: UUID) -> Path:
        return self.spool / f"{key}.jsonl"

    def _parsed(self, path: Path) -> Waiting | None:
        """The keep in `path`, None while its line is not written yet.

        Raises ValueError for an entry that holds more than the line of its own keep, or a file
        keep whose copy is gone.
        """
        events = EventSpool.read(path).events
        if not events:
            return None
        first, *rest = events
        keep = Keep.from_event(first)
        if rest or path.stem != str(first.id) or first.id != keep.key:
            raise ValueError("it holds more than the line of its own keep")
        if keep.kind is Kind.FILE:
            blob = self._blob(keep.key)
            if not blob.is_file():
                raise ValueError(f"the copy of {keep.label} beside it is gone")
            keep = keep.model_copy(update={"source": blob})
        return Waiting(keep=keep, since=first.at)

    def _read(self, path: Path) -> Waiting | None:
        """The keep in `path`, None for an entry set aside or one whose line is not written yet."""
        try:
            return self._parsed(path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as fault:
            self._refused(path, str(fault))
            return None

    def _refused(self, path: Path, why: str) -> None:
        """Set `path` aside as `.refused`, never to be read again."""
        logger.warning(f"{path} set aside, never retried: {why}")
        path.rename(path.with_suffix(".refused"))

    def append(self, keep: Keep) -> None:
        """Spool `keep` unless it is spooled or delivered already; a file is copied beside it.

        Raises MissionError for a file that changed while it was copied.
        """
        if self.receipt(keep.key):
            return
        self.spool.mkdir(mode=0o700, parents=True, exist_ok=True)
        with EventSpool(self._line(keep.key)) as spool:
            if spool.path.stat().st_size:
                return
            if keep.kind is Kind.FILE:
                self._copy(keep)
            spool.append(keep.event(datetime.now(UTC)))

    def receipt(self, key: UUID) -> Receipt | None:
        """The receipt of the keep `key`, None while it was never delivered.

        Raises CorruptFile for a receipt that does not read as one.
        """
        path = self.kept / f"{key}.json"
        try:
            held = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            return Receipt.model_validate_json(held)
        except ValidationError as fault:
            raise CorruptFile(f"{path} is no receipt: move it aside to keep again") from fault

    def refused(self, keep: Keep, why: str) -> None:
        """Set the spool line of `keep` aside, never to be delivered."""
        self._refused(self._line(keep.key), why)

    def settled(self, keep: Keep, document: str) -> None:
        """Record `keep` as delivered, then drop its spool line and copy."""
        self.kept.mkdir(mode=0o700, parents=True, exist_ok=True)
        receipt = Receipt(key=keep.key, document=document, at=datetime.now(UTC))
        write_atomic(self.kept / f"{keep.key}.json", receipt.model_dump_json(), mode=_PRIVATE)
        self._line(keep.key).unlink(missing_ok=True)
        self._blob(keep.key).unlink(missing_ok=True)

    def waiting(self) -> list[Waiting]:
        """The spooled keeps, oldest first; an entry that holds no keep is set aside, left out."""
        found = [self._read(path) for path in sorted(self.spool.glob("*.jsonl"))]
        return sorted(
            (w for w in found if w is not None), key=lambda w: (w.since, str(w.keep.key))
        )

    def waits(self, key: UUID) -> bool:
        """Whether the keep `key` is in the spool, neither delivered nor set aside."""
        return self._line(key).is_file()


class Memory:
    """The front of one workspace's memory: its spool in front of an upstream."""

    @staticmethod
    def aizk() -> Upstream:
        """AIZK's client as the upstream, refused by naming the extra that carries it if absent."""
        # Imported here alone: the module imports the client package at its top, and an absent
        # extra is answered by naming it, as `trials.lanes.adaptive.driver` does.
        try:
            from .aizk import Aizk
        except ImportError as missing:
            raise Unavailable(
                f"AIZK's client is an optional extra of this tool and is not installed here: "
                f"`pip install {_EXTRA}`"
            ) from missing
        return Aizk.open()

    def __init__(self, outbox: Outbox, upstream: Upstream) -> None:
        self.outbox = outbox
        self.upstream = upstream

    def _outcome(self, keep: Keep, *, known: bool) -> Outcome:
        receipt = self.outbox.receipt(keep.key)
        if known or receipt:
            state = State.KNOWN if known else State.KEPT
        else:
            state = State.QUEUED if self.outbox.waits(keep.key) else State.REFUSED
        return Outcome(
            key=keep.key,
            item=keep.label,
            state=state,
            document=receipt.document if receipt else "",
        )

    def drain(self) -> int:
        """Deliver the spooled keeps, oldest first; how many.

        Sets a keep the upstream refuses aside and goes on; stops at the first the upstream cannot
        be reached for, raising `Unreachable` with the rest still spooled; does nothing while
        another pass delivers.
        """
        delivered = 0
        with self.outbox.delivering() as free:
            if not free:
                logger.info("another pass is delivering the spool, so this one sends nothing")
                return 0
            for waiting in self.outbox.waiting():
                try:
                    document = self.upstream.keep(waiting.keep)
                except Refused as fault:
                    self.outbox.refused(waiting.keep, f"{waiting.keep.label}: {fault}")
                    continue
                self.outbox.settled(waiting.keep, document)
                delivered += 1
        return delivered

    def find(self, question: str, *, budget: int | None = None) -> str:
        """Answer `question` after delivering what the spool holds.

        A spool that cannot be delivered is said, and the question goes anyway.
        """
        try:
            self.drain()
        except Unreachable as fault:
            logger.warning(f"{fault}; the keeps wait in the spool, and the question goes anyway")
        return self.upstream.find(question, budget=budget)

    def keep(self, keeps: Sequence[Keep]) -> list[Outcome]:
        """Spool each of `keeps`, then try delivering the spool once; what each one came to.

        An upstream that does not answer leaves them queued, and the next call or the drain pass
        delivers them. A keep delivered earlier is known and spooled again never.
        """
        known = {keep.key for keep in keeps if self.outbox.receipt(keep.key)}
        for keep in keeps:
            self.outbox.append(keep)
        try:
            self.drain()
        except Unreachable as fault:
            logger.warning(f"{fault}; the keeps wait in the spool")
        return [self._outcome(keep, known=keep.key in known) for keep in keeps]
