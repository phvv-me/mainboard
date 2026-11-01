"""What `mb memory keep` is given, and the key that makes a replay one keep.

An item is text, a URI or a regular file. A file that decodes as text is kept as that text, so a
note typed and the same note in a file are one keep; any other file is kept as the bytes it had
when it was given. A file is shown by its name, never by what it holds, and a file that may hold
secrets is never read: one named `.env` or `.env.*`, or one under a place the agent children's
`Read(...)` denies name, since a keep would carry it out past them. The key is a uuid5 of the
content and the scope, so keeping a thing twice names the same record and delivering it twice
adds nothing, whatever note came with it the second time.
"""

import os
import re
import stat
from hashlib import sha256
from pathlib import Path
from typing import Self
from uuid import UUID, uuid5

from patos import FrozenModel
from pydantic import AwareDatetime

from ..core.errors import MissionError
from ..core.events import Event
from ..policy.constants import ENFORCERS
from .enums import Kind, State

_NAMESPACE = UUID("6b30bd38-ef67-4d65-834e-cda9d593a2ab")
_EVENT = "mb.memory.keep"
_URI = re.compile(r"[a-z][a-z0-9+.-]*://")
# A word that names a place on a disk: it starts as a path does or has a slash and a suffix, so a
# slip in it is a missing file, where `TCP/IP` is only text.
_PLACE = re.compile(r"(?:[~/]|\.{1,2}/)\S*|\S+/\S*\.[A-Za-z0-9]{1,5}")
# The places a `Read(...)` deny names, and the `.env` variants beside the `.env` they name. A
# `./` place is in any folder, since a keep knows no workspace and a nested one is as secret.
_READ = re.compile(r"Read\((?P<place>[^)]+)\)")
_SECRETS = ("./.env.*",)
_LABEL = 48


class Keep(FrozenModel):
    """One thing to keep, as the spool holds it.

    kind: what the item is.
    content: the text itself, the URI, or a file's name.
    digest: sha256 of the text's or URI's UTF-8 bytes, or of the file's bytes.
    note: companion text that travels with the item.
    scope: the AIZK scope to keep it in, the caller's default when empty.
    source: where a file's bytes are, the file given first and the spool's copy of it at delivery.
    filename: the name of the file the item was read from, empty for a typed text or a URI.
    """

    kind: Kind
    content: str
    digest: str
    note: str = ""
    scope: str = ""
    source: Path | None = None
    filename: str = ""

    @classmethod
    def _text(cls, kind: Kind, *, content: str, note: str, scope: str, filename: str = "") -> Self:
        return cls(
            kind=kind,
            content=content,
            digest=sha256(content.encode()).hexdigest(),
            note=note,
            scope=scope,
            filename=filename,
        )

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """The keep a spool line holds."""
        return cls.model_validate_json(event.body)

    @classmethod
    def of(cls, item: str, *, note: str = "", scope: str = "") -> Self:
        """The keep of `item`: a URI, else a regular file it names, else the text it is.

        Raises MissionError for a folder, a file that may hold secrets, any other entry that is no
        regular file, and a word that names a place and no file.
        """
        if _URI.match(item):
            return cls._text(Kind.URI, content=item, note=note, scope=scope)
        path = Path(item).expanduser()
        try:
            data = cls._held(path)
        except OSError, ValueError:
            if _PLACE.fullmatch(item):
                raise MissionError(f"{item} is no file; keep a text by quoting it") from None
            return cls._text(Kind.TEXT, content=item, note=note, scope=scope)
        try:
            text = data.decode()
        except UnicodeDecodeError:
            return cls(
                kind=Kind.FILE,
                content=path.name,
                digest=sha256(data).hexdigest(),
                note=note,
                scope=scope,
                source=path.resolve(),
                filename=path.name,
            )
        return cls._text(Kind.TEXT, content=text, note=note, scope=scope, filename=path.name)

    @property
    def key(self) -> UUID:
        """The key of this content in this scope, the same wherever and whenever it is kept."""
        return uuid5(_NAMESPACE, f"{self.scope}\n{self.digest}")

    @property
    def label(self) -> str:
        """The item as one short line for rows: a file by its name, never by what it holds."""
        first = self.filename or self.content.strip().partition("\n")[0]
        return first if len(first) <= _LABEL else f"{first[: _LABEL - 1]}…"

    @staticmethod
    def _held(path: Path) -> bytes:
        """The bytes of the regular file at `path`, opened without waiting on a pipe.

        Raises OSError where nothing opens, and MissionError for what opens and is no file to keep.
        """
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            raise MissionError(f"{path} is a folder; keep its files by name")
        if deny := Keep._secret(path):
            raise MissionError(f"{path} may hold secrets ({deny}), so it is never kept")
        if not stat.S_ISREG(mode):
            raise MissionError(f"{path} is no regular file; keep a file or a text")
        # Non-blocking, so a file swapped for a pipe after the check reads empty, never waits.
        with open(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as handle:
            return handle.read()

    @staticmethod
    def _patterns(place: str) -> set[str]:
        """The absolute globs of a deny's place.

        A `~/` place is in this home however it resolves, and a `./` place in any folder.
        """
        match place[:2]:
            case "~/":
                return {f"{home}/{place[2:]}" for home in (Path.home(), Path.home().resolve())}
            case "./":
                return {f"**/{place[2:]}"}
            case _:
                return {place}

    @staticmethod
    def _secret(path: Path) -> str:
        """The place no agent child may read that holds `path` or the file it resolves to."""
        denied = [
            found["place"]
            for enforcer in ENFORCERS
            for deny in enforcer.denies
            if (found := _READ.fullmatch(deny))
        ]
        held = (path.absolute(), path.resolve())
        return next(
            (
                place
                for place in [*denied, *_SECRETS]
                if any(file.full_match(glob) for file in held for glob in Keep._patterns(place))
            ),
            "",
        )

    def event(self, at: AwareDatetime) -> Event:
        """The spool line of this keep: the key is the event's id, so a replay is one event."""
        return Event(name=_EVENT, at=at, body=self.model_dump_json(), id=self.key)


class Outcome(FrozenModel):
    """What one call did with one item.

    document: AIZK's id for what was kept, empty while it waits in the spool.
    """

    key: UUID
    item: str
    state: State
    document: str = ""
