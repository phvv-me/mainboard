"""The vocabulary every record shares: one key names the same thing in an event, a Git trailer
and a view, so a session, a job or a workspace is never spelled two ways.

Nothing here mints an id. Each owner does (the harness its session, a scheduler its job, jj its
revision, sha256 its bytes) and mb only names them.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Annotated, Self

from patos import FrozenModel
from pydantic import StringConstraints

from .enums import Key

type Scalar = str | bool | int | float
type Attribute = Scalar | list[Scalar]
"""What an OpenTelemetry attribute holds: a scalar or an array of them."""

_TRAILER = re.compile(r"^(?P<token>[A-Za-z][A-Za-z0-9-]*):[ \t]*(?P<value>.*)$")
_CONTINUATION = re.compile(r"^[ \t]+\S")
_BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")


class Trailer(FrozenModel):
    """One `Mb-*` line of a commit message's trailer block.

    value: one line, neither empty nor padded with whitespace.
    """

    key: Key
    value: Annotated[str, StringConstraints(pattern=r"^\S(?:[^\r\n]*\S)?$")]

    @classmethod
    def append(cls, message: str, trailers: Iterable[Trailer]) -> str:
        """`message` with `trailers` ending its trailer block, in a new block when it has none.

        Trailers the message already carries (`Co-Authored-By`) stay as they are, as
        `git interpret-trailers` leaves them.
        """
        head, tail = cls._split(message)
        joint = "\n" if cls._is_block(head, tail=tail) else "\n\n"
        return f"{head}{tail}{joint}" + "\n".join(str(trailer) for trailer in trailers) + "\n"

    @classmethod
    def read(cls, message: str) -> tuple[Self, ...]:
        """Every `Mb-*` trailer of `message`, in order; other trailers are not mb's to read."""
        head, tail = cls._split(message)
        found: list[Self] = []
        for line in tail.split("\n") if cls._is_block(head, tail=tail) else []:
            if (matched := _TRAILER.match(line)) and (key := Key.parsed(matched["token"])):
                found.append(cls(key=key, value=matched["value"].strip()))
        return tuple(found)

    @staticmethod
    def _is_block(head: str, *, tail: str) -> bool:
        """Whether `tail` is a trailer block: not the subject, and only `Token: value` lines.

        A line may instead continue the one before it. This is stricter than Git's 25 percent
        rule, so prose is never read as trailers.
        """
        lines = tail.split("\n")
        return bool(head) and all(_TRAILER.match(ln) or _CONTINUATION.match(ln) for ln in lines)

    @staticmethod
    def _split(message: str) -> tuple[str, str]:
        """`message` as everything up to its last paragraph and that paragraph."""
        body = message.rstrip("\n")
        blanks = list(_BLANK.finditer(body))
        cut = blanks[-1].end() if blanks else 0
        return body[:cut], body[cut:]

    def __str__(self) -> str:
        return f"{self.key.trailer}: {self.value}"


class Identity(FrozenModel):
    """The identities of one record, each empty when the record does not carry it.

    Fields are the `Key` members by name, so `Key[field]` is a field's attribute key.
    """

    host: str = ""
    job: str = ""
    object: str = ""
    previous_session: str = ""
    revision: str = ""
    session: str = ""
    workspace: str = ""

    @classmethod
    def from_attributes(cls, attributes: Mapping[str, Attribute]) -> Self:
        """The identity an event's attributes carry; keys that are not a `Key` are skipped."""
        return cls.model_validate(
            {Key(name).name: value for name, value in attributes.items() if name in Key}
        )

    @classmethod
    def from_trailers(cls, message: str) -> Self:
        """The identity a commit message's trailers carry, the first of a repeated key."""
        found: dict[str, str] = {}
        for trailer in Trailer.read(message):
            found.setdefault(trailer.key.name, trailer.value)
        return cls.model_validate(found)

    def attributes(self) -> dict[Key, str]:
        """The identities set, by key, in the attribute spelling."""
        return {Key[name]: value for name, value in self if value}

    def trailers(self) -> tuple[Trailer, ...]:
        """The identities set, as trailers in `Key` order."""
        return tuple(Trailer(key=key, value=value) for key, value in self.attributes().items())
