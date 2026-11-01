"""The owner's grants: leave for one exact act, within one scope, until an expiry, nothing more.

A grant covers an act only when all three hold at once: the very act (a `push` grant never covers
a `launch`), a target its scope names (`owner/repo:mb/*` covers `owner/repo:mb/save`, never
`owner/repo:main`, `owner/repo:mb-x` or `owner/repo:mb/../main`), and a moment from when it was
granted up to, not including, its expiry. A scope is one target, or a target followed by a
separator and `*`, so no grant covers everything.

The grants of a workspace live in one 0600 file outside it, in the owner's configuration. Only
`serve grant` and `serve answer` write it, and both refuse unless a person at a terminal with no
harness marker types them. That refusal and the file's checks are guards, not a boundary: any
process of the owner's user can write the file, and `env -u <markers> script ...` passes the
refusal. The boundary is the harness sandbox (N1(a)), which keeps a sandboxed agent out of
`~/.config`, plus the agents' deny rules (`constants.DENIED`). Answers are kept here too, never in
the spool: an agent holding the sink's token could otherwise answer its own ask.

The file also holds the owner's card declarations: the limit of the virtual card paying a provider
that bills one after the fact, the only cap on that bill. It sits here rather than in mb.toml,
which only ever limits since an agent edits it, because a card's limit lifts one: without it such
a provider rents nothing.
"""

import os
import re
import stat
from collections.abc import Callable
from datetime import datetime
from errno import ELOOP
from pathlib import Path
from typing import Annotated, Self

from filelock import FileLock
from patos import FrozenModel
from pydantic import AfterValidator, AwareDatetime, Field, ValidationError, model_validator

from ..core import write_atomic
from ..core.errors import CorruptFile, ExposedFile, MissionError
from ..core.project import Project
from ..core.services.base import Managed
from ..enums import Act
from ..render import Terminal

# A target is words of letters, digits and `_.@+-`, joined by `/` or `:`, at most this long.
_WORDS = re.compile(r"[\w.@+-]+(?:[/:][\w.@+-]+)*", re.ASCII)
_SEPARATORS = re.compile(r"[/:]")
_LONGEST = 200


def said(fault: ValidationError) -> str:
    """The first reason `fault` gives, as its validator wrote it."""
    return fault.errors()[0]["msg"].removeprefix("Value error, ")


def _named(text: str) -> bool:
    """Whether `text` is a target: words joined by `/` or `:`, none of them `.` or `..`."""
    words = set(_SEPARATORS.split(text))
    return len(text) <= _LONGEST and bool(_WORDS.fullmatch(text)) and not {".", ".."} & words


def _target(text: str) -> str:
    """`text` as a target, refused unless `_named` holds."""
    if not _named(text):
        raise ValueError(
            f"{text!r} is no target: words of letters, digits and `_.@+-` joined by `/` or `:`, "
            f"no `.` or `..` among them, at most {_LONGEST} characters"
        )
    return text


def _scope(text: str) -> str:
    """`text` as a scope: a target, or a target, one separator and a trailing `*`."""
    named = text.removesuffix("*")
    if named == text:
        return _target(text)
    if not named.endswith(("/", ":")):
        raise ValueError(f"{text!r} is no scope: `*` only ends one, right after `/` or `:`")
    _target(named[:-1])
    return text


# What an act is done to: an `owner/repo:ref` for a push, a provider for a rent, a child for a
# launch.
type Target = Annotated[str, AfterValidator(_target)]
# What a grant covers: one target, or every target below a prefix (`owner/repo:mb/*`).
type Scope = Annotated[str, AfterValidator(_scope)]


class Grant(FrozenModel):
    """The owner's leave for one act within one scope, from `granted` up to `until`.

    ask: the ask it answers, empty for one the owner granted unasked.
    """

    act: Act
    scope: Scope
    granted: AwareDatetime
    until: AwareDatetime
    ask: str = ""

    @model_validator(mode="after")
    def _bounded(self) -> Self:
        """A grant ends after it begins."""
        if self.until <= self.granted:
            raise ValueError(f"a grant until {self.until} ends before it was granted")
        return self

    def covers(self, act: Act, target: str, *, at: datetime) -> bool:
        """Whether this grant lets `act` on `target` happen at `at`: act, scope and time all match.

        A `target` that is no target (`mb/../main`) is covered by nothing.
        """
        named = self.scope.removesuffix("*")
        within = target == self.scope or (named != self.scope and target.startswith(named))
        timely = self.granted <= at < self.until
        return act is self.act and _named(target) and within and timely


class Card(FrozenModel):
    """The owner's declaration of the limit of the card paying `provider`, from `declared` on.

    It has no expiry: the latest declaration for a provider holds until the owner makes another.
    """

    provider: Target
    usd: float = Field(gt=0, allow_inf_nan=False)
    declared: AwareDatetime


class Declined(FrozenModel):
    """The owner's no to one ask, which stays pending again only once it is asked anew."""

    ask: str
    at: AwareDatetime


class Grants(FrozenModel):
    """Every grant, no and card limit the owner wrote for a workspace, old ones kept as record."""

    cards: tuple[Card, ...] = ()
    declined: tuple[Declined, ...] = ()
    grants: tuple[Grant, ...] = ()

    def added(self, grant: Grant) -> Grants:
        """These grants with `grant` among them."""
        return self.model_copy(update={"grants": (*self.grants, grant)})

    def answered(self, ask: str) -> datetime | None:
        """When the owner last answered the ask `ask`, yes or no; None while never."""
        moments = [grant.granted for grant in self.grants if grant.ask == ask]
        moments += [declined.at for declined in self.declined if declined.ask == ask]
        return max(moments, default=None)

    def card(self, provider: str) -> float:
        """The limit the owner last declared for the card paying `provider`, 0 while none."""
        declared = (card for card in self.cards if card.provider == provider)
        latest = max(declared, key=lambda card: card.declared, default=None)
        return 0.0 if latest is None else latest.usd

    def covering(self, act: Act, target: str, *, at: datetime) -> Grant | None:
        """The first grant letting `act` on `target` happen at `at`, None when none does."""
        return next((grant for grant in self.grants if grant.covers(act, target, at=at)), None)

    def declaring(self, card: Card) -> Grants:
        """These answers with the owner's `card` declaration among them."""
        return self.model_copy(update={"cards": (*self.cards, card)})

    def declining(self, refusal: Declined) -> Grants:
        """These answers with the owner's `refusal` among them."""
        return self.model_copy(update={"declined": (*self.declined, refusal)})

    def live(self, *, at: datetime) -> list[Grant]:
        """The grants in force at `at`, soonest to end first."""
        live = (grant for grant in self.grants if grant.granted <= at < grant.until)
        return sorted(live, key=lambda grant: grant.until)

    def revoked(self, act: Act, scope: str, *, at: datetime) -> Grants:
        """These grants with every one of exactly `act` and `scope` in force at `at` ended then."""
        ended = tuple(
            grant.model_copy(update={"until": at})
            if grant.act is act and grant.scope == scope and grant.granted <= at < grant.until
            else grant
            for grant in self.grants
        )
        return self.model_copy(update={"grants": ended})


class OwnerFile:
    """A file under the owner's configuration that only their own verbs write: 0600, never a link.

    Its checks are a guard, not a boundary, since any process of the owner's user can write it.

    parts: where it sits below `~/.config/<package>/`.
    """

    def __init__(self, *parts: str) -> None:
        self.path = Path.home().joinpath(".config", Project().package, *parts)

    def _opened(self) -> int | None:
        """A descriptor on the file, None while there is none; refused when it is a link.

        It never blocks, so a FIFO opens at once and `text` refuses it as no regular file.
        """
        try:
            return os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        except PermissionError:
            raise MissionError(
                f"{self.path} is not readable by its owner; `chmod 600` it"
            ) from None
        except OSError as fault:
            if fault.errno != ELOOP:
                raise
            raise MissionError(f"{self.path} is a link, never the owner's own file") from None

    def text(self) -> str | None:
        """The file's text, None while it does not exist.

        Raises MissionError for a file that is a link, unreadable, another user's, or open to
        others, since anyone who could write such a file could loosen what it holds, and
        CorruptFile for one holding no UTF-8 text.
        """
        descriptor = self._opened()
        if descriptor is None:
            return None
        with os.fdopen(descriptor, "rb") as handle:
            status = os.fstat(handle.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
                raise MissionError(f"{self.path} is not the owner's own file")
            if status.st_mode & 0o077:
                raise ExposedFile(f"{self.path} is open to others; `chmod 600` it, then check it")
            content = handle.read()
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise CorruptFile(
                f"{self.path} holds no text; move it aside, then write it anew"
            ) from None


class GrantsFile(OwnerFile):
    """The one file holding a workspace's grants.

    root: the workspace the grants are for; its resolved path names the file, as the service
        manager's entries are named, so two checkouts never share leave.
    """

    def __init__(self, root: Path) -> None:
        super().__init__("grants", f"{Managed(root).stamp}.json")

    def changed(self, command: str, change: Callable[[Grants], Grants]) -> Grants:
        """Write `change` of the grants whole, under the file's lock; the owner's act alone.

        command: the words the owner typed it as, which the refusal names.

        Raises MissionError unless a person at a terminal, with no harness, runs it.
        """
        Terminal.current().refuse_agents(command)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with FileLock(self.path.with_suffix(".lock"), mode=0o600):
            grants = change(self.read())
            write_atomic(self.path, grants.model_dump_json(indent=2), mode=0o600)
        return grants

    def read(self) -> Grants:
        """The grants, none while the file does not exist.

        Raises MissionError for a file `text` refuses, and CorruptFile for one holding no grants.
        """
        text = self.text()
        if text is None:
            return Grants()
        try:
            return Grants.model_validate_json(text)
        except ValidationError as fault:
            raise CorruptFile(
                f"{self.path} is no grants file ({said(fault)}); move it aside, then grant again"
            ) from None
