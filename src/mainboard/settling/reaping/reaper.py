"""The reap pass: a rental this center created and nothing here owns any more, ended.

A provider keeps billing a machine after its run settled, after a crash between the create and
the record, and after a release that failed once and was never retried. Only a label proves a
machine ours: the unique `mainboard-<uuid>` an `Allocation` records before the create and sends
with it. `decide` is the whole policy, pure over the provider's listing, the creations the run
registry holds and the holds; the pass only lists, applies and logs. Anything the labels cannot
prove is reported and left alone.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Final, Self

from filelock import Timeout
from patos import FrozenModel
from pydantic import AwareDatetime, Field

from ...core.errors import MissionError
from ...dispatch.backends import Inventory, ProviderBackend, Rented
from ...dispatch.hosts.allocation import Allocation
from ...dispatch.jobs import vocabulary
from ...log import logger
from ...state.storage.holds import Holdings
from .constants import GRACE

if TYPE_CHECKING:
    from ...dispatch.state import Cache, RunRecord
    from ...fleet.board import Board

# The status every cloud reports for a machine already ended.
_GONE: Final = "gone"
# What a provider listing, a cancel or a malformed answer raises; the pass logs it and goes on.
_FAULTS: Final = (MissionError, OSError, ValueError, LookupError)


class Action(StrEnum):
    """What the pass does with one listed machine."""

    BIND = auto()
    END = auto()
    KEEP = auto()
    REPORT = auto()


class Stage(StrEnum):
    """How far a creation got, and what is still owed for it."""

    PREPARED = auto()
    SUBMITTING = auto()
    LIVE = auto()
    OWED = auto()
    DONE = auto()


type Ruling = tuple[Action, str]


def _first(rules: Iterable[tuple[bool, Action, str]]) -> Ruling | None:
    """The action and reason of the first rule that holds, None when none does."""
    return next(((action, why) for holds, action, why in rules if holds), None)


class Kept(FrozenModel):
    """One creation this center minted, as the run registry holds it.

    label: the unique label the create was sent with, which a provider echoes back.
    provider: the backend kind it was created through.
    handle: the provider's id once bound, else the label itself.
    release_by: the lease deadline, None when the record carries no lease.
    """

    label: str = Field(min_length=1)
    target: str
    provider: str
    handle: str
    stage: Stage
    submitted: AwareDatetime
    release_by: AwareDatetime | None = None

    @classmethod
    def of(cls, record: RunRecord) -> Self | None:
        """The creation `record` retains, None for a run no provider was asked to create."""
        if not record.creation:
            return None
        return cls(
            label=record.creation,
            target=record.target,
            provider=record.kind,
            handle=record.handle,
            stage=cls._staged(record),
            submitted=datetime.fromisoformat(record.submitted_at),
            release_by=record.lease.release_by if record.lease else None,
        )

    @staticmethod
    def _staged(record: RunRecord) -> Stage:
        """Where `record` stands: unsent, possibly sent, running, awaiting its report, or done."""
        verdict = record.verdict or ""
        if verdict == vocabulary.Word.PREPARED:
            return Stage.PREPARED
        if verdict == vocabulary.Word.SUBMITTING:
            return Stage.SUBMITTING
        if verdict not in vocabulary.Word.settled():
            return Stage.LIVE
        return Stage.OWED if record.reported != record.verdict else Stage.DONE

    def expired(self, now: datetime) -> bool:
        """Whether its lease ran out by more than the grace `Deadlines` is given to end it."""
        return self.release_by is not None and now - self.release_by > GRACE

    def ruling(self, rented: Rented, *, provider: str, now: datetime) -> Ruling:
        """What to do with `rented`, a machine listed by `provider` under this creation's label.

        Ending or binding needs the record itself to say the machine is unowned or unrecorded,
        in the stage and handle it expects; any mismatch is reported instead.
        """
        submitting = self.stage is Stage.SUBMITTING
        return _first(
            (
                (
                    self.provider != provider,
                    Action.REPORT,
                    f"label {self.label} was minted for {self.provider}",
                ),
                (
                    self.stage is Stage.PREPARED,
                    Action.REPORT,
                    "its record says no create was ever sent",
                ),
                (
                    submitting and now - self.submitted < GRACE,
                    Action.KEEP,
                    "its create may still be in flight",
                ),
                (
                    submitting,
                    Action.BIND,
                    "a create with no recorded handle, bound to this machine",
                ),
                (
                    self.handle != rented.handle,
                    Action.REPORT,
                    f"its label is recorded against handle {self.handle}",
                ),
                (
                    self.stage is Stage.DONE,
                    Action.END,
                    "its run settled and was reported, so nothing here releases it",
                ),
                (self.expired(now), Action.END, f"past its release deadline {self.release_by}"),
                (
                    self.stage is Stage.OWED,
                    Action.KEEP,
                    "its run settled and settlement still owes its report",
                ),
            )
        ) or (Action.KEEP, "a live run owns it")


class Decision(FrozenModel):
    """The ruling on one listed machine, with the creation that justified it."""

    provider: str
    rented: Rented
    action: Action
    reason: str
    kept: Kept | None = None

    @property
    def selected(self) -> bool:
        """Whether the pass changes anything for this machine."""
        return self.action in {Action.BIND, Action.END}

    def announce(self, *, delete: bool) -> None:
        """Log the ruling: a machine left alone as a warning, a selected one as plan or deed."""
        machine = f"{self.provider} {self.rented.handle}"
        if self.action is Action.REPORT:
            logger.warning("{} left alone: {}", machine, self.reason)
        elif self.selected:
            logger.info(
                "{} {}{}: {}", machine, "" if delete else "would ", self.action, self.reason
            )


def decide(
    listing: Sequence[Rented],
    *,
    provider: str,
    kept: Sequence[Kept],
    held: Mapping[str, str],
    now: datetime,
) -> list[Decision]:
    """The ruling on every machine `provider` lists; one is ended or bound only if proven ours.

    Ours means its label is one a creation through this provider recorded here. Held, already
    gone, unlabeled, foreign and ambiguous machines are never ended or bound, a label listed
    twice at once being ambiguous: nothing proves which of its machines the create made.

    listing: the provider's rentals.
    kept: every creation the run registry holds, of any provider.
    held: the alias of each held machine by its provider handle.
    """
    minted = {creation.label: creation for creation in kept}
    labels = Counter(rented.label for rented in listing if rented.label)
    return [
        _judged(
            rented,
            minted.get(rented.label),
            provider=provider,
            spared=_spared(rented, held) or _repeated(rented, labels),
            now=now,
        )
        for rented in listing
    ]


def _judged(
    rented: Rented,
    creation: Kept | None,
    *,
    provider: str,
    spared: Ruling | None,
    now: datetime,
) -> Decision:
    """The ruling on one machine: `spared` when it holds, else its creation's."""
    action, reason = spared or (
        creation.ruling(rented, provider=provider, now=now) if creation else _stranger(rented)
    )
    return Decision(provider=provider, rented=rented, action=action, reason=reason, kept=creation)


def _repeated(rented: Rented, labels: Mapping[str, int]) -> Ruling | None:
    """Why `rented` is reported when its label is listed more than once, however it was minted."""
    if (count := labels[rented.label]) > 1:
        return Action.REPORT, f"label {rented.label} is listed {count} times, so it proves nothing"
    return None


def _spared(rented: Rented, held: Mapping[str, str]) -> Ruling | None:
    """Why `rented` is kept whatever its label says: a session holds it, or it already ended."""
    return _first(
        (
            (rented.handle in held, Action.KEEP, f"held as {held.get(rented.handle)}"),
            (rented.status == _GONE, Action.KEEP, "the provider reports it already ended"),
        )
    )


def _stranger(rented: Rented) -> Ruling:
    """Why a machine no creation here claims is reported, never ended."""
    if not rented.label:
        return Action.REPORT, "it carries no label, so nothing proves it ours"
    return Action.REPORT, f"label {rented.label} was not minted here"


class Reaped(FrozenModel):
    """What one reap pass found and did.

    delete: whether it applied its rulings; a dry run only lists and decides.
    acted: the handles it bound or ended.
    failed: the handles it tried to bind or end and could not, with why.
    unreached: the providers it could not list, with why, which it leaves alone.
    busy: another process held the settlement lock, so it did nothing.
    """

    delete: bool
    decisions: list[Decision] = Field(default_factory=list)
    acted: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
    unreached: dict[str, str] = Field(default_factory=dict)
    busy: bool = False


class Reaper:
    """The orphan sweep over every provider that lists the machines of its account.

    It runs on its own period beside the settler, whose `Deadlines` end the leases a record
    still tracks; the reaper finds what no record tracks. It takes the settlement lock without
    waiting, so it never overlaps a launch, a cancel or a settle.
    """

    def __init__(
        self, board: Board, *, providers: Sequence[ProviderBackend] | None = None
    ) -> None:
        """providers: the backends to ask, every registered one when None."""
        self.board = board
        self.cache: Cache = board.dispatcher.cache
        self.providers = (
            [backend() for backend in ProviderBackend.implementations()]
            if providers is None
            else list(providers)
        )

    def _applied(self, decisions: Sequence[Decision]) -> tuple[list[str], dict[str, str]]:
        """The handles of the selected machines that went through, and those that did not."""
        attempts = [
            (each.rented.handle, self._attempt(each)) for each in decisions if each.selected
        ]
        return (
            [handle for handle, why in attempts if not why],
            {handle: why for handle, why in attempts if why},
        )

    def _apply(self, decision: Decision) -> None:
        """Bind the machine to its creation, or ask its provider to end it."""
        if decision.action is Action.BIND:
            self._bind(decision)
        else:
            self._backend(decision.provider).cancel(decision.rented.handle)

    def _attempt(self, decision: Decision) -> str:
        """Apply one ruling, answering why it failed, empty when it went through."""
        try:
            self._apply(decision)
        except _FAULTS as fault:
            logger.error(
                "{} {} {} failed: {}",
                decision.provider,
                decision.rented.handle,
                decision.action,
                fault,
            )
            return str(fault)
        return ""

    def _backend(self, provider: str) -> ProviderBackend:
        """The backend named `provider`, which listed the machine being ruled on."""
        return next(backend for backend in self.providers if backend.name == provider)

    def _bind(self, decision: Decision) -> None:
        """Record the machine's handle against the creation that sent its label."""
        if decision.kept is None:
            raise LookupError("a bind needs the creation it binds")
        record = self.cache.creations.named(decision.kept.label, target=decision.kept.target)
        Allocation(cache=self.cache, record=record).created(decision.rented.handle)

    def _decided(self, listings: Mapping[str, Sequence[Rented]]) -> list[Decision]:
        """The ruling on every listed machine, against the registry and holds read after it.

        Listing first means every machine listed was created before the registry is read, so it
        already has its `SUBMITTING` record.
        """
        kept = [creation for run in self.cache.listed() if (creation := Kept.of(run)) is not None]
        held, now = self._held(), datetime.now(UTC)
        return [
            decision
            for provider, listing in listings.items()
            for decision in decide(listing, provider=provider, kept=kept, held=held, now=now)
        ]

    def _held(self) -> dict[str, str]:
        """The alias of each machine this workspace holds, by its provider handle."""
        root = self.board.root
        home = self.board.manifest.workspace.lake_home(root)
        return {held.handle: alias for alias, held in Holdings(root, home=home).read().items()}

    def _listed(self) -> tuple[dict[str, list[Rented]], dict[str, str]]:
        """Each provider's rentals, and why each provider that could not list is left alone."""
        listings: dict[str, list[Rented]] = {}
        unreached: dict[str, str] = {}
        for backend in self.providers:
            if not isinstance(backend, Inventory):
                continue
            try:
                listings[backend.name] = backend.rentals()
            except _FAULTS as fault:
                logger.debug("{} cannot list its rentals: {}", backend.name, fault)
                unreached[backend.name] = str(fault)
        return listings, unreached

    def _swept(self, *, delete: bool) -> Reaped:
        """List, decide, say what each ruling is, and apply the selected ones when `delete`."""
        self.cache.settlement.phase("reaping orphaned rentals")
        listings, unreached = self._listed()
        decisions = self._decided(listings)
        for decision in decisions:
            decision.announce(delete=delete)
        acted, failed = self._applied(decisions) if delete else ([], {})
        return Reaped(
            delete=delete, decisions=decisions, acted=acted, failed=failed, unreached=unreached
        )

    def once(self, *, delete: bool) -> Reaped:
        """One pass under the settlement lock.

        delete: apply the rulings; a dry run lists, decides and logs, and changes nothing.
        """
        lock = self.cache.settlement
        try:
            lock.acquire(timeout=0)
        except Timeout:
            logger.debug("another process owns settlement; the reap pass leaves it")
            return Reaped(delete=delete, busy=True)
        try:
            return self._swept(delete=delete)
        finally:
            lock.release()
