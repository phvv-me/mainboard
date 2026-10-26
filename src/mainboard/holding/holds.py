# `mb host hold` and `mb host release`: a rented machine kept for a session and reached as
# an ordinary ssh host, so it is set up once and then takes any number of jobs in seconds each.
#
# A rental per job rebuilds the whole environment every time: of a 17.5 minute landing on
# 2026-09-21 the upload was 1.3 minutes and the environment nearly all the rest, for a job of 2.7.
# Holding by hand (an ssh block, a manifest host, a watchdog to destroy it) is steps a tired
# session forgets. Here the rental joins the run registry dispatched rentals live in, its
# deadline as the lease, so the durable sweep releases it on time whoever is watching, and
# `compute` releases whatever is past due before it lists.

import re
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ..core.errors import MissionError
from ..costs.catalog import Offer
from ..dispatch.backends.base import Rentable, route
from ..dispatch.dispatcher import Order
from ..dispatch.hosts.landing import renter
from ..dispatch.hosts.lease import Lease
from ..dispatch.hosts.rentals import handoff
from ..dispatch.jobs import vocabulary
from ..dispatch.mirroring.shipment import Shipment
from ..dispatch.shared import announce
from ..dispatch.transport.aliases import SshAliases
from ..dispatch.transport.core import SshTransport
from ..dispatch.wrapping import connection
from ..engines.compile.provisioner import Provisioner
from ..log import logger
from ..manifest.schema.compute.queue import Defaults
from ..runtime.job import walltime
from ..state.storage.holds import Held, Holdings

if TYPE_CHECKING:
    from ..context.plan import ExecutionPlan
    from ..dispatch.hosts.rentals import Rental
    from ..dispatch.jobs.vocabulary import Resources
    from ..dispatch.shared import Watcher
    from ..dispatch.state.cache import RunRecord
    from ..fleet.board import Board

_DURATION = re.compile(r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?$")

# The rented entrypoint gives up on a landing that never comes, and its exit would read as a
# finished job the sweep settles and releases, so a held machine runs this until released.
_IDLE = "exec sleep infinity"

# Every rented image this house lands on is a Linux container.
_PLATFORM = "linux-64"


@Parameter(name="*")
class Cards(FrozenModel):
    """The cards a held machine carries: a named one, or the cheapest of some capabilities."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    gpu_name: str = ""
    """the card to rent, in the provider's own spelling."""
    arch: str = ""
    """rent the cheapest card of these compute capabilities instead of a named one: `sm_120`
    (RTX 5090, RTX PRO 4500/6000), `sm_100` (B200), `sm_90+`, `hopper`."""
    gpus: int = 0
    """cards per machine, the provider profile's default when 0."""
    spot: bool = False
    """rent interruptible capacity, cheaper and taken back at the provider's will; a machine
    taken back reads as released."""


@Parameter(name="*")
class Holding(FrozenModel):
    """How long a held machine is kept, what it is called, and what it may cost."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    duration: Annotated[str, Parameter(name="--for")]
    """how long to keep it once it is ready, `3h`, `90m` or `1h30m`."""
    alias: Annotated[str, Parameter(name="--as")] = ""
    """the alias to reach it by, `PROVIDER-CARD` when omitted."""
    max_usd: float = 0.0
    """the spend cap over the whole hold, landing included, the provider's default when 0."""
    walltime: str = ""
    """on a PBS host, how long each allocation lasts, `HH:MM:SS`; the queue's ceiling when
    omitted."""

    @property
    def seconds(self) -> int:
        """How long the machine is kept once ready."""
        return duration_seconds(self.duration)

    def asked(self, cards: Cards) -> vocabulary.Resources:
        """What renting `cards` for this hold asks of the provider."""
        return vocabulary.Resources(
            walltime=walltime(self.seconds),
            gpus=cards.gpus,
            gpu_name=cards.gpu_name,
            max_usd=self.max_usd,
            spot=cards.spot,
            arch=cards.arch,
        )


class Holds:
    """The machines this workspace is holding: rented, set up, listed and released.

    aliases: the ssh config the aliases are written into, the user's own by default.
    """

    @staticmethod
    def _drafted(
        name: str, rental: Rental, record: RunRecord, *, plan: ExecutionPlan, seconds: int
    ) -> Held:
        """The hold as recorded before setup: an ssh host priced as its rental was."""
        offer = record.lease.offer if record.lease is not None else None
        profile = plan.profile.model_copy(
            update={
                "kind": "ssh",
                "platform": _PLATFORM,
                "defaults": Defaults(walltime=walltime(seconds)),
                "queues": {},
            }
        )
        return Held(
            alias=name,
            provider=plan.host,
            handle=rental.handle,
            gpu=offer.gpu if offer is not None else "",
            usd_hr=offer.rate_usd_hr if offer is not None else None,
            deadline=datetime.now(UTC) + timedelta(seconds=seconds),
            profile=profile,
        )

    def __init__(self, board: Board, *, aliases: SshAliases | None = None) -> None:
        self.board = board
        self.aliases = aliases or SshAliases()
        self.holdings = Holdings(board.root, home=board.manifest.workspace.lake_home(board.root))

    def _abandon(self, name: str, rental: Rental, provider: str) -> None:
        """End a rental whose setup failed, so it stops billing."""
        logger.warning("hold {} failed during setup; releasing rental {}", name, rental.handle)
        self._end(name, handle=rental.handle, provider=provider)

    def _end(self, alias: str, *, handle: str, provider: str) -> None:
        """Cancel the rental through the ordinary settle path, then forget its alias.

        A rental whose record is gone, a dispatch state wiped or moved, still bills, so it is
        ended straight through its provider rather than left running for want of a row.
        """
        try:
            self.board.dispatcher.cache.run(handle, provider)
        except LookupError:
            _rentable(self.board.on(provider).plan(container="none")).cancel(handle)
        else:
            self.board.verdicts().cancel(handle, host=provider)
        self.aliases.remove(alias)
        self.holdings.drop(alias)
        self._reload()
        self.board.dispatcher.cache.hosts.drop(alias)

    def _keep(
        self, name: str, rental: Rental, *, plan: ExecutionPlan, seconds: int, watch: Watcher
    ) -> Held:
        """Name the machine, set it up, park its entrypoint, and start the clock."""
        self.aliases.add(name, rental.endpoint)
        record = self.board.dispatcher.cache.run(rental.handle, plan.host)
        held = Holds._drafted(name, rental, record, plan=plan, seconds=seconds)
        self._record(held)
        setup = self.board.on(name).setup(plan.env, watch=watch)
        self._park(name, watch=watch)
        ready = held.model_copy(
            update={
                "deadline": datetime.now(UTC) + timedelta(seconds=seconds),
                "profile": held.profile.model_copy(update={"root": setup.root}),
            }
        )
        self._record(ready)
        self._relet(record, ready, kind=plan.profile.kind)
        return ready

    def _named(self, provider: str, holding: Holding, cards: Cards, plan: ExecutionPlan) -> str:
        """The alias the hold goes by: the one asked, else `<provider>-<card>` as an ssh alias.

        An alias already naming a host is refused.
        """
        card = cards.gpu_name or cards.arch.replace("+", "-up") or plan.profile.defaults.gpu_name
        # Lower case, and runs of anything else collapsed to one dash.
        name = holding.alias or re.sub(r"[^a-z0-9]+", "-", f"{provider}-{card}".casefold()).strip(
            "-"
        )
        if name in self.board.manifest.hosts:
            raise MissionError(f"{name!r} already names a host; hold it under another --as")
        return name

    def _park(self, name: str, *, watch: Watcher) -> None:
        """Hand the waiting entrypoint a command that never ends, so the machine stays up."""
        watch(f"parking {name} until it is released")
        with connection(name, SshTransport()) as remote:
            retcode, _, err = (remote["bash"]["-c", handoff()] << f"{_IDLE}\n").run(retcode=None)
        if retcode:
            raise MissionError(f"could not park {name}: {err.strip()[-400:]}")

    def _record(self, held: Held) -> None:
        """Write `held` down and let the workspace resolve its alias as a host from now on."""
        self.holdings.save(held)
        self._reload()

    def _relet(self, record: RunRecord, ready: Held, *, kind: str) -> None:
        """Move the rental's lease to the hold's deadline, so the sweep releases it then."""
        offer = record.lease.offer if record.lease is not None else None
        cache = self.board.dispatcher.cache
        cache.relet(
            cache.delivery(record, "pending"),
            Lease(
                offer=offer or Offer(provider=kind, gpu=ready.gpu, rate_usd_hr=0.0),
                release_by=ready.deadline,
            ),
        )

    def _reload(self) -> None:
        """Forget the loaded manifest, so the next read sees the holdings as they are now."""
        for derived in ("manifest", "resolver"):
            self.board.shared.pop(derived, None)

    def _rent(
        self,
        backend: Rentable,
        plan: ExecutionPlan,
        resources: Resources,
        *,
        name: str,
        duration: str,
    ) -> Rental:
        """Register the hold in the run registry, then rent the machine it describes."""
        shipment = Shipment.of_command(
            f"hold {name} for {duration}", source=self.board.dispatcher.source(), imports=()
        )
        allocation = self.board.dispatcher.allocating(
            Order(plan=plan, shipment=shipment, resources=resources, name=f"hold-{name}"),
            evidence="not_started",
        )
        with ExitStack() as undo:
            undo.callback(allocation.interrupted)
            rental = backend.rent(plan, resources, allocation=allocation)
            undo.pop_all()
        return rental

    def _settled(self, held: Held) -> bool:
        """Whether `held`'s record says the rental already ended, or no record is left."""
        try:
            record = self.board.dispatcher.cache.run(held.handle, held.provider)
        except LookupError:
            return True
        return record.verdict in vocabulary.Word.settled()

    def expire(self) -> list[Held]:
        """Release every held machine past its deadline or already ended, and name them.

        A machine the sweep released at its lease, or the provider took back, has a settled
        record; its alias and profile go here. A release the provider refuses stays held with a
        warning, so the next pass or the sweep's lease asks again and one refusal costs no other.
        """
        now = datetime.now(UTC)
        released: list[Held] = []
        for held in self.holdings.read().values():
            if held.deadline > now and not self._settled(held):
                continue
            try:
                released.append(self.release(held.alias))
            except (MissionError, OSError) as fault:
                logger.warning("could not release {} yet: {}", held.alias, fault)
        return released

    def hold(
        self,
        provider: str,
        holding: Holding,
        cards: Cards,
        *,
        env: str = "",
        watch: Watcher | None = None,
    ) -> Held:
        """Rent a machine through `provider`, set it up as host `alias`, and keep it until due.

        The deadline counts from the moment the machine is ready, since a hold is for the time it
        can take jobs; until then the rental's own lease, priced with the landing, bounds it.

        provider: the provider host to rent through, `vast` say.
        env: the environment to set up, the provider profile's own when empty.
        watch: announces each stage as it begins.
        """
        rented = self.board.on(provider)
        plan = rented.plan(env=env, container="none")
        name = self._named(provider, holding, cards, plan)
        resources = rented.resources(holding.asked(cards), plan=plan)
        Provisioner(self.board.root, self.board.manifest).compiler_for(plan.env).vouch()
        rental = self._rent(_rentable(plan), plan, resources, name=name, duration=holding.duration)
        with ExitStack() as undo:
            undo.callback(self._abandon, name, rental, provider)
            held = self._keep(
                name, rental, plan=plan, seconds=holding.seconds, watch=watch or announce
            )
            undo.pop_all()
        return held

    def release(self, alias: str) -> Held:
        """End the held machine `alias`, settle its record, and forget its alias."""
        try:
            held = self.holdings.read()[alias]
        except KeyError:
            raise MissionError(
                f"{alias!r} is not held; held machines are {sorted(self.holdings.read())}"
            ) from None
        self._end(alias, handle=held.handle, provider=held.provider)
        return held


def duration_seconds(duration: str) -> int:
    """`duration` as a person writes it (`3h`, `90m`, `1h30m` or `HH:MM:SS`) in seconds."""
    spoken = _DURATION.match(duration.strip())
    if spoken and (spoken["hours"] or spoken["minutes"]):
        return int(spoken["hours"] or 0) * 3600 + int(spoken["minutes"] or 0) * 60
    hours, minutes, rest = (duration.strip().split(":") + ["", ""])[:3]
    if not (hours.isdigit() and minutes.isdigit() and rest.isdigit()):
        raise MissionError(f"a hold lasts `3h`, `90m`, `1h30m` or `HH:MM:SS`, not {duration!r}")
    return int(hours) * 3600 + int(minutes) * 60 + int(rest)


def _rentable(plan: ExecutionPlan) -> Rentable:
    """The provider backend `plan` rents through, refusing one that hands out no ssh machine."""
    destination = route(plan.profile.kind)
    renting = None if destination == "ssh-family" else renter(destination(), plan)
    if renting is None:
        raise MissionError(
            f"{plan.host!r} is a {plan.profile.kind!r} host; only a provider that rents an ssh "
            "machine can be held"
        )
    return renting
