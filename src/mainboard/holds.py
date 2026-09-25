# `mainboard hold` and `mainboard release`: a rented machine kept for a session and reached as
# an ordinary ssh host, so it is set up once and then takes any number of jobs in seconds each.
#
# A rental per job rebuilds the whole environment every time: of a 17.5 minute landing on
# 2026-09-21 the upload was 1.3 minutes and the environment nearly all the rest, for a job of 2.7.
# Holding by hand (an ssh block, a manifest host, a watchdog to destroy it) is steps a tired
# session forgets. Here the rental joins the run registry dispatched rentals live in, its
# deadline as the lease, so the durable sweep releases it on time whoever is watching, and
# `compute` releases whatever is past due before it lists.

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from .core.errors import MissionError
from .costs.catalog import Offer
from .dispatch import vocabulary
from .dispatch.aliases import SshAliases
from .dispatch.backends.base import route
from .dispatch.landing import Renter, renter
from .dispatch.lease import Lease
from .dispatch.rentals import handoff
from .dispatch.shared import announce, logger
from .dispatch.shipment import Shipment
from .dispatch.transport import SshTransport
from .dispatch.wrapping import connection
from .engines.compile.provisioner import Provisioner
from .manifest.held import Held, Holdings
from .manifest.schema.queue import Defaults

if TYPE_CHECKING:
    from .board import Board
    from .context.plan import ExecutionPlan
    from .dispatch.rentals import Rental
    from .dispatch.shared import Watcher
    from .dispatch.vocabulary import Resources

_DURATION = re.compile(r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?$")

# The rented entrypoint gives up on a landing that never comes, and its exit would read as a
# finished job the sweep settles and releases, so a held machine runs this until released.
_IDLE = "exec sleep infinity"

# Every rented image this house lands on is a Linux container.
_PLATFORM = "linux-64"


class Holds:
    """The machines this workspace is holding: rented, set up, listed and released.

    aliases: the ssh config the aliases are written into, the user's own by default.
    """

    def __init__(self, board: Board, *, aliases: SshAliases | None = None) -> None:
        self.board = board
        self.aliases = aliases or SshAliases()
        self.holdings = Holdings(board.root)

    def hold(
        self,
        provider: str,
        *,
        duration: str,
        alias: str = "",
        gpu_name: str = "",
        gpus: int = 0,
        max_usd: float = 0.0,
        env: str = "",
        watch: Watcher | None = None,
    ) -> Held:
        """Rent a machine through `provider`, set it up as host `alias`, and keep it until due.

        The deadline counts from the moment the machine is ready, since a hold is for the time it
        can take jobs; until then the rental's own lease, priced with the landing, bounds it.

        provider: the provider host to rent through, `vast` say.
        duration: how long to keep it once ready, `3h`, `90m` or `1h30m`.
        alias: the host name to reach it by, `<provider>-<card>` when empty.
        gpu_name: the card to rent, in the provider's own spelling.
        gpus: cards per machine, the provider profile's default when 0.
        max_usd: the spend cap over the whole hold, landing included.
        env: the environment to set up, the provider profile's own when empty.
        watch: announces each stage as it begins.
        """
        seconds = duration_seconds(duration)
        watch = watch or announce
        rented = self.board.on(provider)
        plan = rented.plan(env=env, container="none")
        backend = _rentable(plan)
        name = alias or _slug(f"{provider}-{gpu_name or plan.profile.defaults.gpu_name}")
        if name in self.board.manifest.hosts:
            raise MissionError(f"{name!r} already names a host; hold it under another --as")
        resources = rented.resources(
            walltime=_walltime(seconds), gpus=gpus, gpu_name=gpu_name, max_usd=max_usd, plan=plan
        )
        Provisioner(self.board.root, self.board.manifest).compiler_for(plan.env).vouch()
        rental = self._rent(backend, plan, resources, name=name, duration=duration)
        try:
            return self._keep(name, rental, plan=plan, seconds=seconds, watch=watch)
        except BaseException:
            logger.warning("hold %s failed during setup; releasing rental %s", name, rental.handle)
            self._end(name, handle=rental.handle, provider=provider)
            raise

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
                logger.warning("could not release %s yet: %s", held.alias, fault)
        return released

    def _rent(
        self,
        backend: Renter,
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
            plan, shipment, resources, name=f"hold-{name}", evidence="not_started"
        )
        try:
            return backend.rent(plan, resources, allocation=allocation)
        except BaseException:
            allocation.interrupted()
            raise

    def _keep(
        self, name: str, rental: Rental, *, plan: ExecutionPlan, seconds: int, watch: Watcher
    ) -> Held:
        """Name the machine, set it up, park its entrypoint, and start the clock."""
        self.aliases.add(name, rental.endpoint)
        record = self.board.dispatcher.cache.run(rental.handle, plan.host)
        offer = record.lease.offer if record.lease is not None else None
        profile = plan.profile.model_copy(
            update={
                "kind": "ssh",
                "platform": _PLATFORM,
                "defaults": Defaults(walltime=_walltime(seconds)),
                "queues": {},
            }
        )
        held = Held(
            alias=name,
            provider=plan.host,
            handle=rental.handle,
            gpu=offer.gpu if offer is not None else "",
            usd_hr=offer.rate_usd_hr if offer is not None else None,
            deadline=datetime.now(UTC) + timedelta(seconds=seconds),
            profile=profile,
        )
        self._record(held)
        setup = self.board.on(name).install(plan.env, watch=watch)
        self._park(name, watch=watch)
        ready = held.model_copy(
            update={
                "deadline": datetime.now(UTC) + timedelta(seconds=seconds),
                "profile": profile.model_copy(update={"root": setup.root}),
            }
        )
        self._record(ready)
        cache = self.board.dispatcher.cache
        cache.relet(
            cache.delivery(record, "pending"),
            Lease(
                offer=offer or Offer(provider=plan.profile.kind, gpu=ready.gpu, rate_usd_hr=0.0),
                release_by=ready.deadline,
            ),
        )
        return ready

    def _park(self, name: str, *, watch: Watcher) -> None:
        """Hand the waiting entrypoint a command that never ends, so the machine stays up."""
        watch(f"parking {name} until it is released")
        with connection(name, SshTransport()) as remote:
            retcode, _, err = (remote["bash"]["-c", handoff()] << f"{_IDLE}\n").run(retcode=None)
        if retcode:
            raise MissionError(f"could not park {name}: {str(err).strip()[-400:]}")

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
        self.board.dispatcher.cache.drop_host(alias)

    def _record(self, held: Held) -> None:
        """Write `held` down and let the workspace resolve its alias as a host from now on."""
        self.holdings.save(held)
        self._reload()

    def _reload(self) -> None:
        """Forget the loaded manifest, so the next read sees the holdings as they are now."""
        for derived in ("manifest", "resolver"):
            self.board.shared.pop(derived, None)

    def _settled(self, held: Held) -> bool:
        """Whether `held`'s record says the rental already ended, or no record is left."""
        try:
            record = self.board.dispatcher.cache.run(held.handle, held.provider)
        except LookupError:
            return True
        return record.verdict in vocabulary.TERMINAL


def duration_seconds(duration: str) -> int:
    """`duration` as a person writes it (`3h`, `90m`, `1h30m` or `HH:MM:SS`) in seconds."""
    spoken = _DURATION.match(duration.strip())
    if spoken and (spoken["hours"] or spoken["minutes"]):
        return int(spoken["hours"] or 0) * 3600 + int(spoken["minutes"] or 0) * 60
    hours, minutes, rest = (duration.strip().split(":") + ["", ""])[:3]
    if not (hours.isdigit() and minutes.isdigit() and rest.isdigit()):
        raise MissionError(f"a hold lasts `3h`, `90m`, `1h30m` or `HH:MM:SS`, not {duration!r}")
    return int(hours) * 3600 + int(minutes) * 60 + int(rest)


def _walltime(seconds: int) -> str:
    """`seconds` as the `HH:MM:SS` walltime a resource request carries."""
    hours, rest = divmod(seconds, 3600)
    return f"{hours:02d}:{rest // 60:02d}:{rest % 60:02d}"


def _slug(text: str) -> str:
    """`text` as an ssh alias: lower case, runs of anything else collapsed to one dash."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


def _rentable(plan: ExecutionPlan) -> Renter:
    """The provider backend `plan` rents through, refusing one that hands out no ssh machine."""
    destination = route(plan.profile.kind)
    renting = None if destination == "ssh-family" else renter(destination(), plan)
    if renting is None:
        raise MissionError(
            f"{plan.host!r} is a {plan.profile.kind!r} host; only a provider that rents an ssh "
            "machine can be held"
        )
    return renting
