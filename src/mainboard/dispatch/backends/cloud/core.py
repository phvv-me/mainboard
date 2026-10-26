# Machines rented from any cloud through one template, no server between.
#
# Studied against SkyPilot (a provisioner per cloud: run, wait, query, stop, terminate instances,
# behind an API server) and dstack (a `Compute` base, `get_offers` / `create_instance` /
# `terminate_instance`, with capability mixins, also behind a server). Both keep the machine's
# lifecycle small and put the work around it (pricing, keys, waiting for ssh, cleaning up a
# machine that failed) in shared code. Here that shared code is `CloudBackend.rent`, and a
# provider supplies its gpuhunt catalog name and a `CloudApi` of four calls. Everything after ssh
# answers is mb's ordinary host path (mirror, tool, lock, pueue), which is why a held machine is
# set up once and then takes jobs like any host.
#
# A cloud machine has no entrypoint mb can rely on (a Lambda VM has none at all), so once ssh
# answers the rent starts the landing's waiter itself, detached, its output and exit status
# written beside the launch script. A held machine parks that waiter on an idle command; a
# one-shot `job submit` hands it the job, and `state` and `logs` read the two files over ssh.

import abc
import shlex
from time import monotonic, sleep
from typing import TYPE_CHECKING, ClassVar

from ....core.errors import MissionError
from ....log import logger
from ...hosts.lease import Lease
from ...hosts.rentals import Rental, identity, reachable, waiting
from ...jobs.vocabulary import JobState, OfferFilter
from ...transport.core import Endpoint, HostUnreachable, SshTransport
from ..base import (
    Account,
    Delivery,
    Inventory,
    LogSource,
    Market,
    ProviderBackend,
    Rentable,
    Standing,
    hourly_cap,
    http_transport,
)
from ..exceptions import CapacityGone
from .offers import as_offer, hunt

if TYPE_CHECKING:
    from collections.abc import Callable

    from gpuhunt import CatalogItem

    from ....context.plan import ExecutionPlan
    from ....costs.catalog import Offer
    from ...hosts.allocation import Allocation
    from ...jobs.vocabulary import Resources
    from ..base import Rented, Transport
    from .api import CloudApi

# How long a created machine may take to publish an address before it is ended unused.
_BOOT_SECONDS = 900
_POLL_SECONDS = 10.0
# How many offers a rent tries when the first ones are taken or out of capacity.
_OFFERS_TRIED = 5
# Where the detached waiter leaves the job's output and exit status, beside the launch script.
_LOG_PATH = "/tmp/mainboard.log"
_EXIT_PATH = "/tmp/mainboard.exit"
# The landing's waiter, its output and exit status kept where `state` and `logs` read them.
_WAITER = (
    f"{{\n{waiting()}\n}} > {_LOG_PATH} 2>&1\n"
    f"echo $status > {_EXIT_PATH}.part && mv {_EXIT_PATH}.part {_EXIT_PATH}\n"
)


class CloudBackend(ProviderBackend, Account, Inventory, LogSource, Market, Rentable):
    """A cloud that rents ssh machines by the hour, held (`mb host hold`) or one per job.

    A subclass names its gpuhunt catalog and its `CloudApi`; renting, leasing, reaching,
    reporting and ending a machine are shared. A job submitted to a cloud host lands on a machine
    rented for it, ended once the job settles; a held machine is set up once and takes any number
    of jobs through its alias.
    """

    # The name gpuhunt lists this cloud under.
    catalog_name: ClassVar[str] = ""

    lacks = {
        Delivery: "a cloud machine is an ssh host: `mb job collect {path} --on <alias>`",
    }

    @property
    @abc.abstractmethod
    def client(self) -> type[CloudApi]:
        """The API this cloud's machines are rented through."""

    def __init__(
        self,
        *,
        transport: Transport = http_transport,
        sleeper: Callable[[float], None] = sleep,
        disk_gb: int = 100,
    ) -> None:
        self.api = self.client(transport=transport)
        self.sleeper = sleeper
        self.disk_gb = disk_gb

    def _booted(self, handle: str, *, private: str) -> Endpoint:
        """The ssh endpoint of `handle` once its cloud publishes one."""
        deadline = monotonic() + _BOOT_SECONDS
        while monotonic() < deadline:
            found = self.api.machine(handle)
            if found.status == "gone":
                raise MissionError(f"{self.name} machine {handle} ended while booting")
            if found.status == "running" and found.host:
                return Endpoint(
                    address=found.host, port=found.port, user=found.user, identity=private
                )
            self.sleeper(_POLL_SECONDS)
        raise MissionError(
            f"{self.name} machine {handle} published no address in {_BOOT_SECONDS}s"
        )

    def _created(self, resources: Resources, *, allocation: Allocation, public_key: str) -> str:
        """A machine from the cheapest offer that still has one, leased before it is created.

        The lease comes first, so a crash between create and record still leaves the run registry
        holding the handle the sweep ends.
        """
        found = self._offers(resources)
        for offer in found:
            allocation.begin(
                lease=Lease.priced(as_offer(offer), resources, setup_s=Rental.LANDING_SECONDS)
            )
            try:
                handle = self.api.create(
                    offer, name=allocation.label, public_key=public_key, disk_gb=self.disk_gb
                )
            except CapacityGone as gone:
                allocation.refused()
                logger.warning(
                    "{} offer {} is gone ({}); trying the next",
                    self.name,
                    offer.instance_name,
                    gone,
                )
                continue
            allocation.created(handle)
            return handle
        raise MissionError(f"{self.name} had no capacity for the {len(found)} cheapest offers")

    def _landed(self, handle: str, *, private: str) -> Endpoint:
        """`handle` answering ssh, with the landing's waiter started on it, detached."""
        endpoint = reachable(self._booted(handle, private=private), sleeper=self.sleeper)
        self._shell(
            endpoint,
            f"nohup setsid bash -c {shlex.quote(_WAITER)} >/dev/null 2>&1 </dev/null &\n",
            operation="start the waiter",
        )
        return endpoint

    def _offers(self, resources: Resources) -> list[CatalogItem]:
        """The cheapest offers under the budget a rent tries, refusing when there is none."""
        rented = resources.model_copy(update={"gpus": max(resources.gpus, 1)})
        wanted = OfferFilter.rented(
            rented, ceiling=hourly_cap(resources, landing=Rental.LANDING_SECONDS)
        )
        if found := hunt(wanted, providers=[self.catalog_name])[:_OFFERS_TRIED]:
            return found
        raise MissionError(
            f"{self.name} lists no {wanted.spoken} offer right now; `mb host offers` shows every "
            "cloud's"
        )

    def _read(self, handle: str, *, path: str) -> str:
        """The file at `path` on `handle`, empty while it is not there yet."""
        endpoint = self.endpoint(handle, key=identity().private)
        return self._shell(
            endpoint, f"cat {shlex.quote(path)} 2>/dev/null || true\n", operation="read"
        )

    def _shell(self, endpoint: Endpoint, script: str, *, operation: str) -> str:
        """Run `script` under bash on `endpoint`, answering its stdout; a failure is refused."""
        policy = SshTransport(endpoint=endpoint)
        where = endpoint.destination
        code, out, err = policy.invoke(
            (*policy.command(where), "bash -s"),
            where,
            operation=operation,
            input_text=script,
        )
        if code:
            raise MissionError(f"{self.name} {operation} on {where} exited {code}: {err.strip()}")
        return out

    def cancel(self, handle: str) -> None:
        self.api.terminate(handle)

    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        found = hunt(OfferFilter(gpu_name=gpu_name, gpus=gpus), providers=[self.catalog_name])
        return [as_offer(item) for item in (found[:limit] if limit else found)]

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        found = self.api.machine(handle)
        if not found.host:
            raise MissionError(f"{self.name} machine {handle} has no address ({found.status})")
        return Endpoint(address=found.host, port=found.port, user=found.user, identity=key)

    def logs(self, handle: str) -> str:
        return self._read(handle, path=_LOG_PATH)

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """The cheapest offer under the budget, created, reachable over ssh, or nothing billing."""
        self.admit(plan, resources)
        key = identity(plan.profile.vars.get("ssh-key", ""))
        handle = self._created(resources, allocation=allocation, public_key=key.public)
        try:
            return Rental(handle=handle, endpoint=self._landed(handle, private=key.private))
        except BaseException:
            logger.warning("{} machine {} never became reachable; ending it", self.name, handle)
            self.api.terminate(handle)
            raise

    def rentals(self) -> list[Rented]:
        return list(self.api.machines())

    def standing(self) -> Standing:
        try:
            self.api.key()
        except MissionError:
            return Standing(note=f"set {' or '.join(self.client.key_variables)} to rent here")
        return Standing(keyed=True, note="gpuhunt prices; `mb host offers` lists them")

    def state(self, handle: str) -> JobState:
        """The machine's status, then the job's own exit file once the machine runs.

        A machine the cloud ended reads `vanished`, which is also what an interruptible machine
        taken back reads as; its job resumes from its checkpoint with `mb job submit --resume`.
        """
        found = self.api.machine(handle)
        if found.status == "gone":
            return JobState(handle=handle, state=found.status, verdict="vanished")
        if found.status != "running" or not found.host:
            return JobState(handle=handle, state=found.status, verdict="running")
        try:
            said = self._read(handle, path=_EXIT_PATH).strip()
        except (HostUnreachable, MissionError) as unanswered:
            logger.debug("{} {} gave no exit file: {}", self.name, handle, unanswered)
            said = ""
        if not said.lstrip("-").isdigit():
            return JobState(handle=handle, state=found.status, verdict="running")
        code = int(said)
        verdict = "ok" if code == 0 else "failed"
        return JobState(handle=handle, state=found.status, exit_code=code, verdict=verdict)

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        raise MissionError(
            f"{self.name} runs no prebuilt image: drop the container from {plan.host!r} so the "
            "job lands on a machine rented for it, or hold one with `mb host hold`"
        )
