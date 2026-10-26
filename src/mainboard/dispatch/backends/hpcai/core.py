# `HpcAiBackend` runs a command on an HPC-AI instance through their REST API (see `api`).
#
# A dispatch lands on an instance over ssh. Unlike vast, no endpoint attaches a key to a running
# instance, so the landing connects with a key the account already registered in the console.
#
# The instance type is chosen from that catalog by the request (card, architecture, spot, count)
# unless the host pins one in `[hosts.<name>.vars] instance-type-id`. Their GPU types are whole
# 8-card nodes as of 2026-09-29, and interruptible capacity is a type of its own (`B200-...-SPOT`).

from contextlib import suppress
from time import sleep
from typing import TYPE_CHECKING
from urllib.error import HTTPError

from ....core.errors import MissionError
from ....costs.catalog import Offer
from ....log import logger
from ...hosts.rentals import Identity, Rental, identity, reachable, waiting
from ...jobs.arch import card
from ...jobs.evidence import framing, staging
from ...jobs.vocabulary import JobState, OfferFilter
from ...transport.core import Endpoint
from ..base import (
    Account,
    Delivery,
    Inventory,
    LogSource,
    Market,
    ProviderBackend,
    Rentable,
    Standing,
    forgotten,
    hourly_cap,
    http_transport,
)
from ..constants import ADDRESS_ATTEMPTS, ADDRESS_SECONDS
from .api import HpcAiApi, Instance, InstanceType, api_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from ....context.plan import ExecutionPlan
    from ....manifest.schema.compute.host import HostProfile
    from ...hosts.allocation import Allocation
    from ...jobs.vocabulary import Resources
    from ..base import Rented, Transport

# The initScript's output and exit-code sentinels, on the instance's own data disk: a rental with
# no `remoteStorages` has no mounted volume, and one instance runs one command, so fixed names do.
_SENTINEL_DIR = "/root/dataDisk"
_LOG_PATH = f"{_SENTINEL_DIR}/mainboard.log"
_EXIT_PATH = f"{_SENTINEL_DIR}/mainboard.exit"
# The one status that can answer ssh at all.
_RUNNING = "Running"
# Polls in a row a start may sit in a back-off before it is read as never starting: an image that
# will not pull backs off for good (`DownloadImage` / `BackOff`, a pinned image id, 2026-09-29).
_BACKOFF_POLLS = 18
# `instanceRuntimeInfo.status` onto our verdicts, keyed in lower case since HPC-AI spells states
# in camel case (`StartingFailed`). The set their list-instances doc publishes; any other status
# reads as "unknown".
_VERDICTS = {
    "initializing": "running",
    "pullingimage": "running",
    "starting": "running",
    "restarting": "running",
    "running": "running",
    "stopping": "running",
    "stopped": "ok",
    "archived": "ok",
    "released": "vanished",
    "startingfailed": "failed",
    "initializationfailed": "failed",
}


class HpcAiBackend(ProviderBackend, Account, Inventory, Market, Rentable):
    """Run a command on an HPC-AI instance, its own REST API standing in for a scheduler.

    HPC-AI reports instance status only, never a process exit code, so the initScript writes the
    command's exit code and output to sentinel files on the data disk: `state` says whether the
    instance is up, the sentinels how the command ended. Output never leaves that disk, so there
    is no `LogSource` or `Delivery`; `lacks` names the path to read by hand.

    spot: whether created instances are spot (cost-optimized, preemptible).
    transport: sends a prepared `Request`, returning its response; injectable for tests.
    sleeper: waits between instance polls, injected so a test drives it without real time.
    """

    name = "hpc-ai"

    lacks = {
        Delivery: "hpc-ai backend cannot deliver {path!r} yet; download "
        f"{_SENTINEL_DIR}/mainboard.* from instance {{handle}} over ssh until that path lands",
        LogSource: f"hpc-ai backend has no server-side logs; read {_LOG_PATH} on instance "
        "{handle} over ssh instead",
    }

    @staticmethod
    def _required(profile: HostProfile, key: str) -> str:
        """`profile.vars[key]`, refusing by name when missing.

        `HostProfile` forbids undeclared fields, so HPC-AI's opaque ids (`instance-type-id`,
        `image-id`, `region`) live in its free-form `vars` table.
        """
        try:
            return profile.vars[key]
        except KeyError:
            raise MissionError(
                f"the hpc-ai backend needs [hosts.<name>.vars] {key!r} set"
            ) from None

    def __init__(
        self,
        *,
        spot: bool = False,
        transport: Transport = http_transport,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.spot = spot
        self.api = HpcAiApi(transport=transport)
        self.sleeper = sleeper

    def _created(
        self, plan: ExecutionPlan, resources: Resources, *, script: str, allocation: Allocation
    ) -> str:
        """Create an instance whose initScript runs `script`, returning HPC-AI's `instanceId`.

        Every field their create endpoint calls required is sent (`billing`, `nodePorts`), since
        the validator rejects rather than defaults an omission. The `name` is ours and never an
        address. `script` runs inside one redirect into the sentinel pair, followed by whatever
        it leaves in `$status`, the only place a post-mortem can read either.

        plan: whose `vars` name the image, and may pin the type and region.
        resources: the request a type is chosen by when the host pins none.
        script: a command for a prebuilt image, or the waiting entrypoint for a landing.
        """
        kind = self.chosen(plan, resources)
        body = {
            "name": allocation.label,
            "isSpotInstance": kind.spot,
            "instanceTypeId": kind.instance_type_id,
            "imageId": HpcAiBackend._required(plan.profile, "image-id"),
            "region": kind.region_id,
            "billing": {"chargeMode": "perHour", "duration": 1},
            "remoteStorages": [],
            "instanceConfiguration": {
                "enableCommonData": False,
                "enableDocker": False,
                "initScript": (
                    f"mkdir -p {_SENTINEL_DIR}\n"
                    f"{{ {script}\n}} > {_LOG_PATH} 2>&1\n"
                    f"echo $status > {_EXIT_PATH}\n"
                ),
            },
            "nodePorts": [],
        }
        allocation.begin()
        payload = self.api.request("POST", path="/instance/create", body=body)
        return allocation.created(str(payload["instanceId"]))

    def _opened(self, handle: str, *, key: Identity) -> Endpoint:
        """Answer once ssh lets us onto `handle`, naming the console key when it never does.

        A machine that answers nothing is almost always this workspace's key missing from the
        account's list, since HPC-AI attaches no key at create time.
        """
        endpoint = self.endpoint(handle, key=key.private)
        try:
            return reachable(endpoint, sleeper=self.sleeper)
        except MissionError as refused:
            raise MissionError(
                f"{refused}. Add the public half of {key.private} to the ssh keys in the HPC-AI "
                "console, since an instance is only reachable with a key the account registered "
                "before it was created."
            ) from None

    def cancel(self, handle: str) -> None:
        """Stop the instance, then destroy it, so a cancelled run stops billing rather than idles.

        `/instance/terminate` destroys (verified live 2026-08-19): their docs page is titled
        "delete" but its cURL calls terminate, and `/instance/delete` answers 404. A refused stop
        never blocks the terminate that ends the billing.
        """
        with suppress(HTTPError):
            self.api.request("POST", path="/instance/stop", body={"instanceId": handle})
        try:
            self.api.request("POST", path="/instance/terminate", body={"instanceId": handle})
        except HTTPError as error:
            forgotten(error)

    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        """The types in stock as offers, cheapest first, CPU ones only when no card is asked."""
        rows = [
            Offer(
                provider=self.name,
                gpu=kind.card or "cpu",
                gpu_count=kind.gpus,
                spot=kind.spot,
                region=kind.region,
                rate_usd_hr=kind.usd_hr,
                source=f"hpc-ai:{kind.instance_type_id}",
            )
            for kind in self.api.types()
            if kind.in_stock
            and kind.usd_hr is not None
            and (kind.card == card(gpu_name) if gpu_name else True)
            and (not gpus or kind.gpus == gpus)
        ]
        return rows[:limit] if limit else rows

    def chosen(self, plan: ExecutionPlan, resources: Resources) -> InstanceType:
        """The type this request rents: the host's pinned one, else the cheapest that fits.

        Fits means in stock, at least the cards asked, the card or architecture asked, spot
        exactly when asked, and an hourly rate the spend cap covers over the walltime.
        """
        spot = self.spot or resources.spot
        if declared := plan.profile.vars.get("instance-type-id", ""):
            region = HpcAiBackend._required(plan.profile, "region")
            return InstanceType(instance_type_id=declared, region_id=region, spot=spot)
        wanted = OfferFilter.rented(
            resources.model_copy(update={"spot": spot}),
            ceiling=hourly_cap(resources, landing=Rental.LANDING_SECONDS),
        )
        gpu = bool(wanted.gpus or wanted.gpu_name or wanted.arch)
        kinds = self.api.types()
        if fitting := [k for k in kinds if k.in_stock and bool(k.gpu) == gpu and k.fits(wanted)]:
            return fitting[0]
        listed = sorted({f"{kind.gpus}x {kind.gpu}" for kind in kinds if kind.gpu})
        raise MissionError(
            f"hpc-ai has no {wanted.spoken or 'CPU'} type in stock right now; it lists "
            f"{', '.join(listed)} (`mb host offers --provider hpc-ai` shows which are in stock)"
        )

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Where ssh reaches instance `handle`, waited for until it runs and publishes one.

        Only a `Running` row carries the address, mapped port and login, after the image is
        pulled and the container started; one that never gets there is refused by id.

        key: the private key file the connection uses, empty to leave that to ssh's own config.
        """
        backing_off, found = 0, Instance(handle=handle)
        for _ in range(ADDRESS_ATTEMPTS):
            found = self.api.instance(handle) or Instance(handle=handle)
            if found.status == _RUNNING and found.address and found.port:
                return Endpoint(
                    address=found.address, port=found.port, user=found.user, identity=key
                )
            backing_off = backing_off + 1 if found.reason == "BackOff" else 0
            if backing_off >= _BACKOFF_POLLS:
                raise MissionError(
                    f"hpc-ai instance {handle} is stuck in {found.said}: {found.message[:200]}. "
                    "In `DownloadImage` the image will not pull; pick an image the console offers "
                    "for this type and set its id as [hosts.<name>.vars] image-id"
                )
            self.sleeper(ADDRESS_SECONDS)
        raise MissionError(
            f"hpc-ai instance {handle} never published an ssh endpoint (last: {found.said}); look "
            "it up in the console and terminate it if it is still billing"
        )

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """Rent an instance whose initScript waits for a landing; end it if it cannot be opened."""
        self.admit(plan, resources)
        key = identity(plan.profile.vars.get("ssh-key", ""))
        handle = self._created(plan, resources, script=f"{waiting()}\n", allocation=allocation)
        try:
            return Rental(handle=handle, endpoint=self._opened(handle, key=key))
        except BaseException:
            logger.warning("hpc-ai instance {} could not be opened, ending the rental", handle)
            self.cancel(handle)
            raise

    def rentals(self) -> list[Rented]:
        """Every instance on the account, whoever created it."""
        return list(self.api.instances())

    def standing(self) -> Standing:
        """The account balance HPC-AI reports, or the key that is missing.

        `/balance` is the one surface of the three providers publishing a number without a
        rental. The row takes `balance`, the pot left once vouchers and monthly credits are spent
        first (the order their billing doc states). No price is quoted without an instance type.
        """
        try:
            api_key()
        except MissionError as unset:
            return Standing(note=str(unset))
        return Standing(keyed=True, credit_usd=self.api.balance())

    def state(self, handle: str) -> JobState:
        if (found := self.api.instance(handle)) is None:
            return JobState(handle=handle, verdict="vanished")
        verdict = _VERDICTS.get(found.status.lower(), "unknown")
        return JobState(handle=handle, state=found.status, verdict=verdict)

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        """Run `command` as the initScript, for a prebuilt image that needs nothing landed."""
        self.admit(plan, resources)
        # `status=$?` comes first, else the framing's own status is what `$?` reports. The
        # receipts are framed into the same captured log, so a reader of that file gets both.
        return self._created(
            plan,
            resources,
            script=f"{staging()}\n{command}\nstatus=$?\n{framing()}\n",
            allocation=allocation,
        )
