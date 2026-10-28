# `VastBackend` rents a Vast.ai machine for one command through their REST API (`api`), choosing
# among the market's offers (`market`).
#
# The plan decides the rental's shape, not a flag. A plan naming no container of its own gets
# `rent`: the `ssh` runtype with the waiting entrypoint, then the ordinary mirror, install,
# provision and pin path. A plan declaring its own image keeps `submit`, the raw command as the
# container's entrypoint in `args` launch mode, since a prebuilt image is the one case where the
# box already holds everything the command needs, and a one-shot container is cheaper and simpler.

import json
from contextlib import suppress
from time import sleep
from typing import TYPE_CHECKING, ClassVar
from urllib.error import HTTPError

from ....core.errors import MissionError
from ....costs.imports import from_vast
from ....log import logger
from ...hosts.lease import Lease
from ...hosts.rentals import Identity, Rental, identity, reachable, seeded, waiting
from ...jobs.arch import arch
from ...jobs.evidence import framing, staging
from ...jobs.vocabulary import JobState
from ...transport.core import Endpoint
from ..base import (
    ProviderBackend,
    Rentable,
    forgotten,
    hourly_cap,
    http_transport,
    image_cuda,
)
from ..capabilities import Account, Delivery, Inventory, LogSource, Market, Rented, Standing
from ..constants import ADDRESS_ATTEMPTS, ADDRESS_SECONDS
from .api import VastApi, api_key
from .market import Ask, VastMarket, base_image, bid

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ....context.plan import ExecutionPlan
    from ....costs.catalog import Offer
    from ...hosts.allocation import Allocation
    from ...jobs.vocabulary import Resources
    from ..base import Transport

# Offers one rental may try when the market takes each between search and create, which Vast
# answers with this token (RTX 5090 offer 26371154, 2026-09-12).
# Six since a stale listing keeps answering `no_such_ask` and, being the most reliable, keeps
# being picked first (sm_120 spot offer 51352327 in two rents running, 2026-09-29).
_PICK_ATTEMPTS = 6
_NO_SUCH_ASK = "no_such_ask"
# Local disk per rental in GB, also what a search prices storage at, so quote and machine agree.
# The mirror is a few hundred MB but the environment is a whole CUDA stack plus its package cache,
# tens of GB; storage is cents per GB-month, while running out of disk costs the whole job.
_DISK_GB = 64.0
# Echoed after the command with its exit code, the only place a verdict learns how the command
# ended, since Vast reports container status only.
_EXIT_SENTINEL = "mainboard-exit:"
# Statuses whose container has not run the command yet, so no marker can exist.
_PENDING_STATUSES = frozenset({"created", "loading"})
# Statuses whose container is still up; any other (a new Vast state, or none) reads "unknown"
# once the log has failed to say how the command ended.
_LIVE_STATUSES = frozenset({"created", "loading", "running", "stopping"})
# One card always listed in volume, so a price sample reads as a real market rate.
_SAMPLE_GPU = "RTX 4090"
_SSH_USER = "root"
# The status a machine reaches before its sshd can answer at all.
_RUNNING = "running"


class OfferTaken(MissionError):
    """The offer a create named was rented by someone else between the search and the create."""

    def __init__(self, identifier: int | str) -> None:
        super().__init__(f"vast offer {identifier} was taken before the create landed")


class VastBackend(ProviderBackend, Account, Inventory, LogSource, Market, Rentable):
    """Rent a Vast.ai machine for one command, the container's own lifetime being the job's.

    Vast rents containers, not jobs, and reports container status, never a process exit code,
    so the wrapper echoes an exit sentinel into the log and `state` reads it back. A finished
    command does not end the rental: Vast holds an instance at its `intended_status`, restarting
    the exited container so the command runs again and appends another sentinel (verified live
    2026-08-19, thirteen restarts in five minutes). Only `cancel` stops the meter, so a caller
    reaching a terminal verdict must still cancel, as the durable sweep does.

    Stateless between calls: every method addresses an instance by its contract id. The disk a
    rental wrote to is destroyed with it, hence `Delivery` in `lacks`.
    """

    name = "vast"

    lacks = {
        Delivery: "vast backend cannot deliver {path!r} yet; a rented machine's disk dies with "
        "the instance, so have the command upload its own results and read `logs {handle}` "
        "until that path lands",
    }

    CUDA_FLOOR: ClassVar[float] = VastMarket.CUDA_FLOOR

    @staticmethod
    def __refusal(refused: HTTPError, *, offer: Mapping, allocation: Allocation) -> MissionError:
        """The error a refused create raises, its reservation closed when nothing was created."""
        reason = VastBackend._detail(refused)
        if 400 <= refused.code < 500:
            # Declined after validation, so nothing was created and the reservation closes.
            allocation.refused()
        if _NO_SUCH_ASK in reason:
            return OfferTaken(offer["id"])
        return MissionError(
            f"vast refused offer {offer['id']} (HTTP {refused.code}): {reason[:400]}"
        )

    @staticmethod
    def _detail(refused: HTTPError) -> str:
        """What Vast said when it refused a call, empty when it said nothing readable."""
        try:
            detail = json.load(refused)
        except ValueError:
            return ""
        return str(detail.get("msg") or detail.get("message") or detail.get("error") or "")

    @staticmethod
    def _exit_sentinel(log: str) -> int | None:
        """The exit status the wrapper echoed into `log`, None when no marker is readable.

        The last marker wins, since a container Vast restarted appends its own below the first, and
        a marker without a number is skipped rather than read as a zero.
        """
        for line in reversed(log.splitlines()):
            _, marked, status = line.partition(_EXIT_SENTINEL)
            if marked:
                with suppress(ValueError):
                    return int(status.strip())
        return None

    def __ask(self, plan: ExecutionPlan, resources: Resources, *, landing: int) -> Ask:
        """What a rental of `plan` under `resources` asks the market for.

        landing: seconds of setup billed before the job, which sit inside the price ceiling.
        """
        own = image_cuda(plan.container.image) if plan.container is not None else None
        return Ask(
            gpu_name=resources.gpu_name,
            gpus=max(resources.gpus, 1),
            max_usd_hr=hourly_cap(resources, landing=landing),
            cuda=max(self.CUDA_FLOOR, own or 0.0),
            arch=arch(resources.arch) if resources.arch else None,
            datacenter=resources.datacenter,
        )

    def __attach(self, handle: str, *, key: str) -> None:
        """Put this workspace's public key on the rental, so the landing can log in.

        Vast copies an account key onto a new instance itself; this adds the one key this
        machine holds the private half of. A refusal is fatal: nobody could log into the rental.
        """
        try:
            self.api.request("POST", path=f"/instances/{handle}/ssh/", body={"ssh_key": key})
        except HTTPError as refused:
            raise MissionError(
                f"vast refused the ssh key for instance {handle} ({refused}); add the key at "
                "https://cloud.vast.ai/manage-keys/ and submit again"
            ) from refused

    def __exit_code(self, handle: str) -> int | None:
        """`handle`'s process exit status from the sentinel in its log tail.

        None when the log cannot be fetched or carries no marker, so a clean container stop is
        never read as a clean run.
        """
        try:
            log = self.api.logs(handle)
        except HTTPError, MissionError:
            return None
        return VastBackend._exit_sentinel(log)

    def __init__(
        self,
        *,
        spot: bool = False,
        disk_gb: float = _DISK_GB,
        transport: Transport = http_transport,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        """spot: whether to rent interruptible (bid) capacity instead of on-demand.
        disk_gb: local disk per rental, also the storage an offer search is priced at.
        transport: sends a prepared `Request`, returning its response, injectable for tests.
        sleeper: waits between polls, injected so a test drives it without real time.
        """
        self.api = VastApi(transport=transport, sleeper=sleeper)
        self.market = VastMarket(self.api, spot=spot, disk_gb=disk_gb)

    def __opened(self, handle: str, *, key: Identity) -> Endpoint:
        """Answer once ssh really lets us onto `handle`, its key attached as soon as it is up.

        The key goes on once there is a running instance to copy it into, and the knocking that
        follows absorbs the seconds it takes to reach the container's authorized keys.
        """
        endpoint = self.endpoint(handle, key=key.private)
        self.__attach(handle, key=key.public)
        return reachable(endpoint, sleeper=self.api.sleeper)

    def __placed(
        self,
        ask: Ask,
        *,
        plan: ExecutionPlan,
        resources: Resources,
        launch: dict,
        allocation: Allocation,
    ) -> str:
        """Create a rental from the page `ask` finds, answering its contract id.

        An offer taken before the create is retried from the same page, never a fresh search.
        """
        page = self.market.search(ask)
        for _ in range(_PICK_ATTEMPTS):
            if not page:
                self.market.refuse(ask)
            offer = self.market.best(page)
            try:
                return self.__rented(
                    offer, plan=plan, resources=resources, launch=launch, allocation=allocation
                )
            except OfferTaken:
                page = [row for row in page if row["id"] != offer["id"]]
                logger.warning(
                    "vast offer {} was taken before the create; picking again", offer["id"]
                )
        if not page:
            self.market.refuse(ask)
        raise MissionError(
            f"vast took {_PICK_ATTEMPTS} offers out from under the create in a row; the market "
            "is moving faster than a rental can be placed, try again in a minute"
        )

    def __rented(
        self,
        offer: Mapping,
        *,
        plan: ExecutionPlan,
        resources: Resources,
        launch: dict,
        allocation: Allocation,
    ) -> str:
        """Create the instance for `offer`, returning the contract id that starts the meter.

        A plan's own container image is rented when declared, else the newest base image `offer`'s
        driver loads; a plan with no image of its own is landed, and the landing minutes are
        billed into the lease.

        launch: the launch-mode fields, the `ssh` runtype's waiting onstart for a landing or the
            `args` entrypoint for a prebuilt image.
        """
        body = {
            "client_id": "me",
            "image": plan.container.image if plan.container is not None else base_image(offer),
            "disk": self.market.disk_gb,
            "label": allocation.label,
            # Fail the rent outright, rather than park a stopped instance we would still owe
            # storage on, when the offer is taken between the search and the create.
            "cancel_unavail": True,
            **launch,
            **({"price": bid(offer)} if self.market.spot else {}),
        }
        accepted = from_vast([dict(offer)], spot=self.market.spot)[0].model_copy(
            update={
                "source": f"probed:vast:offer:{offer['id']}",
                "rate_usd_hr": self.market.rate(offer),
            }
        )
        setup = 0 if plan.containerized else Rental.LANDING_SECONDS
        allocation.begin(lease=Lease.priced(accepted, resources, setup_s=setup))
        try:
            payload = self.api.created(offer, body)
        except HTTPError as refused:
            raise VastBackend.__refusal(refused, offer=offer, allocation=allocation) from refused
        return allocation.created(str(payload["new_contract"]))

    def cancel(self, handle: str) -> None:
        """Destroy the rental, the call that stops the meter, tolerating a forgotten instance."""
        try:
            reply = self.api.request("DELETE", path=f"/instances/{handle}/")
        except HTTPError as error:
            forgotten(error)
            return
        if reply.get("success") is not True:
            raise MissionError(
                f"vast did not confirm destruction of instance {handle}; "
                "release remains pending and billing may continue"
            )

    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        """A live offer search as catalog rows, the authed refresh of the imported price feed."""
        offers = self.market.search(Ask(gpu_name=gpu_name, gpus=gpus), limit=limit)
        return from_vast(offers, spot=self.market.spot)

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Where ssh reaches instance `handle`, waited for until vast publishes it and it runs.

        Vast pulls the image and starts the container before publishing the proxy address and
        port its ssh goes through; a machine that never gets there is refused by id.

        key: the private key file the connection uses, empty to leave that to ssh's own config.
        """
        for _ in range(ADDRESS_ATTEMPTS):
            instance = self.api.instance(handle)
            address = str(instance.get("ssh_host") or "")
            port = int(instance.get("ssh_port") or 0)
            if address and port and str(instance.get("actual_status") or "") == _RUNNING:
                return Endpoint(address=address, port=port, user=_SSH_USER, identity=key)
            self.api.sleeper(ADDRESS_SECONDS)
        raise MissionError(
            f"vast instance {handle} never came up with an ssh address; look it up at "
            "https://cloud.vast.ai/instances/ and destroy it if it is still billing"
        )

    def logs(self, handle: str) -> str:
        return self.api.logs(handle)

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """Rent a machine whose entrypoint seeds the key and waits for a landing.

        The landing minutes are billed, so they sit inside the ceiling the search filters on. A
        rental that cannot be opened is ended before the error leaves.
        """
        self.admit(plan, resources)
        self.market.spot = self.market.spot or resources.spot
        key = identity(plan.profile.vars.get("ssh-key", ""))
        script = f"{seeded(key.public)}\n{waiting()}\necho {_EXIT_SENTINEL}$status\nexit $status\n"
        handle = self.__placed(
            self.__ask(plan, resources, landing=Rental.LANDING_SECONDS),
            plan=plan,
            resources=resources,
            launch={"runtype": "ssh", "onstart": script},
            allocation=allocation,
        )
        try:
            endpoint = self.__opened(handle, key=key)
        except BaseException:
            logger.warning("vast instance {} could not be opened, ending the rental", handle)
            self.cancel(handle)
            raise
        return Rental(handle=handle, endpoint=endpoint)

    def rentals(self) -> list[Rented]:
        """Every instance on the account, whoever created it."""
        listed = self.api.request("GET", path="/instances/", query={"owner": "me"})
        return [
            Rented(
                handle=str(row.get("id") or ""),
                label=str(row.get("label") or ""),
                gpu=f"{row.get('num_gpus') or 1}x {row.get('gpu_name') or 'unknown card'}",
                status=str(row.get("actual_status") or ""),
                usd_hr=float(row["dph_total"]) if row.get("dph_total") is not None else None,
            )
            for row in listed.get("instances") or []
        ]

    def standing(self) -> Standing:
        """The account's credit and one live rate for the sample card, or the missing key.

        `/users/current` carries the spendable `credit` (its sibling `balance` is the invoicing
        figure, zero on a prepaid account) and one narrow search prices the market. Neither call
        happens without a key.
        """
        try:
            api_key()
        except MissionError as unset:
            return Standing(note=str(unset))
        credit = self.api.request("GET", path="/users/current/").get("credit")
        spendable = float(credit) if credit is not None else None
        sample = Ask(gpu_name=_SAMPLE_GPU, gpus=1)
        cheapest = next(iter(self.market.search(sample, limit=1)), None)
        if cheapest is None:
            return Standing(
                keyed=True, credit_usd=spendable, note=f"no 1x {_SAMPLE_GPU} offer right now"
            )
        # A machine whose city is unset carries its country as `, US`; trim the stray comma.
        where = str(cheapest.get("geolocation") or "").strip(" ,")
        return Standing(
            keyed=True,
            credit_usd=spendable,
            usd_hr=self.market.rate(cheapest),
            note=f"1x {_SAMPLE_GPU} {where}".strip(),
        )

    def state(self, handle: str) -> JobState:
        """`handle`'s verdict, from the command's own marker ahead of the container status.

        A restarted container reads `running` about a command that already ended, and believing
        it is how a sweep never reaches a terminal verdict, never cancels, and bills on (eight
        instances still billing after a campaign finished, $2.35 against $0.65 expected,
        2026-08-26). So once a container has been up, one log fetch per poll asks for the marker.
        """
        instance = self.api.instance(handle)
        if not instance:
            return JobState(handle=handle, verdict="vanished")
        status = str(instance.get("actual_status") or "")
        if status in _PENDING_STATUSES:
            return JobState(handle=handle, state=status, verdict="running")
        code = self.__exit_code(handle)
        if code is None:
            unfinished = "running" if status in _LIVE_STATUSES else "unknown"
            return JobState(handle=handle, state=status, verdict=unfinished)
        return JobState(
            handle=handle, state=status, exit_code=code, verdict="ok" if code == 0 else "failed"
        )

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        """Run `command` as the container's entrypoint, for a plan that brings its own image."""
        self.admit(plan, resources)
        self.market.spot = self.market.spot or resources.spot
        offer = self.market.pick(self.__ask(plan, resources, landing=0))
        # Receipts are staged before the command and framed after it, because the log is all that
        # leaves a rental and vast cuts every line at 500 characters.
        script = (
            f"{staging()}\n{command}\nstatus=$?\n{framing()}\n"
            f"echo {_EXIT_SENTINEL}$status\nexit $status\n"
        )
        # `args` launch mode runs the image as is, `onstart` its entrypoint and `args` its argv,
        # the official CLI's one-shot container.
        return self.__rented(
            offer,
            plan=plan,
            resources=resources,
            launch={"runtype": "args", "onstart": "bash", "args": ["-c", script]},
            allocation=allocation,
        )
