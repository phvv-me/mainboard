# `VastBackend` rents a Vast.ai machine for one command through their REST API. Auth is a console
# API key sent as `Authorization: Bearer` on every call, and the transport is the same injected
# callable the other pure-REST backend uses, so no test ever reaches the network.
#
# A rental comes in two shapes, and which one a dispatch gets is decided by the plan rather than
# by a flag. A plan that names no container of its own rents a machine this workspace lands on:
# `rent` creates it in the `ssh` runtype with the waiting entrypoint, and the ordinary mirror,
# install, provision and pin path then puts the workspace, the tool and the environment on it
# before the job starts. A plan that declares its own image keeps `submit`, which runs the raw
# command as the container's entrypoint in `args` launch mode, because a prebuilt image is the one
# case where the box already holds everything the command needs.

import json
import os
from contextlib import suppress
from time import sleep
from typing import TYPE_CHECKING, ClassVar, NoReturn
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

from ...core.errors import MissionError
from ...costs.imports import from_vast
from ...runtime.job import walltime_seconds
from ..evidence import framing, staging
from ..lease import Lease
from ..rentals import LANDING_SECONDS, Identity, Rental, identity, reachable, seeded, waiting
from ..shared import logger
from ..transport import Endpoint
from ..vocabulary import JobState
from .base import (
    Account,
    Credentials,
    Delivery,
    Inventory,
    LogSource,
    Market,
    ProviderBackend,
    Rentable,
    Rented,
    Standing,
    forgotten,
    http_transport,
    image_cuda,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ...context.plan import ExecutionPlan
    from ...costs.catalog import Offer
    from ..allocation import Allocation
    from ..vocabulary import Resources
    from .base import Transport

# The images an uncontainerized plan rents under: Vast's own base image at each CUDA minor the
# house toolchain spans, oldest first. A rental takes the newest one its offer's driver can load,
# so THE OFFER FILTER'S FLOOR IS THE OLDEST REFERENCE HERE and written nowhere else. One pinned
# image failed both ways: at 13.3.1 with the floor left at 13.0 it rented hosts whose driver tops
# out at 13.0, where the container never started and each rental was destroyed after the whole
# address wait (RTX 5090 51865823 and RTX 5080 51869485 and 51891087, 2026-09-21), and with the
# floor then raised to 13.3 it shut every L40S, A100 and H100 host out of the market, since those
# run 13.0 to 13.2 drivers (2026-09-25). The workspace itself declares CUDA 13.0 (`[system]`).
_BASE_IMAGES = (
    "vastai/base-image:cuda-13.0.3-auto",
    "vastai/base-image:cuda-13.1.2-auto",
    "vastai/base-image:cuda-13.2.1-auto",
    "vastai/base-image:cuda-13.3.1-auto",
)
# The two offer fields a rental has to clear, the highest CUDA version a machine's driver can
# load and the card's own compute capability (`750` for `sm_75`). Vast publishes both on every
# bundle row, which is what makes them filterable before renting rather than after.
_CUDA_FIELD = "cuda_max_good"
# The host's measured download rate, in Mbps, as the bundle row publishes it. A cold rental pulls
# a multi-gigabyte image before its container exists, and a host below this floor spent the whole
# address wait pulling and ended with no container at all (three rentals, 2026-09-12), while hosts
# at a few Gbps were running inside two minutes.
_DOWNLOAD_FIELD = "inet_down"
# How many offers one rental may try when the market takes each one between the search and
# the create, which Vast answers with this token (RTX 5090 offer 26371154, 2026-09-12).
_PICK_ATTEMPTS = 3
_NO_SUCH_ASK = "no_such_ask"
_DOWNLOAD_FLOOR_MBPS = 500.0
_CAPABILITY_FIELD = "compute_cap"
# Local disk per rental, in GB. It is also what an offer search prices storage at, so one number
# keeps the quoted rate and the rented machine honest about each other. Sized for what a landing
# actually puts on the box rather than for the workspace alone: the mirror is a few hundred
# megabytes, and the environment installed beside it is a whole CUDA stack plus the package cache
# it was linked from, which is tens of gigabytes. Storage is cents a month per gigabyte, so the
# headroom costs a rounding error per hour and a rental that runs out of disk costs the whole job.
_DISK_GB = 64.0
# How many offers one search asks for. The query already orders by price, so this only bounds the
# reply size a `catalog` refresh has to carry.
_SEARCH_LIMIT = 32
# The marker the wrapper echoes after the command, carrying its real exit code into the log.
# Vast reports container status only, never a process exit code, so this line is the only place
# a verdict can learn how the command itself ended.
_EXIT_SENTINEL = "mainboard-exit:"
# Log lines one `request_logs` upload carries back.
_LOG_TAIL_LINES = 2000
# Vast answers `request_logs` before the log itself reaches storage, so the fetch is retried this
# many times, this many seconds apart, before the url is handed to the caller instead.
_LOG_ATTEMPTS = 20
_LOG_POLL_SECONDS = 1.0
# `actual_status` values that mean the container has not run the command yet, so no marker can
# exist and asking for a log costs the upload poll for nothing.
_PENDING_STATUSES = frozenset({"created", "loading"})
# `actual_status` values that mean the container is still up. A status in neither this set nor
# `_PENDING_STATUSES` (a new Vast state, or a row carrying none) reads as "unknown" once the log
# has failed to say how the command ended, rather than crashing.
_LIVE_STATUSES = frozenset({"created", "loading", "running", "stopping"})
# The card a price sample quotes. One card, always listed in volume, so the sample reads as a
# real market rate rather than a quote for hardware nobody rents today.
_SAMPLE_GPU = "RTX 4090"
# The login every vast image hands out, and the `actual_status` a machine reaches before its ssh
# daemon can answer at all.
_SSH_USER = "root"
_RUNNING = "running"
# How long to wait for vast to pull the image, start the container and publish the proxy address
# its ssh goes through. Fifteen minutes, because a cold CUDA base image is gigabytes and a machine
# still pulling one is exactly the machine a landing must not knock at yet.
_ADDRESS_ATTEMPTS = 90
_ADDRESS_SECONDS = 10.0


def exit_sentinel(log: str) -> int | None:
    """The exit status the onstart wrapper echoed into `log`, None when no marker is readable.

    The last marker wins, since a container Vast restarted appends its own line below the first,
    and a line that carries the marker without a number is skipped rather than read as a zero.

    log: the container log tail as `logs` fetched it.
    """
    for line in reversed(log.splitlines()):
        _, marked, status = line.partition(_EXIT_SENTINEL)
        if marked:
            with suppress(ValueError):
                return int(status.strip())
    return None


def cuda_max_good(offer: Mapping) -> float:
    """The highest CUDA version `offer`'s driver can load, 0.0 when the row publishes none.

    Silence never satisfies a floor, here or in `capability`: an offer whose driver version is
    unknown is exactly the offer that hands back a contract id and no instance.

    offer: one bundle row as the offer search returned it.
    """
    try:
        return float(offer[_CUDA_FIELD])
    except KeyError, TypeError, ValueError:
        return 0.0


def base_image(offer: Mapping) -> str:
    """The newest base image `offer`'s driver can load, the oldest when it publishes no version.

    offer: one bundle row as the offer search returned it, already past the CUDA floor.
    """
    loadable = [
        image for image in _BASE_IMAGES if (image_cuda(image) or 0) <= cuda_max_good(offer)
    ]
    return loadable[-1] if loadable else _BASE_IMAGES[0]


class OfferTaken(MissionError):
    """The offer a create named was rented by someone else between the search and the create."""

    def __init__(self, identifier: int | str) -> None:
        super().__init__(f"vast offer {identifier} was taken before the create landed")


def download(offer: Mapping) -> float:
    """The host's download rate in Mbps, 0 for a row that publishes none.

    offer: one bundle row as the offer search returned it.
    """
    try:
        return float(offer.get(_DOWNLOAD_FIELD) or 0.0)
    except TypeError, ValueError:
        return 0.0


def capability(offer: Mapping) -> int:
    """`offer`'s compute capability the way a provider spells it (`750` is `sm_75`), 0 for none.

    offer: one bundle row as the offer search returned it.
    """
    try:
        return int(offer[_CAPABILITY_FIELD])
    except KeyError, TypeError, ValueError:
        return 0


def _describe(offer: Mapping, gpu_name: str) -> str:
    """`offer` as one human phrase a refusal names it by, id first so it can be looked up.

    offer: the bundle row the refusal is about.
    gpu_name: the card the caller asked for, standing in when the row names none.
    """
    where = str(offer.get("geolocation") or "").strip(" ,")
    card = offer.get("gpu_name") or gpu_name or "unknown card"
    return f"offer {offer.get('id')} ({card}, {where or 'unknown location'})"


def api_key() -> str:
    """The Vast key from `VAST_API_KEY` or `VASTAI_API_KEY`, refusing with a setup hint when unset.

    Console API keys authenticate the whole v0 namespace through the `Authorization: Bearer`
    header, which is what their own CLI sends, so no login flow or cookie exists here. Both
    spellings are accepted because gpuhunt reads `VASTAI_API_KEY` while Vast's CLI documents
    `VAST_API_KEY`. The workspace `.env` the refusal names is merged in first, so the hint below
    is advice this same function then acts on rather than a chore left to whoever reads it.
    """
    Credentials().load()
    key = os.environ.get("VAST_API_KEY", "") or os.environ.get("VASTAI_API_KEY", "")
    if not key:
        raise MissionError(
            "set VAST_API_KEY (or VASTAI_API_KEY) in the workspace .env, an API key minted at "
            "https://cloud.vast.ai/manage-keys/"
        )
    return key


class VastBackend(ProviderBackend, Account, Inventory, LogSource, Market, Rentable):
    """Rent a Vast.ai machine for one command, the container's own lifetime being the job's.

    Vast rents whole containers rather than running jobs, so `submit` picks a rentable offer
    matching the request and creates the instance in `args` launch mode with the command as its
    entrypoint. Since Vast reports container status and never a process exit code, the wrapper
    echoes an exit sentinel into the log, and `state` reads it back through `logs` once the
    container is terminal, so a verdict describes the command rather than the container that
    happened to stop cleanly around it.

    A finished command does not end the rental. Vast keeps an instance at its `intended_status`,
    so it restarts the exited container and the command runs again, appending another sentinel
    (verified live 2026-08-19, thirteen restarts in five minutes). That is why `state` asks the
    log for a marker before it reads the container's status at all: a restarted container says
    `running` about a command that already ended, and only the marker knows better. `cancel` is
    then what actually stops the meter, so a caller that reaches a terminal verdict must still
    cancel, which is exactly what the durable sweep does with one.

    Stateless between calls: every method addresses an instance by the id `submit` returned.

    It is the one backend that quotes a market, since renting is what it does, and the one that
    cannot deliver an artifact, since the disk it wrote to is destroyed with the rental.
    """

    name = "vast"

    lacks = {
        Delivery: "vast backend cannot deliver {path!r} yet; a rented machine's disk dies with "
        "the instance, so have the command upload its own results and read `logs {handle}` "
        "until that path lands",
    }

    # What a host's driver must reach to load the oldest base image, never below the house
    # floor. Derived, so the filter and the images cannot disagree again.
    CUDA_FLOOR: ClassVar[float] = max(
        ProviderBackend.CUDA_FLOOR, min(image_cuda(image) or 0 for image in _BASE_IMAGES)
    )

    # A host below this download rate spends the address wait pulling the image; see the module
    # note beside `_DOWNLOAD_FLOOR_MBPS`.
    DOWNLOAD_FLOOR_MBPS: ClassVar[float] = _DOWNLOAD_FLOOR_MBPS

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
        sleeper: waits between log-upload polls, injected so a test drives it without real time.
        """
        self.spot = spot
        self.disk_gb = disk_gb
        self.transport = transport
        self.sleeper = sleeper

    @staticmethod
    def hourly_cap(resources: Resources, *, landing: int = 0) -> float:
        """The hourly ceiling `resources` implies, 0 when the request leaves the job open-ended.

        A spend cap only bounds an hourly rental once the job also says how long it may run, so a
        walltime-less request searches the whole market and leans on `max_usd` alone.

        landing: seconds the rental bills before its job starts, which on a machine this
            workspace lands on is the mirror, the tool install and the environment; 0 for a
            prebuilt container that runs the command the moment it boots.
        """
        seconds = walltime_seconds(resources.walltime) if resources.walltime else 0
        return resources.max_usd * 3600.0 / (seconds + landing) if seconds else 0.0

    def attach(self, handle: str, *, key: str) -> None:
        """Put this workspace's public key on the rental, so the landing can log in.

        Vast copies an account key onto a new instance on its own, and this says it again for the
        one key this machine actually holds the private half of, which is the only key a landing
        can use. A refusal here is fatal on purpose rather than warned about: an instance nobody
        can log into is a rental that will bill for a landing that can never happen.
        """
        try:
            self.request("POST", path=f"/instances/{handle}/ssh/", body={"ssh_key": key})
        except HTTPError as refused:
            raise MissionError(
                f"vast refused the ssh key for instance {handle} ({refused}); add the key at "
                "https://cloud.vast.ai/manage-keys/ and submit again"
            ) from refused

    def cancel(self, handle: str) -> None:
        """Destroy the rental, tolerating an instance Vast has already forgotten.

        The call that actually stops the meter, since a finished command leaves the rental up.
        It is asked more than once by design, by a sweep that settles the same run twice and by
        anyone who already destroyed the instance in the console, so the 404 a gone instance
        answers is this method's own destination rather than a fault to raise from.
        """
        try:
            reply = self.request("DELETE", path=f"/instances/{handle}/")
        except HTTPError as error:
            forgotten(error)
            return
        if reply.get("success") is not True:
            raise MissionError(
                f"vast did not confirm destruction of instance {handle}; "
                "release remains pending and billing may continue"
            )

    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        """A live offer search as catalog rows, the authed refresh of the imported price feed.

        gpu_name: the Vast GPU name to narrow to, empty for the whole market.
        gpus: the GPU count per machine, 0 for any.
        limit: how many offers to bring back, 0 for this backend's own page size.
        """
        return from_vast(
            self.search(gpu_name=gpu_name, gpus=gpus, limit=limit or _SEARCH_LIMIT),
            spot=self.spot,
        )

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Where ssh reaches instance `handle`, waited for until vast publishes it and it runs.

        A rental is created long before it is reachable: vast pulls the image, starts the
        container, and only then publishes the proxy address and port its own ssh goes through.
        So this reads the instance row until all three are true, and a machine that never gets
        there is a refusal naming the id to look up rather than a landing knocking at an address
        that does not exist yet.

        handle: the contract id the rental was created under.
        key: the private key file the connection uses, empty to leave that to ssh's own config.
        """
        for _ in range(_ADDRESS_ATTEMPTS):
            instance = self.instance(handle)
            address = str(instance.get("ssh_host") or "")
            port = int(instance.get("ssh_port") or 0)
            if address and port and str(instance.get("actual_status") or "") == _RUNNING:
                return Endpoint(address=address, port=port, user=_SSH_USER, identity=key)
            self.sleeper(_ADDRESS_SECONDS)
        raise MissionError(
            f"vast instance {handle} never came up with an ssh address; look it up at "
            "https://cloud.vast.ai/instances/ and destroy it if it is still billing"
        )

    def exit_code(self, handle: str) -> int | None:
        """`handle`'s real process exit status, read from the sentinel in its log tail.

        A container status says only that the container stopped, never why, so the verdict comes
        from the marker the onstart wrapper echoed after the command. None when the log cannot be
        fetched or carries no marker, which keeps an unknown verdict honest instead of reading a
        clean container stop as a clean run.
        """
        try:
            log = self.logs(handle)
        except HTTPError, MissionError:
            return None
        return exit_sentinel(log)

    def instance(self, handle: str) -> dict:
        """`handle`'s instance row, empty once Vast has forgotten the instance.

        A destroyed instance answers either a null row or a 404 depending on how long ago it went,
        and a post-mortem reads both the same way, so both come back empty here.
        """
        try:
            payload = self.request("GET", path=f"/instances/{handle}/", query={"owner": "me"})
        except HTTPError as error:
            return forgotten(error)
        return payload.get("instances") or {}

    def logs(self, handle: str) -> str:
        payload = self.request(
            "PUT",
            path=f"/instances/request_logs/{handle}/",
            body={"tail": str(_LOG_TAIL_LINES)},
        )
        url = str(payload.get("result_url") or "")
        if not url:
            raise MissionError(
                f"vast refused logs for instance {handle}: {payload.get('msg') or payload}"
            )
        return self.uploaded(url)

    def pick(
        self, *, gpu_name: str, gpus: int, max_usd_hr: float = 0.0, cuda: float = 0.0
    ) -> dict:
        """The offer to rent: the most reliable machine the budget and the CUDA floors allow.

        Renting the lowest-priced listing is what put earlier rentals at the bottom of the
        market, where the container is billed for and never starts, so price decides admission
        here and nothing more. The search returns the cheapest page of what fits under the cap
        the caller's own budget implies and above the house floors, and the pick is the highest
        measured host reliability on that page, ties going to the cheaper machine, which lands
        mid-market rather than at either end.

        An empty page is handed to `refuse` rather than reported as a bare absence, since the
        three reasons a page can be empty read identically otherwise and the middle one, a card
        whose every host runs too old a driver, is what cost five dispatches.

        gpu_name: the Vast GPU name the job needs, empty for any.
        gpus: the GPU count per machine.
        max_usd_hr: an hourly ceiling the offer must sit under, 0 for none.
        cuda: the driver version the image about to be rented needs, 0 for the base images'.
        """
        offers = self.search(gpu_name=gpu_name, gpus=gpus, max_usd_hr=max_usd_hr, cuda=cuda)
        if not offers:
            self.refuse(gpu_name=gpu_name, gpus=gpus, max_usd_hr=max_usd_hr, floor=cuda)
        return self.best(offers)

    def best(self, offers: Sequence[Mapping]) -> dict:
        """The most reliable offer of a page, ties going to the cheaper machine."""
        chosen = max(offers, key=lambda offer: (float(offer["reliability2"]), -self.rate(offer)))
        return dict(chosen)

    def rate(self, offer: Mapping) -> float:
        """What one hour of `offer` costs under this backend's pricing mode."""
        return float(offer["min_bid"] if self.spot else offer["dph_total"])

    def refuse(
        self, *, gpu_name: str, gpus: int, max_usd_hr: float, floor: float = 0.0
    ) -> NoReturn:
        """Say why nothing was rentable, naming whichever floor turned the market away.

        Reached only once the floored search came back empty, and it spends one more search,
        the same one unfloored, to tell an empty market from a market this house has aged out
        of. That round trip is paid on the refusal path alone. The driver question is asked
        first because it is the earlier of the two failures, the instance that never starts.

        gpu_name: the Vast GPU name the job asked for, empty for any.
        gpus: the GPU count per machine.
        max_usd_hr: the hourly ceiling the search ran under, 0 for none.
        floor: the driver floor the search ran under, 0 for this backend's own.
        """
        floor = max(self.CUDA_FLOOR, floor)
        ceiling = f" under ${max_usd_hr:.2f}/hr" if max_usd_hr else ""
        card = f"{gpus}x {gpu_name or 'any'}"
        raw = self.search(gpu_name=gpu_name, gpus=gpus, max_usd_hr=max_usd_hr, floored=False)
        if not raw:
            raise MissionError(f"vast has no rentable {card} offer{ceiling} right now")
        loadable = [row for row in raw if cuda_max_good(row) >= floor]
        if not loadable:
            best = max(raw, key=cuda_max_good)
            raise MissionError(
                f"vast has {len(raw)} rentable {card} offer(s){ceiling} and not one driver "
                f"reaches CUDA {floor}, the floor this rental needs. The best is "
                f"{_describe(best, gpu_name)} at CUDA {cuda_max_good(best)}. Renting it would "
                "hand back a contract id and no instance, because vast destroys a container its "
                "driver cannot start. Ask for a card whose hosts run a newer driver."
            )
        buildable = [row for row in loadable if capability(row) >= self.CAPABILITY_FLOOR]
        if not buildable:
            best = max(loadable, key=capability)
            raise MissionError(
                f"vast has {len(loadable)} rentable {card} offer(s){ceiling} whose driver "
                f"reaches CUDA {floor}, and not one is an architecture that CUDA still "
                f"builds for. The best is {_describe(best, gpu_name)} at compute capability "
                f"{capability(best)}, below the sm_{self.CAPABILITY_FLOOR // 10} floor. Renting "
                "it would boot, bill, and die at the first kernel launch with no kernel image "
                "for its own card. Maxwell, Pascal and Volta went with it; ask for Turing or "
                "newer."
            )
        best = max(buildable, key=download)
        raise MissionError(
            f"vast has {len(buildable)} rentable {card} offer(s){ceiling} this house could run "
            f"on, and not one host downloads at {self.DOWNLOAD_FLOOR_MBPS:.0f} Mbps or more. The "
            f"best is {_describe(best, gpu_name)} at {download(best):.0f} Mbps. Renting it would "
            "spend the whole landing window pulling the image and end with no container, which "
            "is how three rentals went on 2026-09-12; raise the ceiling to reach a faster host."
        )

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """Rent a machine that answers ssh and hold its entrypoint until a dispatch lands on it.

        The entrypoint waits rather than running the job, because the workspace, the tool and the
        environment reach the box minutes after it boots and a command that starts before them
        finds nothing to run (exit 127, three rentals, 2026-09-03). Those minutes are billed, so
        they sit inside the ceiling the offer search filters on rather than outside anyone's
        budget.

        A rental that cannot be opened is ended here rather than left to the entrypoint's own
        deadline, since this is the last place that still holds the handle.
        """
        self.admit(plan, resources)
        key = identity(plan.profile.vars.get("ssh-key", ""))
        marker = f"echo {_EXIT_SENTINEL}$status\nexit $status\n"
        script = f"{seeded(key.public)}\n{waiting()}\n{marker}"
        gpus = max(resources.gpus, 1)
        cap = self.hourly_cap(resources, landing=LANDING_SECONDS)
        floor = self.floor(plan)
        page = self.search(gpu_name=resources.gpu_name, gpus=gpus, max_usd_hr=cap, cuda=floor)
        if not page:
            self.refuse(gpu_name=resources.gpu_name, gpus=gpus, max_usd_hr=cap, floor=floor)
        for _ in range(_PICK_ATTEMPTS):
            offer = self.best(page)
            try:
                handle = self.rented(
                    offer,
                    plan=plan,
                    launch={"runtype": "ssh", "onstart": script},
                    allocation=allocation,
                    resources=resources,
                    setup_s=LANDING_SECONDS,
                )
                break
            except OfferTaken:
                # The market moved between the search and the create; the next best offer on
                # the same page is asked for, and the reservation is reopened for it.
                page = [row for row in page if row["id"] != offer["id"]]
                logger.warning(
                    "vast offer %s was taken before the create; picking again", offer["id"]
                )
                if not page:
                    self.refuse(
                        gpu_name=resources.gpu_name, gpus=gpus, max_usd_hr=cap, floor=floor
                    )
        else:
            raise MissionError(
                f"vast took {_PICK_ATTEMPTS} offers out from under the create in a row; the "
                "market is moving faster than a rental can be placed, try again in a minute"
            )
        opened = False
        try:
            endpoint = self.opened(handle, key=key)
            opened = True
        finally:
            if not opened:
                logger.warning("vast instance %s could not be opened, ending the rental", handle)
                self.cancel(handle)
        return Rental(handle=handle, endpoint=endpoint)

    def opened(self, handle: str, *, key: Identity) -> Endpoint:
        """Answer once ssh really lets us onto `handle`, its key attached as soon as it is up.

        The key goes on after the machine is running rather than at create time, because that is
        when there is an instance to copy it into, and the knocking that follows is what absorbs
        the seconds it takes to reach the container's own authorized keys.
        """
        endpoint = self.endpoint(handle, key=key.private)
        self.attach(handle, key=key.public)
        return reachable(endpoint, sleeper=self.sleeper)

    @staticmethod
    def image(plan: ExecutionPlan, offer: Mapping) -> str:
        """The image `plan` rents on `offer`: its container's, else the newest base it loads."""
        return plan.container.image if plan.container is not None else base_image(offer)

    def floor(self, plan: ExecutionPlan) -> float:
        """The driver version `plan`'s rental needs: its own image's CUDA, else the base floor."""
        own = image_cuda(plan.container.image) if plan.container is not None else None
        return max(self.CUDA_FLOOR, own or 0.0)

    def rented(
        self,
        offer: Mapping,
        *,
        plan: ExecutionPlan,
        launch: dict,
        allocation: Allocation,
        resources: Resources,
        setup_s: int = 0,
    ) -> str:
        """Create the instance for `offer`, returning the contract id that starts the meter.

        offer: the bundle row `pick` chose.
        plan: the resolved execution context, whose own container image is rented when it
            declares one and the newest base image `offer`'s driver loads otherwise.
        launch: the launch-mode fields, either the `ssh` runtype's waiting onstart script for a
            machine a dispatch lands on, or the `args` entrypoint for a prebuilt image.
        """
        body = {
            "client_id": "me",
            "image": self.image(plan, offer),
            "disk": self.disk_gb,
            "label": allocation.label,
            # Fail the rent outright rather than parking a stopped instance we would still owe
            # storage on when the offer is taken between the search and the create.
            "cancel_unavail": True,
            **launch,
        }
        if self.spot:
            body["price"] = float(offer["min_bid"])
        accepted = from_vast([dict(offer)], spot=self.spot)[0].model_copy(
            update={"source": f"probed:vast:offer:{offer['id']}"}
        )
        lease = Lease.priced(accepted, resources, setup_s=setup_s)
        allocation.begin(lease=lease)
        try:
            payload = self.request("PUT", path=f"/asks/{offer['id']}/", body=body)
        except HTTPError as refused:
            try:
                detail = json.load(refused)
            except ValueError:
                detail = {}
            reason = detail.get("msg") or detail.get("message") or detail.get("error") or ""
            if 400 <= refused.code < 500:
                # The provider validated the request and declined it, so nothing was created
                # and the reservation can close on its own.
                allocation.refused()
            if _NO_SUCH_ASK in str(reason):
                raise OfferTaken(offer["id"]) from refused
            raise MissionError(
                f"vast refused offer {offer['id']} (HTTP {refused.code}): {str(reason)[:400]}"
            ) from refused
        return allocation.created(str(payload["new_contract"]))

    def rentals(self) -> list[Rented]:
        """Every instance on the account, as vast lists them, whoever created it."""
        listed = self.request("GET", path="/instances/", query={"owner": "me"})
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

    def request(
        self, method: str, *, path: str, body: dict | None = None, query: dict | None = None
    ) -> dict:
        """An authenticated call to the v0 API under the console API key.

        Every endpoint hangs off `https://console.vast.ai/api/v0`, the root their own CLI
        defaults to, spelled inline so the https root is visible where the `Request` is built.

        method: the HTTP verb Vast expects for this endpoint.
        path: the endpoint path below the v0 root, trailing slash included.
        body: the JSON payload, an empty object when the endpoint takes none.
        query: query-string parameters, when the endpoint reads any.
        """
        tail = f"{path}?{urlencode(query)}" if query else path
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"}
        request = Request(
            f"https://console.vast.ai/api/v0{tail}",
            method=method,
            data=json.dumps(body or {}).encode(),
            headers=headers,
        )
        return json.loads(self.transport(request).read())

    def search(
        self,
        *,
        gpu_name: str = "",
        gpus: int = 0,
        max_usd_hr: float = 0.0,
        limit: int = _SEARCH_LIMIT,
        floored: bool = True,
        cuda: float = 0.0,
    ) -> list[dict]:
        """Rentable offers matching the filters, cheapest first under this backend's pricing mode.

        Vast's offer query is a table of `{field: {operator: value}}` constraints posted as the
        request body. The four constant filters are the ones their console applies to every
        search, keeping unverified hosts, resold capacity and already-rented machines out. Vast
        ranks by the on-demand total whichever mode is asked for, so a spot search is re-ranked
        here by the bid floor it will actually pay.

        Both CUDA floors ride on the wire rather than being applied to the reply, because the
        query is paged and ordered by price: filtering afterwards would judge the market on the
        cheapest thirty-two rows and refuse a card whose usable offers were simply further down.
        Every caller gets them without asking, which is what makes the market this backend
        quotes, prices and rents from one market rather than three.

        gpu_name: the Vast GPU name (`RTX 4090`), underscores read as spaces, empty for any.
        gpus: the GPU count per machine, 0 for any.
        max_usd_hr: an hourly total-price ceiling, 0 for none.
        limit: how many offers to ask for.
        floored: whether the house CUDA floors ride on the query. Only `refuse` drops them, to
            ask the raw market what the floors turned away.
        cuda: the CUDA version the image about to be rented names, which raises the driver
            floor for this search when a plan brings an image newer than the base images.
        """
        query: dict = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "type": "bid" if self.spot else "on-demand",
            "order": [["dph_total", "asc"]],
            "allocated_storage": self.disk_gb,
            "limit": limit,
        }
        if gpu_name:
            query["gpu_name"] = {"eq": gpu_name.replace("_", " ")}
        if gpus:
            query["num_gpus"] = {"eq": gpus}
        if max_usd_hr:
            query["dph_total"] = {"lte": max_usd_hr}
        if floored:
            query[_CUDA_FIELD] = {"gte": max(self.CUDA_FLOOR, cuda)}
            query[_CAPABILITY_FIELD] = {"gte": self.CAPABILITY_FLOOR}
            query[_DOWNLOAD_FIELD] = {"gte": self.DOWNLOAD_FLOOR_MBPS}
        offers = self.request("POST", path="/bundles/", body=query).get("offers") or []
        # The service has returned rows above its requested ceiling. The quoted rate,
        # not successful submission of a filter, decides whether spending is authorized.
        if max_usd_hr:
            offers = [offer for offer in offers if self.rate(offer) <= max_usd_hr]
        return sorted(offers, key=self.rate)

    def standing(self) -> Standing:
        """The account's credit and one live rate for the sample card, or the key that is missing.

        Vast is the provider that answers both halves cheaply: `/users/current` carries the
        spendable `credit` for the authed user (its sibling `balance` is the invoicing figure,
        which sits at zero on a prepaid account), and one narrow offer search prices the market
        as it stands. Neither call happens until a key is found, so an unconfigured Vast row
        costs nothing but the environment lookup.
        """
        try:
            api_key()
        except MissionError as unset:
            return Standing(note=str(unset))
        credit = self.request("GET", path="/users/current/").get("credit")
        spendable = float(credit) if credit is not None else None
        cheapest = next(iter(self.search(gpu_name=_SAMPLE_GPU, gpus=1, limit=1)), None)
        if cheapest is None:
            return Standing(
                keyed=True, credit_usd=spendable, note=f"no 1x {_SAMPLE_GPU} offer right now"
            )
        # A Vast machine whose city is unset still carries its country as `, US`, so the
        # separator is trimmed along with the whitespace rather than printed as a stray comma.
        where = str(cheapest.get("geolocation") or "").strip(" ,")
        return Standing(
            keyed=True,
            credit_usd=spendable,
            usd_hr=self.rate(cheapest),
            note=f"1x {_SAMPLE_GPU} {where}".strip(),
        )

    def state(self, handle: str) -> JobState:
        """`handle`'s verdict, taken from the command's own marker ahead of the container status.

        The marker decides and the container only says whether to keep waiting, because the two
        disagree by design. Vast holds an instance at its intended status, so it restarts the
        container the command exited from and runs the command again, which means a run that
        finished cleanly reads `running` on nearly every poll after it ended. Believing that
        status is how a sweep never reaches a terminal verdict, never cancels, and lets the meter
        run on work that was already over (eight instances still billing after a campaign had
        finished, $2.35 against $0.65 expected, 2026-08-26). So a log carrying the wrapper's
        marker is a command that has already ended, whatever the container is doing now.

        The marker therefore costs one log fetch per poll of a container that has been up, which
        is the only signal a rented machine gives that its work is over, and is not asked of a
        container that has not started one yet.
        """
        instance = self.instance(handle)
        if not instance:
            return JobState(handle=handle, verdict="vanished")
        status = str(instance.get("actual_status") or "")
        if status in _PENDING_STATUSES:
            return JobState(handle=handle, state=status, verdict="running")
        code = self.exit_code(handle)
        if code is None:
            unfinished = "running" if status in _LIVE_STATUSES else "unknown"
            return JobState(handle=handle, state=status, verdict=unfinished)
        return JobState(
            handle=handle,
            state=status,
            exit_code=code,
            verdict="ok" if code == 0 else "failed",
        )

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        """Run `command` as the container's own entrypoint, for a plan that brings its own image.

        The raw-command shape, and the one case it is still right for: a prebuilt image already
        holds everything its command needs, so there is nothing for a landing to install and a
        one-shot container is both cheaper and simpler. Every other plan rents through `rent`,
        because a bare image has no workspace, no tool and no environment to run anything with.
        """
        self.admit(plan, resources)
        offer = self.pick(
            gpu_name=resources.gpu_name,
            gpus=max(resources.gpus, 1),
            max_usd_hr=self.hourly_cap(resources),
            cuda=self.floor(plan),
        )
        # Receipts are staged before the command and framed after it, in that order, because the
        # log is the only thing that leaves a rental and vast cuts every line of it at 500
        # characters. A trial that printed its receipt straight out would arrive here in half.
        script = (
            f"{staging()}\n{command}\nstatus=$?\n{framing()}\n"
            f"echo {_EXIT_SENTINEL}$status\nexit $status\n"
        )
        # `args` launch mode runs the image as it is, with `onstart` as the entrypoint and `args`
        # as its argv, which is how the official CLI spells a one-shot container.
        return self.rented(
            offer,
            plan=plan,
            launch={"runtype": "args", "onstart": "bash", "args": ["-c", script]},
            allocation=allocation,
            resources=resources,
        )

    def uploaded(self, url: str) -> str:
        """The log body Vast uploaded at `url`, polled while the upload is still in flight.

        `request_logs` answers before the log reaches storage, so the first fetches come back 404
        until it lands. The url is storage's own signed link rather than ours, so it is checked
        for an https scheme and then fetched with no API key attached.
        """
        host = url.removeprefix("https://")
        if host == url:
            raise MissionError(f"vast answered with a non-https log url {url!r}")
        for _ in range(_LOG_ATTEMPTS):
            try:
                return self.transport(Request(f"https://{host}")).read().decode()
            except HTTPError:
                self.sleeper(_LOG_POLL_SECONDS)
        raise MissionError(f"vast has not uploaded that log yet; fetch {url} directly")
