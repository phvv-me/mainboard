# `VastBackend` rents a Vast.ai machine for one command through their REST API, under a console
# API key sent as `Authorization: Bearer` on every call.
#
# The plan decides the rental's shape, not a flag. A plan naming no container of its own gets
# `rent`: the `ssh` runtype with the waiting entrypoint, then the ordinary mirror, install,
# provision and pin path. A plan declaring its own image keeps `submit`, the raw command as the
# container's entrypoint in `args` launch mode, since a prebuilt image is the one case where the
# box already holds everything the command needs, and a one-shot container is cheaper and simpler.

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

# The images an uncontainerized plan rents under, Vast's base image at each CUDA minor the house
# toolchain spans, oldest first. A rental takes the newest its offer's driver loads, so THE OFFER
# FILTER'S FLOOR IS THE OLDEST REFERENCE HERE and written nowhere else. One pinned image failed
# both ways: 13.3.1 with the floor at 13.0 rented hosts whose driver tops out at 13.0, where the
# container never started and each rental was destroyed after the whole address wait (RTX 5090
# 51865823, RTX 5080 51869485 and 51891087, 2026-09-21); the floor raised to 13.3 then shut every
# L40S, A100 and H100 host (13.0 to 13.2 drivers) out of the market (2026-09-25). The workspace
# itself declares CUDA 13.0 (`[system]`).
_BASE_IMAGES = (
    "vastai/base-image:cuda-13.0.3-auto",
    "vastai/base-image:cuda-13.1.2-auto",
    "vastai/base-image:cuda-13.2.1-auto",
    "vastai/base-image:cuda-13.3.1-auto",
)
# Bundle-row fields every offer publishes, so the floors filter before renting: the highest CUDA
# the driver loads, the compute capability (`750` for `sm_75`), and the measured download in Mbps.
_CUDA_FIELD = "cuda_max_good"
_CAPABILITY_FIELD = "compute_cap"
_DOWNLOAD_FIELD = "inet_down"
# Offers one rental may try when the market takes each between search and create, which Vast
# answers with this token (RTX 5090 offer 26371154, 2026-09-12).
_PICK_ATTEMPTS = 3
_NO_SUCH_ASK = "no_such_ask"
# Local disk per rental in GB, also what a search prices storage at, so quote and machine agree.
# The mirror is a few hundred MB but the environment is a whole CUDA stack plus its package cache,
# tens of GB; storage is cents per GB-month, while running out of disk costs the whole job.
_DISK_GB = 64.0
# Offers one search asks for; the query already orders by price, so this bounds the reply size.
_SEARCH_LIMIT = 32
# Echoed after the command with its exit code, the only place a verdict learns how the command
# ended, since Vast reports container status only.
_EXIT_SENTINEL = "mainboard-exit:"
_LOG_TAIL_LINES = 2000
# `request_logs` answers before the log reaches storage, so the fetch is retried before the url is
# handed to the caller instead.
_LOG_ATTEMPTS = 20
_LOG_POLL_SECONDS = 1.0
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
# Fifteen minutes to pull the image, start the container and publish the proxy ssh address: a
# cold CUDA base image is gigabytes, and a machine still pulling must not be knocked at yet.
_ADDRESS_ATTEMPTS = 90
_ADDRESS_SECONDS = 10.0


def exit_sentinel(log: str) -> int | None:
    """The exit status the wrapper echoed into `log`, None when no marker is readable.

    The last marker wins, since a container Vast restarted appends its own below the first, and a
    marker without a number is skipped rather than read as a zero.
    """
    for line in reversed(log.splitlines()):
        _, marked, status = line.partition(_EXIT_SENTINEL)
        if marked:
            with suppress(ValueError):
                return int(status.strip())
    return None


def cuda_max_good(offer: Mapping) -> float:
    """The highest CUDA version `offer`'s driver loads, 0.0 when unpublished.

    Silence never satisfies a floor, here or in `capability` and `download`: an unknown driver is
    exactly the offer that hands back a contract id and no instance.
    """
    try:
        return float(offer[_CUDA_FIELD])
    except KeyError, TypeError, ValueError:
        return 0.0


def base_image(offer: Mapping) -> str:
    """The newest base image `offer`'s driver loads, the oldest when it publishes no version."""
    loadable = [
        image for image in _BASE_IMAGES if (image_cuda(image) or 0) <= cuda_max_good(offer)
    ]
    return loadable[-1] if loadable else _BASE_IMAGES[0]


class OfferTaken(MissionError):
    """The offer a create named was rented by someone else between the search and the create."""

    def __init__(self, identifier: int | str) -> None:
        super().__init__(f"vast offer {identifier} was taken before the create landed")


def download(offer: Mapping) -> float:
    """The host's download rate in Mbps, 0 when unpublished."""
    try:
        return float(offer.get(_DOWNLOAD_FIELD) or 0.0)
    except TypeError, ValueError:
        return 0.0


def capability(offer: Mapping) -> int:
    """`offer`'s compute capability, `750` for `sm_75`, 0 when unpublished."""
    try:
        return int(offer[_CAPABILITY_FIELD])
    except KeyError, TypeError, ValueError:
        return 0


def _describe(offer: Mapping, gpu_name: str) -> str:
    """`offer` as the phrase a refusal names it by, id first so it can be looked up.

    gpu_name: the card asked for, standing in when the row names none.
    """
    where = str(offer.get("geolocation") or "").strip(" ,")
    card = offer.get("gpu_name") or gpu_name or "unknown card"
    return f"offer {offer.get('id')} ({card}, {where or 'unknown location'})"


def api_key() -> str:
    """The Vast key from `VAST_API_KEY` or `VASTAI_API_KEY`, refusing with a setup hint when unset.

    A console key authenticates the whole v0 namespace as a Bearer header, as their CLI sends it.
    gpuhunt reads `VASTAI_API_KEY` while Vast's CLI documents `VAST_API_KEY`, so both are read.
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

    # Derived from the oldest base image, never below the house floor, so the filter and the
    # images cannot disagree again.
    CUDA_FLOOR: ClassVar[float] = max(
        ProviderBackend.CUDA_FLOOR, min(image_cuda(image) or 0 for image in _BASE_IMAGES)
    )

    # A cold rental pulls a multi-gigabyte image before its container exists: hosts below this
    # spent the whole address wait pulling and ended with no container (three rentals,
    # 2026-09-12), while hosts at a few Gbps were running inside two minutes.
    DOWNLOAD_FLOOR_MBPS: ClassVar[float] = 500.0

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
        self.spot = spot
        self.disk_gb = disk_gb
        self.transport = transport
        self.sleeper = sleeper

    @staticmethod
    def hourly_cap(resources: Resources, *, landing: int = 0) -> float:
        """The hourly ceiling `resources` implies, 0 for a walltime-less request.

        A spend cap bounds an hourly rental only once the job says how long it runs; without a
        walltime the search takes the whole market and leans on `max_usd` alone.

        landing: seconds billed before the job starts (mirror, tool install, environment), 0 for
            a prebuilt container that runs the command the moment it boots.
        """
        seconds = walltime_seconds(resources.walltime) if resources.walltime else 0
        return resources.max_usd * 3600.0 / (seconds + landing) if seconds else 0.0

    def attach(self, handle: str, *, key: str) -> None:
        """Put this workspace's public key on the rental, so the landing can log in.

        Vast copies an account key onto a new instance itself; this adds the one key this
        machine holds the private half of. A refusal is fatal: nobody could log into the rental.
        """
        try:
            self.request("POST", path=f"/instances/{handle}/ssh/", body={"ssh_key": key})
        except HTTPError as refused:
            raise MissionError(
                f"vast refused the ssh key for instance {handle} ({refused}); add the key at "
                "https://cloud.vast.ai/manage-keys/ and submit again"
            ) from refused

    def cancel(self, handle: str) -> None:
        """Destroy the rental, the call that stops the meter, tolerating a forgotten instance."""
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
        """A live offer search as catalog rows, the authed refresh of the imported price feed."""
        return from_vast(
            self.search(gpu_name=gpu_name, gpus=gpus, limit=limit or _SEARCH_LIMIT),
            spot=self.spot,
        )

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Where ssh reaches instance `handle`, waited for until vast publishes it and it runs.

        Vast pulls the image and starts the container before publishing the proxy address and
        port its ssh goes through; a machine that never gets there is refused by id.

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
        """`handle`'s process exit status from the sentinel in its log tail.

        None when the log cannot be fetched or carries no marker, so a clean container stop is
        never read as a clean run.
        """
        try:
            log = self.logs(handle)
        except HTTPError, MissionError:
            return None
        return exit_sentinel(log)

    def instance(self, handle: str) -> dict:
        """`handle`'s instance row, empty once Vast has forgotten it (a null row or a 404)."""
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

        Renting the cheapest listing put earlier rentals at the bottom of the market, where the
        container is billed for and never starts, so price only admits: the pick is the most
        reliable machine on the cheapest page under the cap, landing mid-market.

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

        An empty page reads the same for an empty market and one this house aged out of (the
        latter cost five dispatches), so one more search, unfloored and paid on this path alone,
        tells them apart. The driver is asked first, being the earlier failure.

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
        """Rent a machine whose entrypoint seeds the key and waits for a landing.

        The landing minutes are billed, so they sit inside the ceiling the search filters on. An
        offer taken before the create is retried from the same page, never a fresh search.
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

        The key goes on once there is a running instance to copy it into, and the knocking that
        follows absorbs the seconds it takes to reach the container's authorized keys.
        """
        endpoint = self.endpoint(handle, key=key.private)
        self.attach(handle, key=key.public)
        return reachable(endpoint, sleeper=self.sleeper)

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

        plan: its own container image is rented when declared, else the newest base image
            `offer`'s driver loads.
        launch: the launch-mode fields, the `ssh` runtype's waiting onstart for a landing or the
            `args` entrypoint for a prebuilt image.
        """
        body = {
            "client_id": "me",
            "image": plan.container.image if plan.container is not None else base_image(offer),
            "disk": self.disk_gb,
            "label": allocation.label,
            # Fail the rent outright, rather than park a stopped instance we would still owe
            # storage on, when the offer is taken between the search and the create.
            "cancel_unavail": True,
            **launch,
        }
        if self.spot:
            body["price"] = float(offer["min_bid"])
        accepted = from_vast([dict(offer)], spot=self.spot)[0].model_copy(
            update={"source": f"probed:vast:offer:{offer['id']}"}
        )
        allocation.begin(lease=Lease.priced(accepted, resources, setup_s=setup_s))
        try:
            payload = self.request("PUT", path=f"/asks/{offer['id']}/", body=body)
        except HTTPError as refused:
            try:
                detail = json.load(refused)
            except ValueError:
                detail = {}
            reason = detail.get("msg") or detail.get("message") or detail.get("error") or ""
            if 400 <= refused.code < 500:
                # Declined after validation, so nothing was created and the reservation closes.
                allocation.refused()
            if _NO_SUCH_ASK in str(reason):
                raise OfferTaken(offer["id"]) from refused
            raise MissionError(
                f"vast refused offer {offer['id']} (HTTP {refused.code}): {str(reason)[:400]}"
            ) from refused
        return allocation.created(str(payload["new_contract"]))

    def rentals(self) -> list[Rented]:
        """Every instance on the account, whoever created it."""
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
        """An authenticated call below `https://console.vast.ai/api/v0`, their CLI's default root.

        path: the endpoint below the v0 root, trailing slash included.
        body: the JSON payload, an empty object when the endpoint takes none.
        """
        tail = f"{path}?{urlencode(query)}" if query else path
        request = Request(
            f"https://console.vast.ai/api/v0{tail}",
            method=method,
            data=json.dumps(body or {}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
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

        The query is a table of `{field: {operator: value}}` constraints. The four constant
        filters are the ones their console applies, keeping unverified hosts, resold capacity and
        rented machines out. Vast ranks by on-demand total in either mode, so a spot search is
        re-ranked by the bid floor it will pay. The floors ride on the wire, since the reply is
        paged by price and filtering afterwards would judge the market on its cheapest 32 rows.

        gpu_name: the Vast GPU name (`RTX 4090`), underscores read as spaces, empty for any.
        max_usd_hr: an hourly total-price ceiling, 0 for none.
        floored: whether the house floors ride on the query; only `refuse` drops them, to ask the
            raw market what the floors turned away.
        cuda: the CUDA the image about to be rented names, raising the driver floor above the
            base images'.
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
        # The service has returned rows above the requested ceiling, and the quoted rate, not a
        # filter the request carried, decides whether spending is authorized.
        if max_usd_hr:
            offers = [offer for offer in offers if self.rate(offer) <= max_usd_hr]
        return sorted(offers, key=self.rate)

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
        credit = self.request("GET", path="/users/current/").get("credit")
        spendable = float(credit) if credit is not None else None
        cheapest = next(iter(self.search(gpu_name=_SAMPLE_GPU, gpus=1, limit=1)), None)
        if cheapest is None:
            return Standing(
                keyed=True, credit_usd=spendable, note=f"no 1x {_SAMPLE_GPU} offer right now"
            )
        # A machine whose city is unset carries its country as `, US`; trim the stray comma.
        where = str(cheapest.get("geolocation") or "").strip(" ,")
        return Standing(
            keyed=True,
            credit_usd=spendable,
            usd_hr=self.rate(cheapest),
            note=f"1x {_SAMPLE_GPU} {where}".strip(),
        )

    def state(self, handle: str) -> JobState:
        """`handle`'s verdict, from the command's own marker ahead of the container status.

        A restarted container reads `running` about a command that already ended, and believing
        it is how a sweep never reaches a terminal verdict, never cancels, and bills on (eight
        instances still billing after a campaign finished, $2.35 against $0.65 expected,
        2026-08-26). So once a container has been up, one log fetch per poll asks for the marker.
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
            handle=handle, state=status, exit_code=code, verdict="ok" if code == 0 else "failed"
        )

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        """Run `command` as the container's entrypoint, for a plan that brings its own image."""
        self.admit(plan, resources)
        offer = self.pick(
            gpu_name=resources.gpu_name,
            gpus=max(resources.gpus, 1),
            max_usd_hr=self.hourly_cap(resources),
            cuda=self.floor(plan),
        )
        # Receipts are staged before the command and framed after it, because the log is all that
        # leaves a rental and vast cuts every line at 500 characters.
        script = (
            f"{staging()}\n{command}\nstatus=$?\n{framing()}\n"
            f"echo {_EXIT_SENTINEL}$status\nexit $status\n"
        )
        # `args` launch mode runs the image as is, `onstart` its entrypoint and `args` its argv,
        # the official CLI's one-shot container.
        return self.rented(
            offer,
            plan=plan,
            launch={"runtype": "args", "onstart": "bash", "args": ["-c", script]},
            allocation=allocation,
            resources=resources,
        )

    def uploaded(self, url: str) -> str:
        """The log Vast uploaded at `url`, polled while the upload is in flight.

        The first fetches 404 until the log lands. The url is storage's own signed link, so it
        must be https and is fetched with no API key attached.
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
