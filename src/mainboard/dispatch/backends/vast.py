# `VastBackend` rents a Vast.ai machine for one command through their REST API. Auth is a console
# API key sent as `Authorization: Bearer` on every call, and the transport is the same injected
# callable the other pure-REST backend uses, so no test ever reaches the network.

import json
import os
from contextlib import suppress
from time import sleep
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

from ...core.errors import MissionError
from ...costs.imports import from_vast
from ..jobs.spec import walltime_seconds
from ..schedulers.base import JobState
from .base import (
    Account,
    Delivery,
    LogSource,
    Market,
    ProviderBackend,
    Standing,
    http_transport,
    require_budget,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...context.plan import ExecutionPlan
    from ...costs.catalog import Offer
    from ..schedulers.base import Resources
    from .base import Transport

# The image an uncontainerized plan rents under: Vast's own base image, whose `-auto` tag resolves
# to the CUDA build matching the host's driver.
_DEFAULT_IMAGE = "vastai/base-image:cuda-12.9.2-auto"
# Local disk per rental, in GB. It is also what an offer search prices storage at, so one number
# keeps the quoted rate and the rented machine honest about each other.
_DISK_GB = 16.0
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
# `actual_status` values that mean the container has not finished yet, and the ones Vast's own
# docs call terminal. A status in neither set (a new Vast state) reads as "unknown" rather than
# crashing, and never costs a log fetch.
_LIVE_STATUSES = frozenset({"created", "loading", "running", "stopping"})
_TERMINAL_STATUSES = frozenset({"exited", "stopped", "offline", "error"})
# The card a price sample quotes. One card, always listed in volume, so the sample reads as a
# real market rate rather than a quote for hardware nobody rents today.
_SAMPLE_GPU = "RTX 4090"


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


def api_key() -> str:
    """The Vast key from `VAST_API_KEY` or `VASTAI_API_KEY`, refusing with a setup hint when unset.

    Console API keys authenticate the whole v0 namespace through the `Authorization: Bearer`
    header, which is what their own CLI sends, so no login flow or cookie exists here. Both
    spellings are accepted because gpuhunt reads `VASTAI_API_KEY` while Vast's CLI documents
    `VAST_API_KEY`.
    """
    key = os.environ.get("VAST_API_KEY", "") or os.environ.get("VASTAI_API_KEY", "")
    if not key:
        raise MissionError(
            "set VAST_API_KEY (or VASTAI_API_KEY) in the workspace .env, an API key minted at "
            "https://cloud.vast.ai/manage-keys/"
        )
    return key


class VastBackend(ProviderBackend, Account, LogSource, Market):
    """Rent a Vast.ai machine for one command, the container's own lifetime being the job's.

    Vast rents whole containers rather than running jobs, so `submit` picks a rentable offer
    matching the request and creates the instance in `args` launch mode with the command as its
    entrypoint. Since Vast reports container status and never a process exit code, the wrapper
    echoes an exit sentinel into the log, and `state` reads it back through `logs` once the
    container is terminal, so a verdict describes the command rather than the container that
    happened to stop cleanly around it.

    A finished command does not end the rental. Vast keeps an instance at its `intended_status`,
    so it restarts the exited container and the command runs again, appending another sentinel
    (verified live 2026-08-19, thirteen restarts in five minutes). `state` reading the last marker
    is what keeps the verdict right through that, and `cancel` is what actually stops the meter,
    so a caller that reaches a terminal verdict must still cancel.

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

    def cancel(self, handle: str) -> None:
        self.request("DELETE", path=f"/instances/{handle}/")

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

    def state(self, handle: str) -> JobState:
        instance = self.instance(handle)
        if not instance:
            return JobState(handle=handle, verdict="vanished")
        status = str(instance.get("actual_status") or "")
        if status in _LIVE_STATUSES:
            return JobState(handle=handle, state=status, verdict="running")
        if status not in _TERMINAL_STATUSES:
            return JobState(handle=handle, state=status, verdict="unknown")
        code = self.exit_code(handle)
        if code is None:
            return JobState(handle=handle, state=status, verdict="unknown")
        return JobState(
            handle=handle,
            state=status,
            exit_code=code,
            verdict="ok" if code == 0 else "failed",
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

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        require_budget(resources)
        offer = self.pick(
            gpu_name=resources.gpu_name,
            gpus=max(resources.gpus, 1),
            max_usd_hr=self.hourly_cap(resources),
        )
        script = f"{command}\nstatus=$?\necho {_EXIT_SENTINEL}$status\nexit $status\n"
        body = {
            "client_id": "me",
            "image": plan.container.image if plan.containerized else _DEFAULT_IMAGE,
            "disk": self.disk_gb,
            "label": f"mainboard-{plan.host}",
            # `args` launch mode runs the image as it is, with `onstart` as the entrypoint and
            # `args` as its argv, which is how the official CLI spells a one-shot container.
            "runtype": "args",
            "onstart": "bash",
            "args": ["-c", script],
            # Fail the rent outright rather than parking a stopped instance we would still owe
            # storage on when the offer is taken between the search and the create.
            "cancel_unavail": True,
        }
        if self.spot:
            body["price"] = float(offer["min_bid"])
        payload = self.request("PUT", path=f"/asks/{offer['id']}/", body=body)
        return str(payload["new_contract"])

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

    def pick(self, *, gpu_name: str, gpus: int, max_usd_hr: float = 0.0) -> dict:
        """The offer to rent: the most reliable machine the budget already allows.

        Renting the lowest-priced listing is what put earlier rentals at the bottom of the
        market, where the container is billed for and never starts, so price decides admission
        here and nothing more. The search returns the cheapest page of what fits under the cap
        the caller's own budget implies, and the pick is the highest measured host reliability on
        that page, ties going to the cheaper machine, which lands mid-market rather than at
        either end. Refuses when the market has no matching offer at all.

        gpu_name: the Vast GPU name the job needs, empty for any.
        gpus: the GPU count per machine.
        max_usd_hr: an hourly ceiling the offer must sit under, 0 for none.
        """
        offers = self.search(gpu_name=gpu_name, gpus=gpus, max_usd_hr=max_usd_hr)
        if not offers:
            ceiling = f" under ${max_usd_hr:.2f}/hr" if max_usd_hr else ""
            raise MissionError(
                f"vast has no rentable {gpus}x {gpu_name or 'any'} offer{ceiling} right now"
            )
        return max(offers, key=lambda offer: (float(offer["reliability2"]), -self.rate(offer)))

    @staticmethod
    def hourly_cap(resources: Resources) -> float:
        """The hourly ceiling `resources` implies, 0 when the request leaves the job open-ended.

        A spend cap only bounds an hourly rental once the job also says how long it may run, so a
        walltime-less request searches the whole market and leans on `max_usd` alone.
        """
        seconds = walltime_seconds(resources.walltime) if resources.walltime else 0
        return resources.max_usd * 3600.0 / seconds if seconds else 0.0

    def instance(self, handle: str) -> dict:
        """`handle`'s instance row, empty once Vast has forgotten the instance.

        A destroyed instance answers either a null row or a 404 depending on how long ago it went,
        and a post-mortem reads both the same way, so both come back empty here.
        """
        try:
            payload = self.request("GET", path=f"/instances/{handle}/", query={"owner": "me"})
        except HTTPError as error:
            if error.status != 404:
                raise
            return {}
        return payload.get("instances") or {}

    def rate(self, offer: dict) -> float:
        """What one hour of `offer` costs under this backend's pricing mode."""
        return float(offer["min_bid"] if self.spot else offer["dph_total"])

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
    ) -> list[dict]:
        """Rentable offers matching the filters, cheapest first under this backend's pricing mode.

        Vast's offer query is a table of `{field: {operator: value}}` constraints posted as the
        request body. The four constant filters are the ones their console applies to every
        search, keeping unverified hosts, resold capacity and already-rented machines out. Vast
        ranks by the on-demand total whichever mode is asked for, so a spot search is re-ranked
        here by the bid floor it will actually pay.

        gpu_name: the Vast GPU name (`RTX 4090`), underscores read as spaces, empty for any.
        gpus: the GPU count per machine, 0 for any.
        max_usd_hr: an hourly total-price ceiling, 0 for none.
        limit: how many offers to ask for.
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
        offers = self.request("POST", path="/bundles/", body=query).get("offers") or []
        return sorted(offers, key=self.rate)

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
