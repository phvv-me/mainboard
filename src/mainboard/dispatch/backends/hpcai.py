# `HpcAiBackend` runs a command on an HPC-AI (hpc-ai.com) instance through their REST API.
# Auth is a console API key sent as X-API-Key on every call (verified live), and every
# call refreshes it first when it has gone stale.
#
# Their published docs cover the instance and storage namespaces only, so `/balance` and
# `/resource/user/instance/list` are undocumented endpoints the same key authenticates, both
# verified live 2026-08-19: `/balance` carries `balance`, `availableBalance`,
# `availableVoucherAmount` and `availableCreditAmount`, and the resource listing is the priced,
# stock-aware catalog the console itself reads to fill its launch form.
#
# An instance is also an ssh box, and that is what a dispatch lands on. Their docs put the address
# at `instanceSpecInfo.regionInfo.sshAddress`, the port in the `instanceSpecInfo.nodePorts` entry
# for container port 22, and the login at `instanceMetadata.instanceUsername`, which is the
# `ssh -p <nodePort> <user>@<address>` line their console prints. Unlike vast there is no endpoint
# that attaches a key to a running instance, so the key the landing connects with is the one the
# account already registered in the console, and a machine that never answers says so by name.

import json
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from time import sleep
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import Request

from ...core.errors import MissionError
from ..evidence import framing, staging
from ..rentals import Identity, Rental, identity, reachable, waiting
from ..shared import logger
from ..transport import Endpoint
from ..vocabulary import JobState
from .base import (
    Account,
    Credentials,
    Delivery,
    LogSource,
    ProviderBackend,
    Rentable,
    Standing,
    forgotten,
    http_transport,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...context.plan import ExecutionPlan
    from ...manifest.schema.host import HostProfile
    from ..allocation import Allocation
    from ..vocabulary import Resources
    from .base import Transport

# Where an initScript's exit-code and output sentinels land. The instance's own data disk, since
# a rental with no `remoteStorages` entry has no mounted volume and one instance runs one command,
# so a fixed pair of names needs no handle in it.
_SENTINEL_DIR = "/root/dataDisk"
_LOG_PATH = f"{_SENTINEL_DIR}/mainboard.log"
_EXIT_PATH = f"{_SENTINEL_DIR}/mainboard.exit"
# What one `/instance/list` page carries. The endpoint refuses a request without a pager, so the
# page size is ours to pick and the walk below pages until the handle turns up.
_PAGE_SIZE = 50
# `instanceRuntimeInfo.status` mapped onto our verdict vocabulary, keyed in lower case since
# HPC-AI spells its states in camel case (`Running`, `StartingFailed`). The set is the one their
# list-instances doc publishes. A status outside this table (a new HPC-AI state) reads as
# "unknown" rather than crashing.
# Where an instance publishes the ssh endpoint a landing reaches it at, and the container port
# whose mapping carries it. `Running` is the one status that can answer ssh at all.
_SSH_PORT = 22
_RUNNING = "Running"
# How long to wait for an instance to pull its image and come up, fifteen minutes at this
# cadence.
_ADDRESS_ATTEMPTS = 90
_ADDRESS_SECONDS = 10.0

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


def api_key() -> str:
    """The HPC-AI key from `HPCAI_API_KEY`, refusing with the setup hint when unset.

    Console API keys authenticate the instance namespace through the `X-API-Key`
    header (verified live 2026-08-18), so no login flow or cookie JWT exists here. The workspace
    `.env` the refusal names is merged in first, so the hint below is advice this same function
    then acts on rather than a chore left to whoever reads it.
    """
    Credentials().load()
    key = os.environ.get("HPCAI_API_KEY", "")
    if not key:
        raise MissionError(
            "set HPCAI_API_KEY in the workspace .env, an API key minted in the HPC-AI console"
        )
    return key


def _required_var(profile: HostProfile, key: str) -> str:
    """`profile.vars[key]`, raising a clear `MissionError` naming `key` when it is missing.

    `HostProfile` forbids undeclared fields, so the opaque, provider-specific values HPC-AI
    needs (`instance-type-id`, `image-id`, `region`) live in its free-form `vars` table instead.
    """
    try:
        return profile.vars[key]
    except KeyError:
        raise MissionError(
            f"host {profile.root!r} needs [hosts.<name>.vars] {key!r} set for the hpc-ai backend"
        ) from None


def _mapped_port(rows: Sequence[Mapping]) -> int:
    """The host port an instance maps container port 22 to, 0 when it publishes none.

    A row names the container port under `port` and the port it is reachable at under
    `nodePort`, which is the pair their console's own ssh line is built from.
    """
    for row in rows:
        if int(row.get("port") or 0) == _SSH_PORT:
            return int(row.get("nodePort") or 0)
    return 0


class HpcAiBackend(ProviderBackend, Account, Rentable):
    """Run a command on an HPC-AI instance, its own REST API standing in for a scheduler.

    HPC-AI reports instance-level status only (`instanceRuntimeInfo.status`), never a process
    exit code, so `submit` wraps the command in an initScript that writes its real exit code and
    captured output to sentinel files on the instance's data disk; `state` reports whether the
    instance itself is still up, and the sentinel files are the source of truth for the command's
    own outcome.

    Those sentinel files are also why this is the one backend that carries neither `LogSource`
    nor `Delivery`: an instance's output never leaves its own disk, so there is no server-side log
    to fetch and nothing to deliver from here. Both gaps are declared in `lacks` with the path to
    read by hand instead.
    """

    name = "hpc-ai"

    lacks = {
        Delivery: "hpc-ai backend cannot deliver {path!r} yet; download "
        f"{_SENTINEL_DIR}/mainboard.* from instance {{handle}} over ssh until that path lands",
        LogSource: f"hpc-ai backend has no server-side logs; read {_LOG_PATH} on instance "
        "{handle} over ssh instead",
    }

    def __init__(
        self,
        *,
        spot: bool = False,
        transport: Transport = http_transport,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        """spot: whether created instances are spot (cost-optimized, preemptible).
        transport: sends a prepared `Request`, returning its response; injectable for tests.
        sleeper: waits between instance polls, injected so a test drives it without real time.
        """
        self.spot = spot
        self.transport = transport
        self.sleeper = sleeper

    def cancel(self, handle: str) -> None:
        """Stop the instance, then destroy it, so a cancelled run stops billing rather than idle.

        `/instance/terminate` is the destroy endpoint, verified live 2026-08-19. Their docs page
        for it is titled "delete" and its own cURL example calls `/instance/terminate`; the
        `/instance/delete` path the title implies answers 404.

        Both halves tolerate an instance that is already down, since every run the durable sweep
        settles is cancelled and the same run can be settled twice, by a pass killed before it
        advanced its cursor or by someone who ended the rental by hand. Stopping is preparation
        rather than the point, so its refusal never blocks the terminate that ends the billing,
        and a terminate the provider answers with a 404 has reached the state it was asked for.
        """
        with suppress(HTTPError):
            self.request("POST", path="/instance/stop", body={"instanceId": handle})
        try:
            self.request("POST", path="/instance/terminate", body={"instanceId": handle})
        except HTTPError as error:
            forgotten(error)

    def catalog(self) -> list[dict]:
        """Every rentable instance type, flattened to one row per type per region.

        The console's own launch-form feed, which is the only place HPC-AI publishes a price, a
        GPU name or a stock count. Rows carry `gpu`, `gpus`, `usd_hr`, `region`, `region_id`,
        `instance_type_id` and `in_stock`, which is exactly what a `[hosts.<name>.vars]` table
        needs, and are ordered cheapest first so the first in-stock row is the one to take.
        """
        payload = self.request("POST", path="/resource/user/instance/list", body={})
        rows = []
        for family in payload.get("instanceInfos") or []:
            rows += [
                {
                    "gpu": family.get("gpuName") or "",
                    "gpus": kind.get("gpuNum") or 0,
                    "usd_hr": HpcAiBackend._hourly(kind),
                    "region": region.get("regionName") or "",
                    "region_id": region.get("regionId") or "",
                    "instance_type_id": kind.get("instanceTypeId") or "",
                    "in_stock": kind.get("stockStatus") == "InStock",
                }
                for region in family.get("regionInfos") or []
                for kind in region.get("instanceTypeInfos") or []
            ]
        return sorted(
            rows,
            key=lambda row: (
                not row["in_stock"],
                row["usd_hr"] is None,
                row["usd_hr"] if row["usd_hr"] is not None else 0.0,
            ),
        )

    def instance(self, handle: str) -> dict:
        """`handle`'s listed instance row, empty once HPC-AI no longer lists it.

        `/instance/list` refuses a request that carries no pager (it answers 500), and it pages,
        so the walk asks for one page at a time and stops at the first page carrying the id or
        once it has run past the total the pager reports.
        """
        page = 1
        while True:
            payload = self.request(
                "POST",
                path="/instance/list",
                body={"pager": {"currentPage": page, "pageSize": _PAGE_SIZE}},
            )
            for item in payload.get("instances") or []:
                if item.get("instanceMetadata", {}).get("instanceId") == handle:
                    return item
            total = int((payload.get("pager") or {}).get("totalEntries") or 0)
            if page * _PAGE_SIZE >= total:
                return {}
            page += 1

    def create(self, plan: ExecutionPlan, *, script: str, allocation: Allocation) -> str:
        """Create an instance whose initScript runs `script`, returning HPC-AI's instance id.

        Every field their create endpoint calls required is sent, `billing` and `nodePorts`
        included, since the validator rejects a body that omits one rather than defaulting it.
        The returned handle is the provider's own `instanceId`, which is what `state`, `stop` and
        `delete` address the rental by; the `name` is ours and is never an address.

        HPC-AI reports instance status and never a process exit code, so `script` runs inside the
        one redirect that puts its output and whatever it leaves in `$status` into the sentinel
        pair on the instance's data disk, which is the only place a post-mortem can read either.

        plan: the resolved execution context, whose `vars` name the type, image and region.
        script: the initScript body, a command for a prebuilt image or the waiting entrypoint for
            a machine a dispatch lands on.
        """
        body = {
            "name": allocation.label,
            "isSpotInstance": self.spot,
            "instanceTypeId": _required_var(plan.profile, "instance-type-id"),
            "imageId": _required_var(plan.profile, "image-id"),
            "region": _required_var(plan.profile, "region"),
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
        payload = self.request("POST", path="/instance/create", body=body)
        return allocation.created(str(payload["instanceId"]))

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Where ssh reaches instance `handle`, waited for until it runs and publishes one.

        An instance is created before it can be logged into: the image is pulled, the container
        starts, and only a `Running` row carries the address, the mapped port and the login their
        console's own ssh line is built from. A machine that never gets there is a refusal naming
        the id, rather than a landing knocking at an address that does not exist yet.

        handle: the instance id the rental was created under.
        key: the private key file the connection uses, empty to leave that to ssh's own config.
        """
        for _ in range(_ADDRESS_ATTEMPTS):
            entry = self.instance(handle)
            spec = entry.get("instanceSpecInfo") or {}
            address = str((spec.get("regionInfo") or {}).get("sshAddress") or "")
            port = _mapped_port(spec.get("nodePorts") or [])
            running = str((entry.get("instanceRuntimeInfo") or {}).get("status") or "") == _RUNNING
            if running and address and port:
                user = str((entry.get("instanceMetadata") or {}).get("instanceUsername") or "")
                return Endpoint(address=address, port=port, user=user, identity=key)
            self.sleeper(_ADDRESS_SECONDS)
        raise MissionError(
            f"hpc-ai instance {handle} never published an ssh endpoint; look it up in the "
            "console and terminate it if it is still billing"
        )

    def opened(self, handle: str, *, key: Identity) -> Endpoint:
        """Answer once ssh lets us onto `handle`, naming the console key when it never does.

        HPC-AI attaches no key at create time, so the only way in is a key the account already
        registered, and a machine that answers nothing is almost always this workspace's key
        missing from that list rather than a machine that failed to boot.
        """
        try:
            return reachable(self.endpoint(handle, key=key.private), sleeper=self.sleeper)
        except MissionError as refused:
            raise MissionError(
                f"{refused}. Add the public half of {key.private} to the ssh keys in the HPC-AI "
                "console, since an instance is only reachable with a key the account registered "
                "before it was created."
            ) from None

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """Rent an instance that answers ssh and hold its initScript until a dispatch lands on it.

        The initScript waits rather than running the job, because the workspace, the tool and the
        environment reach the box minutes after it boots and a command that starts before them
        has nothing to run with. A rental that cannot be opened is ended here rather than left to
        the initScript's own deadline, since this is the last place that still holds the handle.
        """
        self.admit(plan, resources)
        key = identity(plan.profile.vars.get("ssh-key", ""))
        handle = self.create(plan, script=f"{waiting()}\n", allocation=allocation)
        opened = False
        try:
            endpoint = self.opened(handle, key=key)
            opened = True
        finally:
            if not opened:
                logger.warning("hpc-ai instance %s could not be opened, ending the rental", handle)
                self.cancel(handle)
        return Rental(handle=handle, endpoint=endpoint)

    def request(self, method: str, *, path: str, body: dict) -> dict:
        """An authenticated call to the instance API under the console API key.

        Every endpoint hangs off `https://www.hpc-ai.com/api`, spelled inline so the https root
        is visible at the one place a `Request` is built.
        """
        headers = {"Content-Type": "application/json", "X-API-Key": api_key()}
        request = Request(
            f"https://www.hpc-ai.com/api{path}",
            method=method,
            data=json.dumps(body).encode(),
            headers=headers,
        )
        return json.loads(self.transport(request).read())

    def standing(self) -> Standing:
        """The account balance HPC-AI reports, or the key that is missing.

        `/balance` answers the same console key every other call here carries, and it is the one
        surface of the three providers that publishes a number without a rental in hand. It
        returns four figures, and the row takes `balance`, the pot that is left once vouchers and
        monthly credits have been spent first, which is the order their billing doc states. The
        provider quotes no price without an instance type in hand, so no rate rides along.
        """
        try:
            api_key()
        except MissionError as unset:
            return Standing(note=str(unset))
        payload = self.request("GET", path="/balance", body={})
        return Standing(keyed=True, credit_usd=float(payload["balance"]))

    def state(self, handle: str) -> JobState:
        entry = self.instance(handle)
        if not entry:
            return JobState(handle=handle, verdict="vanished")
        status = str(entry.get("instanceRuntimeInfo", {}).get("status") or "")
        return JobState(
            handle=handle, state=status, verdict=_VERDICTS.get(status.lower(), "unknown")
        )

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        """Run `command` as the instance's own initScript, for a plan that brings its own image.

        The raw-command shape, and the one case it is still right for: a prebuilt image already
        holds everything its command needs, so there is nothing for a landing to install. Every
        other plan rents through `rent`, because a bare image has no workspace, no tool and no
        environment to run anything with.
        """
        self.admit(plan, resources)
        # The command's own status is captured before anything else runs, since the framing
        # below would otherwise be what `$?` reports. Receipts are framed into the same captured
        # log the sentinel pair already writes, so whoever reads that file by hand gets the
        # trials as well as the output.
        return self.create(
            plan, script=f"{staging()}\n{command}\nstatus=$?\n{framing()}\n", allocation=allocation
        )

    @staticmethod
    def _hourly(kind: Mapping) -> float | None:
        """The on-demand hourly rate, None when the provider publishes no hourly price.

        A type carries one price entry per charge mode (`perHour`, `perDay`, the tide-priced
        `tidePerHour`), and only the plain hourly one is comparable across types.
        """
        for price in kind.get("price") or []:
            if price.get("chargeMode") == "perHour":
                value = price.get("price")
                return float(value) if value is not None else None
        return None
