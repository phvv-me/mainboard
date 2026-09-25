# `HpcAiBackend` runs a command on an HPC-AI (hpc-ai.com) instance through their REST API, under
# a console API key sent as `X-API-Key` on every call (verified live 2026-08-18).
#
# Their docs cover the instance and storage namespaces only; `/balance` and
# `/resource/user/instance/list` are undocumented endpoints the same key authenticates (verified
# live 2026-08-19): `/balance` carries `balance`, `availableBalance`, `availableVoucherAmount` and
# `availableCreditAmount`, and the resource listing is the priced, stock-aware catalog the console
# fills its launch form from.
#
# A dispatch lands on an instance over ssh, at `instanceSpecInfo.regionInfo.sshAddress`, the
# `instanceSpecInfo.nodePorts` entry for container port 22 and the login
# `instanceMetadata.instanceUsername`: the `ssh -p <nodePort> <user>@<address>` line their console
# prints. Unlike vast, no endpoint attaches a key to a running instance, so the landing connects
# with a key the account already registered in the console.

import json
import os
from collections.abc import Iterator, Mapping, Sequence
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
    Inventory,
    LogSource,
    ProviderBackend,
    Rentable,
    Rented,
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

# The initScript's output and exit-code sentinels, on the instance's own data disk: a rental with
# no `remoteStorages` has no mounted volume, and one instance runs one command, so fixed names do.
_SENTINEL_DIR = "/root/dataDisk"
_LOG_PATH = f"{_SENTINEL_DIR}/mainboard.log"
_EXIT_PATH = f"{_SENTINEL_DIR}/mainboard.exit"
# `/instance/list` refuses a request without a pager, so the page size is ours.
_PAGE_SIZE = 50
_SSH_PORT = 22
# The one status that can answer ssh at all.
_RUNNING = "Running"
# Fifteen minutes for an instance to pull its image and come up.
_ADDRESS_ATTEMPTS = 90
_ADDRESS_SECONDS = 10.0
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


def api_key() -> str:
    """The HPC-AI key from `HPCAI_API_KEY` (no login flow or cookie JWT), refusing when unset."""
    Credentials().load()
    key = os.environ.get("HPCAI_API_KEY", "")
    if not key:
        raise MissionError(
            "set HPCAI_API_KEY in the workspace .env, an API key minted in the HPC-AI console"
        )
    return key


def _required_var(profile: HostProfile, key: str) -> str:
    """`profile.vars[key]`, refusing by name when missing.

    `HostProfile` forbids undeclared fields, so HPC-AI's opaque ids (`instance-type-id`,
    `image-id`, `region`) live in its free-form `vars` table.
    """
    try:
        return profile.vars[key]
    except KeyError:
        raise MissionError(
            f"host {profile.root!r} needs [hosts.<name>.vars] {key!r} set for the hpc-ai backend"
        ) from None


def _mapped_port(rows: Sequence[Mapping]) -> int:
    """The `nodePort` an instance maps container `port` 22 to, 0 when it publishes none."""
    return next(
        (int(row.get("nodePort") or 0) for row in rows if int(row.get("port") or 0) == _SSH_PORT),
        0,
    )


def _rented(item: Mapping) -> Rented:
    """One listed instance row as the rental it is."""
    metadata = item.get("instanceMetadata") or {}
    return Rented(
        handle=str(metadata.get("instanceId") or ""),
        label=str(metadata.get("instanceName") or metadata.get("name") or ""),
        status=str((item.get("instanceRuntimeInfo") or {}).get("status") or ""),
    )


class HpcAiBackend(ProviderBackend, Account, Inventory, Rentable):
    """Run a command on an HPC-AI instance, its own REST API standing in for a scheduler.

    HPC-AI reports instance status only, never a process exit code, so the initScript writes the
    command's exit code and output to sentinel files on the data disk: `state` says whether the
    instance is up, the sentinels how the command ended. Output never leaves that disk, so there
    is no `LogSource` or `Delivery`; `lacks` names the path to read by hand.
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
        """Stop the instance, then destroy it, so a cancelled run stops billing rather than idles.

        `/instance/terminate` destroys (verified live 2026-08-19): their docs page is titled
        "delete" but its cURL calls terminate, and `/instance/delete` answers 404. A refused stop
        never blocks the terminate that ends the billing.
        """
        with suppress(HTTPError):
            self.request("POST", path="/instance/stop", body={"instanceId": handle})
        try:
            self.request("POST", path="/instance/terminate", body={"instanceId": handle})
        except HTTPError as error:
            forgotten(error)

    def catalog(self) -> list[dict]:
        """Every rentable instance type, one row per type per region, in stock and cheapest first.

        The console's launch-form feed is the only place HPC-AI publishes a price, a GPU name or
        a stock count, and each row carries what a `[hosts.<name>.vars]` table needs.
        """
        payload = self.request("POST", path="/resource/user/instance/list", body={})
        rows = [
            {
                "gpu": family.get("gpuName") or "",
                "gpus": kind.get("gpuNum") or 0,
                "usd_hr": HpcAiBackend._hourly(kind),
                "region": region.get("regionName") or "",
                "region_id": region.get("regionId") or "",
                "instance_type_id": kind.get("instanceTypeId") or "",
                "in_stock": kind.get("stockStatus") == "InStock",
            }
            for family in payload.get("instanceInfos") or []
            for region in family.get("regionInfos") or []
            for kind in region.get("instanceTypeInfos") or []
        ]
        return sorted(
            rows,
            key=lambda row: (not row["in_stock"], row["usd_hr"] is None, row["usd_hr"] or 0.0),
        )

    def instance(self, handle: str) -> dict:
        """`handle`'s listed row, empty once unlisted; the walk stops at the page carrying it."""
        return next(
            (
                item
                for item in self.instances()
                if item.get("instanceMetadata", {}).get("instanceId") == handle
            ),
            {},
        )

    def instances(self) -> Iterator[dict]:
        """Every instance on the account, one listing page at a time.

        `/instance/list` answers 500 to a request without a pager, so the walk stops once it has
        run past the total the pager reports.
        """
        page = 1
        while True:
            payload = self.request(
                "POST",
                path="/instance/list",
                body={"pager": {"currentPage": page, "pageSize": _PAGE_SIZE}},
            )
            yield from payload.get("instances") or []
            total = int((payload.get("pager") or {}).get("totalEntries") or 0)
            if page * _PAGE_SIZE >= total:
                return
            page += 1

    def rentals(self) -> list[Rented]:
        """Every instance on the account, whoever created it."""
        return [_rented(item) for item in self.instances()]

    def create(self, plan: ExecutionPlan, *, script: str, allocation: Allocation) -> str:
        """Create an instance whose initScript runs `script`, returning HPC-AI's `instanceId`.

        Every field their create endpoint calls required is sent (`billing`, `nodePorts`), since
        the validator rejects rather than defaults an omission. The `name` is ours and never an
        address. `script` runs inside one redirect into the sentinel pair, followed by whatever
        it leaves in `$status`, the only place a post-mortem can read either.

        plan: whose `vars` name the type, image and region.
        script: a command for a prebuilt image, or the waiting entrypoint for a landing.
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

        Only a `Running` row carries the address, mapped port and login, after the image is
        pulled and the container started; one that never gets there is refused by id.

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

        A machine that answers nothing is almost always this workspace's key missing from the
        account's list, since HPC-AI attaches no key at create time.
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
        """Rent an instance whose initScript waits for a landing; end it if it cannot be opened."""
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
        """A call below `https://www.hpc-ai.com/api` (spelled inline), the key read afresh."""
        request = Request(
            f"https://www.hpc-ai.com/api{path}",
            method=method,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": api_key()},
        )
        return json.loads(self.transport(request).read())

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
        """Run `command` as the initScript, for a prebuilt image that needs nothing landed."""
        self.admit(plan, resources)
        # `status=$?` comes first, else the framing's own status is what `$?` reports. The
        # receipts are framed into the same captured log, so a reader of that file gets both.
        return self.create(
            plan, script=f"{staging()}\n{command}\nstatus=$?\n{framing()}\n", allocation=allocation
        )

    @staticmethod
    def _hourly(kind: Mapping) -> float | None:
        """The `perHour` rate, the one comparable across types (vs `perDay`, `tidePerHour`)."""
        for price in kind.get("price") or []:
            if price.get("chargeMode") == "perHour":
                value = price.get("price")
                return float(value) if value is not None else None
        return None
