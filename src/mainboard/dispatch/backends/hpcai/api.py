# HPC-AI's REST API (hpc-ai.com), under a console API key sent as `X-API-Key` on every call
# (verified live 2026-08-18).
#
# Their docs cover the instance and storage namespaces only; `/balance` and
# `/resource/user/instance/list` are undocumented endpoints the same key authenticates (verified
# live 2026-08-19): `/balance` carries `balance`, `availableBalance`, `availableVoucherAmount` and
# `availableCreditAmount`, and the resource listing is the priced, stock-aware catalog the console
# fills its launch form from.
#
# An instance is reached over ssh at `instanceSpecInfo.regionInfo.sshAddress`, the
# `instanceSpecInfo.nodePorts` entry for container port 22 and the login
# `instanceMetadata.instanceUsername`: the `ssh -p <nodePort> <user>@<address>` line their console
# prints.

import json
import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Self
from urllib.request import Request

from patos import FrozenModel

from ...jobs.arch import capability, card
from ..base import Credentials, http_transport
from ..capabilities import Rented

if TYPE_CHECKING:
    from ...jobs.vocabulary import OfferFilter
    from ..base import Transport

# `/instance/list` refuses a request without a pager, so the page size is ours.
_PAGE_SIZE = 50
_SSH_PORT = 22
# What a type's GPU name carries beyond the card (`B200-SXM-180GB-SPOT`, `H200-SXM-141GB-US`).
_FORM = re.compile(r"-(?:SXM|PCIE|NVL|\d+GB|SPOT|US|EU)\b.*$", re.IGNORECASE)


def api_key() -> str:
    """The HPC-AI key from `HPCAI_API_KEY` (no login flow or cookie JWT), refusing when unset."""
    return Credentials().key("HPCAI_API_KEY", hint="an API key minted in the HPC-AI console")


class Instance(Rented):
    """One instance as `/instance/list` reports it.

    address / port / user: where ssh reaches it, empty until it runs.
    phase / reason / message: where its start stands, and why it is stuck when it is.
    """

    address: str = ""
    port: int = 0
    user: str = ""
    phase: str = ""
    reason: str = ""
    message: str = ""

    @classmethod
    def listed(cls, item: Mapping) -> Self:
        """A `/instance/list` row as the instance it describes."""
        metadata = item.get("instanceMetadata") or {}
        runtime = item.get("instanceRuntimeInfo") or {}
        spec = item.get("instanceSpecInfo") or {}
        ports = spec.get("nodePorts") or []
        return cls(
            handle=str(metadata.get("instanceId") or ""),
            label=str(metadata.get("instanceName") or metadata.get("name") or ""),
            status=str(runtime.get("status") or ""),
            address=str((spec.get("regionInfo") or {}).get("sshAddress") or ""),
            port=next(
                (
                    int(row.get("nodePort") or 0)
                    for row in ports
                    if int(row.get("port") or 0) == _SSH_PORT
                ),
                0,
            ),
            user=str(metadata.get("instanceUsername") or ""),
            phase=str(runtime.get("phase") or ""),
            reason=str(runtime.get("diagnosisReason") or ""),
            message=str(runtime.get("diagnosisMessage") or ""),
        )

    @property
    def said(self) -> str:
        """Where its start stands, as a refusal names it (`DownloadImage BackOff`)."""
        return f"{self.phase or self.status or 'unlisted'} {self.reason}".strip()


class InstanceType(FrozenModel):
    """One rentable type in one region, as the console's launch form lists it.

    gpu: HPC-AI's GPU name (`B200-SXM-180GB-SPOT`), empty for a CPU type.
    usd_hr: the `perHour` rate, the one comparable across types, None when it lists none.
    """

    instance_type_id: str
    region_id: str
    spot: bool
    gpu: str = ""
    gpus: int = 0
    usd_hr: float | None = None
    region: str = ""
    in_stock: bool = False

    @classmethod
    def listed(cls, *, family: Mapping, region: Mapping, kind: Mapping) -> Self:
        """One `instanceTypeInfos` row, under its region and GPU family, as the type it is."""
        gpu = str(family.get("gpuName") or "")
        return cls(
            gpu=gpu,
            gpus=kind.get("gpuNum") or 0,
            usd_hr=cls._hourly(kind),
            region=region.get("regionName") or "",
            region_id=region.get("regionId") or "",
            instance_type_id=kind.get("instanceTypeId") or "",
            in_stock=kind.get("stockStatus") == "InStock",
            spot=bool(family.get("isSpotInstance")) or "SPOT" in gpu.upper(),
        )

    @property
    def card(self) -> str:
        """The card the type carries (`B200-SXM-180GB-SPOT` -> `B200`)."""
        return card(_FORM.sub("", self.gpu))

    @staticmethod
    def _hourly(kind: Mapping) -> float | None:
        """The `perHour` rate, the one comparable across types (vs `perDay`, `tidePerHour`)."""
        for price in kind.get("price") or []:
            if price.get("chargeMode") == "perHour":
                rate = price.get("price")
                return None if rate is None else float(rate)
        return None

    def fits(self, wanted: OfferFilter) -> bool:
        """Whether the type answers a request.

        It must be priced, hold at least `gpus` cards, be the card or architecture asked, be spot
        as asked (None for either), and sit under the hourly ceiling (0 for none).
        """
        return (
            self.usd_hr is not None
            and self.gpus >= wanted.gpus
            and (not wanted.gpu_name or self.card == card(wanted.gpu_name))
            and (wanted.arch is None or wanted.arch.holds(capability(self.card)))
            and wanted.pricing.admits(spot=self.spot)
            and (not wanted.max_usd_hr or self.usd_hr <= wanted.max_usd_hr)
        )


class HpcAiApi:
    """The calls a rental makes, each under the key read afresh.

    transport: sends a prepared `Request`, returning its response; injectable for tests.
    """

    def __init__(self, *, transport: Transport = http_transport) -> None:
        self.transport = transport

    def balance(self) -> float:
        """The `balance` `/balance` reports, what is left once vouchers and credits are spent."""
        return float(self.request("GET", path="/balance", body={})["balance"])

    def instance(self, handle: str) -> Instance | None:
        """`handle` as listed, None once unlisted; the walk stops at the page carrying it."""
        return next((found for found in self.instances() if found.handle == handle), None)

    def instances(self) -> Iterator[Instance]:
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
            yield from map(Instance.listed, payload.get("instances") or [])
            if page * _PAGE_SIZE >= int((payload.get("pager") or {}).get("totalEntries") or 0):
                return
            page += 1

    def request(self, method: str, *, path: str, body: dict) -> dict:
        """A call below `https://www.hpc-ai.com/api` (spelled inline), the key read afresh."""
        request = Request(
            f"https://www.hpc-ai.com/api{path}",
            method=method,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": api_key()},
        )
        return json.loads(self.transport(request))

    def types(self) -> list[InstanceType]:
        """Every rentable instance type, one row per type per region, in stock and cheapest first.

        The console's launch-form feed is the only place HPC-AI publishes a price, a GPU name or
        a stock count, and each row carries what a `[hosts.<name>.vars]` table needs.
        """
        payload = self.request("POST", path="/resource/user/instance/list", body={})
        regions = [
            (family, region)
            for family in payload.get("instanceInfos") or []
            for region in family.get("regionInfos") or []
        ]
        rows = [
            InstanceType.listed(family=family, region=region, kind=kind)
            for family, region in regions
            for kind in region.get("instanceTypeInfos") or []
        ]
        return sorted(
            rows, key=lambda row: (not row.in_stock, row.usd_hr is None, row.usd_hr or 0.0)
        )
