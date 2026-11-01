# DigitalOcean GPU Droplets as held machines, through its REST API (`api.digitalocean.com/v2`).
#
# A GPU Droplet is a whole VM with ssh as `root` from the first boot, so nothing runs at start:
# this machine's key is registered once, under the fingerprint DigitalOcean names it by, and the
# droplet is created with that fingerprint. The image is the plan's AI/ML ready base image, which
# carries the AMD or NVIDIA driver; an ordinary image would have none. A Droplet bills until it is
# destroyed, powered off or not, so `terminate` is a delete.

import base64
import hashlib
import re
from typing import TYPE_CHECKING

from .....core.errors import MissionError
from .....enums import Bound
from ..api import CloudApi, Machine
from ..core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

    from ....hosts.lease import Lease

_API = "https://api.digitalocean.com/v2"
# A GPU plan's slug names its card and count (`gpu-h100x1-80gb`), and its base image drops the
# memory.
_PLAN = re.compile(r"^gpu-(?P<card>[a-z0-9]+x\d+)-")
_AMD_IMAGE = "gpu-amd-base"


class DigitalOceanApi(CloudApi):
    """DigitalOcean's GPU Droplets, created with this machine's key registered by fingerprint."""

    key_variables = ("DIGITAL_OCEAN_API_KEY", "DIGITALOCEAN_ACCESS_TOKEN")
    hint = "a personal access token from the DigitalOcean control panel's API page"
    exhausted = ("capacity", "unavailable", "not available")
    machine_url = _API + "/droplets/{}"
    machine_key = "droplet"

    @staticmethod
    def _fingerprint(public_key: str) -> str:
        """The colon-separated MD5 of `public_key`'s blob, which DigitalOcean names a key by."""
        blob = base64.b64decode(public_key.split()[1])
        digest = hashlib.md5(blob, usedforsecurity=False).hexdigest()
        return ":".join(digest[at : at + 2] for at in range(0, len(digest), 2))

    @staticmethod
    def _image(plan: str) -> str:
        """The AI/ML ready base image of a GPU plan (`gpu-h100x1-80gb` -> `gpu-h100x1-base`)."""
        if (found := _PLAN.match(plan)) is None:
            raise MissionError(
                f"digitalocean plan {plan!r} is not a GPU plan this house can image"
            )
        card = found["card"]
        return _AMD_IMAGE if card.startswith("mi") else f"gpu-{card}-base"

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        status = str(found.get("status", ""))
        addresses = (found.get("networks") or {}).get("v4") or []
        host = next((each["ip_address"] for each in addresses if each.get("type") == "public"), "")
        size = found.get("size") or {}
        price = size.get("price_hourly")
        booted = "running" if status == "active" and host else "pending"
        return Machine(
            handle=str(found.get("id", "")),
            status="gone" if status == "archive" else booted,
            host=host,
            user="root",
            label=str(found.get("name") or ""),
            gpu=str((found.get("gpu_info") or size.get("gpu_info") or {}).get("model", "")),
            usd_hr=float(price) if price is not None else None,
        )

    def create(
        self, offer: CatalogItem, *, lease: Lease, name: str, public_key: str, disk_gb: int
    ) -> str:
        del disk_gb  # every GPU plan carries its own boot and scratch disks
        del lease  # the declared card limit caps DigitalOcean, which takes no deadline
        body: dict[str, JsonValue] = {
            "name": name[:255],
            "size": offer.instance_name,
            "region": offer.location,
            "image": self._image(offer.instance_name),
            "ssh_keys": [self.registered(public_key)],
            "tags": ["mainboard"],
        }
        launched = self._launched(url=f"{_API}/droplets", body=body)
        return str(launched["droplet"]["id"])

    def machines(self) -> list[Machine]:
        listed = self._call("GET", url=f"{_API}/droplets?type=gpus&per_page=200", answer=dict)
        return [self._machine(each) for each in listed.get("droplets", [])]

    def registered(self, public_key: str) -> str:
        """The fingerprint `public_key` is registered under on this account, registering it."""
        key = public_key.strip()
        fingerprint = self._fingerprint(key)
        if self._fetched(f"{_API}/account/keys/{fingerprint}") is None:
            body: dict[str, JsonValue] = {"name": self._named(key), "public_key": key}
            self._call("POST", url=f"{_API}/account/keys", answer=dict, body=body)
        return fingerprint


class DigitalOceanBackend(CloudBackend):
    """Hold a GPU Droplet as an ssh host: `mb host hold digitalocean --gpu-name H100 --for 2h`."""

    # DigitalOcean bills a card monthly, after the fact.
    bound = Bound.CARD
    name = "digitalocean"
    catalog_name = "digitalocean"
    client = DigitalOceanApi
