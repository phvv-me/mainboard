# Hyperstack GPU virtual machines as held machines, through its REST API
# (`infrahub-api.nexgencloud.com/v1`).
#
# Hyperstack bills per minute from a prepaid balance and lists on-demand H100, H200, A100, L40,
# B200 and B300 flavors with live stock per region. gpuhunt has no catalog of it, so the offers
# come from two calls: the flavors with their stock, and the pricebook that prices a GPU model by
# the hour. A flavor lives in a region's environment and boots with a keypair of that environment,
# so one `mainboard-<region>` environment and this machine's key in it are made on the first rent
# there. A machine takes ssh as `ubuntu` on a floating address, and bills from the moment it is
# ACTIVE until it is deleted. A drained balance ends every machine, so the balance is also the
# spend cap: fund what a campaign should burn.

import re
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import quote

from gpuhunt import AcceleratorVendor, CatalogItem

from ....core.errors import MissionError
from .api import CloudApi, Machine
from .core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue

_API = "https://infrahub-api.nexgencloud.com/v1"
_IMAGE = re.compile(r"^Ubuntu Server (\d+)\.(\d+) LTS R(\d+) CUDA (\d+)\.(\d+)$")
# What a refusal says when the environment or the keypair is already there.
_TAKEN = ("exist", "already", "in use", "duplicate")
_SSH: dict[str, JsonValue] = {
    "direction": "ingress",
    "protocol": "tcp",
    "port_range_min": 22,
    "port_range_max": 22,
    "ethertype": "IPv4",
    "remote_ip_prefix": "0.0.0.0/0",
}


class HyperstackApi(CloudApi):
    """Hyperstack's virtual machines, created in a per-region environment of this machine's."""

    key_variables = ("HYPERSTACK_API_KEY",)
    hint = "an API key from the Hyperstack console's API keys page"
    exhausted = ("stock", "insufficient", "not enough", "no available", "capacity", "unavailable")

    @staticmethod
    def _card(model: str) -> str:
        """The card in a flavor's GPU name: `H100-80G-PCIe` is an H100, `RTX-A6000` an A6000."""
        return next(part for part in model.split("-") if part != "RTX")

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        status = str(found.get("status", ""))
        host = str(found.get("floating_ip") or "")
        gone = status in ("ERROR", "DELETING", "DELETED")
        return Machine(
            handle=str(found.get("id", "")),
            status="gone" if gone else "running" if status == "ACTIVE" and host else "pending",
            host=host,
            user="ubuntu",
            label=str(found.get("name") or ""),
            gpu=str((found.get("flavor") or {}).get("gpu", "")),
        )

    @staticmethod
    def _rank(image: re.Match[str]) -> tuple[tuple[int, int], int, tuple[int, int]]:
        """How new `image` is: its CUDA, then its driver, then its Ubuntu."""
        ubuntu_major, ubuntu_minor, driver, cuda_major, cuda_minor = map(int, image.groups())
        return (cuda_major, cuda_minor), driver, (ubuntu_major, ubuntu_minor)

    def _authorized(self) -> dict[str, str]:
        return {"api_key": self.key()}

    def _ensured(self, *, url: str, body: Mapping[str, JsonValue]) -> None:
        """Create what `body` names at `url`, unless the cloud says it already has it."""
        try:
            self._call("POST", url=url, answer=dict, body=body)
        except HTTPError as refused:
            said = refused.read().decode(errors="replace").lower()
            if refused.code not in (400, 409, 422) or not any(word in said for word in _TAKEN):
                raise

    def _image(self, region: str) -> str:
        """The newest Ubuntu image with CUDA drivers in `region`."""
        listed = self._call("GET", url=f"{_API}/core/images?region={quote(region)}", answer=dict)
        names = [
            image["name"] for group in listed.get("images") or [] for image in group["images"]
        ]
        if images := [found for name in names if (found := _IMAGE.match(name))]:
            return max(images, key=self._rank).string
        raise MissionError(f"hyperstack lists no Ubuntu CUDA image in {region}")

    def create(self, offer: CatalogItem, *, name: str, public_key: str, disk_gb: int) -> str:
        del disk_gb  # a flavor carries its own root and ephemeral disks
        region = offer.location
        environment = f"mainboard-{region.lower()}"
        key = self._named(public_key)
        self._ensured(
            url=f"{_API}/core/environments", body={"name": environment, "region": region}
        )
        self._ensured(
            url=f"{_API}/core/keypairs",
            body={"name": key, "environment_name": environment, "public_key": public_key.strip()},
        )
        body: dict[str, JsonValue] = {
            "name": name[:50],
            "environment_name": environment,
            "image_name": self._image(region),
            "flavor_name": offer.instance_name,
            "key_name": key,
            "count": 1,
            "assign_floating_ip": True,
            "security_rules": [_SSH],
        }
        launched = self._launched(url=f"{_API}/core/virtual-machines", body=body)
        return str(launched["instances"][0]["id"])

    def machine(self, handle: str) -> Machine:
        found = self._fetched(f"{_API}/core/virtual-machines/{quote(handle, safe='')}")
        return (
            Machine(handle=handle, status="gone")
            if found is None
            else self._machine(found["instance"])
        )

    def machines(self) -> list[Machine]:
        listed = self._call("GET", url=f"{_API}/core/virtual-machines", answer=dict)
        return [self._machine(each) for each in listed.get("instances") or []]

    def offers(self) -> list[CatalogItem]:
        flavors = self._call("GET", url=f"{_API}/core/flavors", answer=dict)
        prices = {
            each["name"]: float(each["value"])
            for each in self._call("GET", url=f"{_API}/pricebook", answer=list[dict])
        }
        return [
            CatalogItem(
                provider="hyperstack",
                instance_name=flavor["name"],
                location=group["region_name"],
                price=prices[flavor["gpu"]] * flavor["gpu_count"],
                cpu=flavor["cpu"],
                memory=flavor["ram"],
                gpu_count=flavor["gpu_count"],
                gpu_name=self._card(flavor["gpu"]),
                gpu_memory=None,
                spot=False,
                disk_size=flavor["disk"] + flavor["ephemeral"],
                gpu_vendor=AcceleratorVendor.NVIDIA,
            )
            for group in flavors.get("data") or []
            for flavor in group["flavors"]
            if flavor["gpu_count"] and flavor["stock_available"] and flavor["gpu"] in prices
        ]

    def terminate(self, handle: str) -> None:
        self._ended("DELETE", url=f"{_API}/core/virtual-machines/{quote(handle, safe='')}")


class HyperstackBackend(CloudBackend):
    """Hold a Hyperstack GPU VM as an ssh host: `mb host hold hyperstack --gpu-name B200`."""

    name = "hyperstack"
    client = HyperstackApi
