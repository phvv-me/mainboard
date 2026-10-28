# Shadeform as one key to many clouds, through its REST API (`api.shadeform.ai/v1`).
#
# Shadeform lists the GPU machines of the clouds behind it (Lambda, Crusoe, Hyperstack, Vultr,
# Nebius, Voltage Park and others) as one catalog with live availability per region and a boot
# time per type, and creates, bills and deletes them under its own account. gpuhunt has no
# catalog of it, so the offers come from `GET /instances/types`; each row keeps the cloud and
# region behind it in `location` (`lambda/us-east-1`) so a receipt names the hardware's owner, not
# only the broker. A machine takes ssh as the user and port its info names, with this machine's
# key registered on the account and sent by id. The image is the newest Ubuntu one with CUDA that
# the instance type lists, since a type's default can be too old for a Blackwell card.

import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from gpuhunt import AcceleratorVendor, CatalogItem

from .api import CloudApi, Machine
from .core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pydantic import JsonValue

_API = "https://api.shadeform.ai/v1"
_OS = re.compile(r"^ubuntu(\d+)\.(\d+)_cuda(\d+)\.(\d+)")
# The statuses that end a machine's life; `active` is the one that takes ssh.
_GONE = frozenset({"deleting", "deleted"})


class ShadeformApi(CloudApi):
    """Shadeform's instances, created for a cloud and region with this machine's key by id."""

    key_variables = ("SHADEFORM_API_KEY",)
    hint = "an API key from the Shadeform dashboard"
    exhausted = ("not available", "unavailable", "no capacity", "out of stock")

    @staticmethod
    def _image(options: Iterable[str]) -> str | None:
        """The newest Ubuntu image with CUDA among an instance type's options, if it lists any."""
        images = [found for option in options if (found := _OS.match(option))]
        return max(images, key=ShadeformApi._rank).string if images else None

    @staticmethod
    def _item(kind: Mapping, *, where: Mapping) -> CatalogItem:
        """One instance type in one region and rental type, as the catalog row it rents as."""
        setup = kind["configuration"]
        cents = kind["hourly_price"]
        price = float(where.get("hourly_price") or cents / 100)
        gpu = str(setup["gpu_type"]).split("_")[0]
        return CatalogItem(
            provider="shadeform",
            instance_name=kind["shade_instance_type"],
            location=f"{kind['cloud']}/{where['region']}",
            price=price,
            cpu=kind["vcpus"],
            memory=kind["memory_in_gb"],
            gpu_count=kind["num_gpus"],
            gpu_name=gpu,
            gpu_memory=setup["vram_per_gpu_in_gb"],
            spot=where["rental_type"] == "spot",
            disk_size=kind["storage_in_gb"],
            gpu_vendor=AcceleratorVendor.cast(setup["gpu_manufacturer"]),
            provider_data={
                "cloud": kind["cloud"],
                "region": where["region"],
                "boot_min_s": kind["boot_time"]["min_boot_in_sec"],
                "boot_max_s": kind["boot_time"]["max_boot_in_sec"],
                "os": ShadeformApi._image(setup.get("os_options") or []),
            },
        )

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        status = str(found.get("status", ""))
        host = str(found.get("ip") or "")
        gone = status in _GONE or status.startswith("error")
        cents = found.get("hourly_price")
        return Machine(
            handle=str(found.get("id", "")),
            status="gone" if gone else "running" if status == "active" and host else "pending",
            host=host,
            port=int(found.get("ssh_port") or 22),
            user=str(found.get("ssh_user") or "shadeform"),
            label=str(found.get("name") or ""),
            gpu=str((found.get("configuration") or {}).get("gpu_type", "")),
            usd_hr=cents / 100 if cents is not None else None,
        )

    @staticmethod
    def _rank(image: re.Match[str]) -> tuple[tuple[int, int], tuple[int, int]]:
        """How new `image` is: its CUDA, then its Ubuntu."""
        ubuntu_major, ubuntu_minor, cuda_major, cuda_minor = map(int, image.groups())
        return (cuda_major, cuda_minor), (ubuntu_major, ubuntu_minor)

    def create(self, offer: CatalogItem, *, name: str, public_key: str, disk_gb: int) -> str:
        del disk_gb  # a type carries its own storage
        where = offer.provider_data
        body: dict[str, JsonValue] = {
            "cloud": str(where["cloud"]),
            "region": str(where["region"]),
            "shade_instance_type": offer.instance_name,
            "shade_cloud": True,
            "name": name[:63],
            "ssh_key_id": self.registered(public_key),
            "rental_type": "spot" if offer.spot else "on_demand",
        }
        if image := where.get("os"):
            body["os"] = str(image)
        return str(self._launched(url=f"{_API}/instances/create", body=body)["id"])

    def machine(self, handle: str) -> Machine:
        found = self._fetched(f"{_API}/instances/{quote(handle, safe='')}/info")
        return Machine(handle=handle, status="gone") if found is None else self._machine(found)

    def machines(self) -> list[Machine]:
        listed = self._call("GET", url=f"{_API}/instances", answer=dict)
        return [self._machine(each) for each in listed.get("instances") or []]

    def offers(self) -> list[CatalogItem]:
        listed = self._call("GET", url=f"{_API}/instances/types", answer=dict)
        return [
            self._item(kind, where=where)
            for kind in listed.get("instance_types") or []
            for where in kind.get("availability") or []
            if where["available"]
        ]

    def registered(self, public_key: str) -> str:
        """The id `public_key` is registered under on this account, registering it first."""
        known = self._call("GET", url=f"{_API}/sshkeys", answer=dict)
        same = (
            each
            for each in known.get("ssh_keys") or []
            if self._same(each["public_key"], public_key=public_key)
        )
        if (match := next(same, None)) is not None:
            return str(match["id"])
        body: dict[str, JsonValue] = {
            "name": self._named(public_key),
            "public_key": public_key.strip(),
        }
        return str(self._call("POST", url=f"{_API}/sshkeys/add", answer=dict, body=body)["id"])

    def terminate(self, handle: str) -> None:
        self._ended("POST", url=f"{_API}/instances/{quote(handle, safe='')}/delete")


class ShadeformBackend(CloudBackend):
    """Hold a machine from any cloud Shadeform fronts: `mb host hold shadeform --gpu-name H100`."""

    name = "shadeform"
    client = ShadeformApi
