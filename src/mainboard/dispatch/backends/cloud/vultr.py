# Vultr Cloud GPU and Bare Metal as held machines, through its REST API (`api.vultr.com/v2`).
#
# A GPU plan is either a cloud instance (`/instances`, plans `vcg-*` and `vdm-*`) or a bare metal
# server (`/bare-metals`, plans `vbm-*`, where the GH200 and the 8-GPU Hopper, Blackwell and
# Instinct nodes are sold). Both boot an Ubuntu image with ssh as `root` and take this machine's
# key by id. The two kinds share no id space, so a handle is looked up as an instance first and
# as a bare metal server when Vultr has no such instance. Preemptible plans are not in the
# catalog gpuhunt publishes, so only on-demand plans rent here. The public plan list carries
# prices but no stock for the GPU plans: their regions come from an authenticated availability
# call, so what is rentable shows only with a key.

from typing import TYPE_CHECKING, NamedTuple
from urllib.parse import quote

from .api import CloudApi, Machine
from .core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

_API = "https://api.vultr.com/v2"
# Ubuntu 24.04 LTS x64 in Vultr's OS list (`GET /os`, public).
_UBUNTU = 2284
# What an address reads before Vultr assigns one.
_UNASSIGNED = ("", "0.0.0.0")


class _Kind(NamedTuple):
    """One of the two things Vultr rents: its URL path, and the keys its answers wrap it in."""

    path: str
    one: str
    many: str


_INSTANCE = _Kind("instances", "instance", "instances")
_METAL = _Kind("bare-metals", "bare_metal", "bare_metals")


class VultrApi(CloudApi):
    """Vultr's instances and bare metal servers, created with this machine's key by id."""

    key_variables = ("VULTR_API_KEY",)
    hint = "an API key from the Vultr customer portal's API page, this machine's address allowed"
    exhausted = ("not available", "out of stock", "no capacity", "insufficient")

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        address = str(found.get("main_ip") or "")
        host = "" if address in _UNASSIGNED else address
        booted = found.get("status") == "active" and found.get("server_status", "ok") == "ok"
        return Machine(
            handle=str(found.get("id", "")),
            status="running" if booted and host else "pending",
            host=host,
            user="root",
            label=str(found.get("label") or ""),
            gpu=str(found.get("plan") or ""),
        )

    def create(self, offer: CatalogItem, *, name: str, public_key: str, disk_gb: int) -> str:
        del disk_gb  # every GPU plan carries its own disks
        body: dict[str, JsonValue] = {
            "region": offer.location,
            "plan": offer.instance_name,
            "os_id": _UBUNTU,
            "sshkey_id": [self.registered(public_key)],
            "label": name[:63],
            "hostname": name[:63],
        }
        kind = _METAL if offer.instance_name.startswith("vbm-") else _INSTANCE
        launched = self._launched(url=f"{_API}/{kind.path}", body=body)
        return str(launched[kind.one]["id"])

    def machine(self, handle: str) -> Machine:
        for kind in (_INSTANCE, _METAL):
            if (
                found := self._fetched(f"{_API}/{kind.path}/{quote(handle, safe='')}")
            ) is not None:
                return self._machine(found[kind.one])
        return Machine(handle=handle, status="gone")

    def machines(self) -> list[Machine]:
        listed = (
            (kind, self._call("GET", url=f"{_API}/{kind.path}?per_page=500", answer=dict))
            for kind in (_INSTANCE, _METAL)
        )
        return [
            self._machine(each) for kind, answer in listed for each in answer.get(kind.many, [])
        ]

    def registered(self, public_key: str) -> str:
        """The id `public_key` is registered under on this account, registering it first."""
        known = self._call("GET", url=f"{_API}/ssh-keys?per_page=500", answer=dict)
        same = (
            each
            for each in known.get("ssh_keys", [])
            if self._same(each["ssh_key"], public_key=public_key)
        )
        if (match := next(same, None)) is not None:
            return str(match["id"])
        body: dict[str, JsonValue] = {
            "name": self._named(public_key),
            "ssh_key": public_key.strip(),
        }
        added = self._call("POST", url=f"{_API}/ssh-keys", answer=dict, body=body)
        return str(added["ssh_key"]["id"])

    def terminate(self, handle: str) -> None:
        for kind in (_INSTANCE, _METAL):
            self._ended("DELETE", url=f"{_API}/{kind.path}/{quote(handle, safe='')}")


class VultrBackend(CloudBackend):
    """Hold a Vultr GPU machine as an ssh host: `mb host hold vultr --gpu-name GH200 --for 2h`."""

    name = "vultr"
    catalog_name = "vultr"
    client = VultrApi
