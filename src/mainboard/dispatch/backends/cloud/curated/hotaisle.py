# Hot Aisle AMD virtual machines as held machines, through its REST API
# (`admin.hotaisle.app/api`).
#
# Hot Aisle authenticates by a user API key sent as `Authorization: Token <key>`, and every
# resource lives under a team handle, so a second variable names the team. A machine is created
# from the specs object its availability list publishes, which gpuhunt keeps whole in the offer's
# `provider_data`, and it takes ssh as the `hotaisle` user through a NAT port: the keys allowed in
# are the ones registered on the user, fetched by the machine at each login, so this machine's key
# is registered once and needs no id at create. A machine bills a minimum of 30 minutes, and a
# delete before that needs `force`, which forfeits only the unused minimum.

from typing import TYPE_CHECKING, ClassVar
from urllib.parse import quote

from pydantic import JsonValue, TypeAdapter

from .....enums import Bound
from ..api import CloudApi, Machine
from ..core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem

    from ....hosts.lease import Lease

_API = "https://admin.hotaisle.app/api"


class HotAisleApi(CloudApi):
    """Hot Aisle's virtual machines, created from an availability entry's specs."""

    key_variables = ("HOTAISLE_API_KEY",)
    team_variables: ClassVar[tuple[str, ...]] = ("HOTAISLE_TEAM_HANDLE",)
    hint = "an API key and the team handle from the Hot Aisle admin site"
    exhausted_statuses = {404: "hot aisle has no virtual machine of those specs available"}

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        access = found.get("ssh_access") or {}
        host = str(
            access.get("dns_name") or access.get("ip_address") or found.get("ip_address") or ""
        )
        gpus = found.get("gpus") or []
        ready = found.get("status") == "ready" and bool(host)
        return Machine(
            handle=str(found.get("deployment_id", "")),
            status="running" if ready else "pending",
            host=host if ready else "",
            port=int(access.get("port") or 22),
            user="hotaisle",
            label=str(found.get("name") or ""),
            gpu=str(gpus[0].get("model", "")) if gpus else "",
        )

    def _authorized(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.key()}"}

    def _listing(self, url: str) -> list[dict]:
        """The array at `url`, which this API answers `null` for when it holds nothing."""
        raw = self._send("GET", url=url)
        return [] if raw in (b"", b"null") else TypeAdapter(list[dict]).validate_json(raw)

    def _locate(self, handle: str) -> str:
        return f"{self._machines()}{quote(handle, safe='')}/"

    def _machines(self) -> str:
        """The URL of this team's virtual machines."""
        team = quote(self._secret(self.team_variables), safe="")
        return f"{_API}/teams/{team}/virtual_machines/"

    def create(
        self, offer: CatalogItem, *, lease: Lease, name: str, public_key: str, disk_gb: int
    ) -> str:
        del name, disk_gb  # a machine is named by Hot Aisle, and carries the disk its specs list
        del lease  # the prepaid balance caps Hot Aisle, which takes no deadline
        self.registered(public_key)
        specs = TypeAdapter(dict[str, JsonValue]).validate_python(offer.provider_data["vm_specs"])
        return str(self._launched(url=self._machines(), body=specs)["deployment_id"])

    def key(self) -> str:
        """The API key, refused unless the team handle is set too."""
        self._secret(self.team_variables)
        return super().key()

    def machines(self) -> list[Machine]:
        return [self._machine(each) for each in self._listing(self._machines())]

    def registered(self, public_key: str) -> None:
        """Register `public_key` on the user unless it already is."""
        known = self._listing(f"{_API}/user/ssh_keys/")
        if not any(
            self._same(str(each.get("public_key", "")), public_key=public_key) for each in known
        ):
            kind, blob, *_ = public_key.split()
            body: dict[str, JsonValue] = {
                "authorized_key": f"{kind} {blob} {self._named(public_key)}"
            }
            self._call("POST", url=f"{_API}/user/ssh_keys/", answer=dict, body=body)

    def terminate(self, handle: str) -> None:
        self._ended("DELETE", url=f"{self._locate(handle)}?force=true")


class HotAisleBackend(CloudBackend):
    """Hold a Hot Aisle MI300X as an ssh host: `mb host hold hotaisle --gpu-name MI300X`."""

    # Hot Aisle bills by the minute from prepaid credits, 30 minutes at least (the header above).
    bound = Bound.PREPAID
    BILLED_MINIMUM_S = 30 * 60
    name = "hotaisle"
    catalog_name = "hotaisle"
    client = HotAisleApi

    def rentable(self, item: CatalogItem) -> bool:
        """Virtual machines only: a bare metal server is reserved through another endpoint."""
        return "vm_specs" in (item.provider_data or {})
