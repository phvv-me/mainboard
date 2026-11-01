# Verda (formerly DataCrunch) instances as held machines, through its REST API
# (`api.verda.com/v1`).
#
# Verda signs nobody in with a static key: a client id and secret are exchanged for a bearer token
# that expires, which `bearer` renews a minute early. An instance is a whole VM with ssh as `root`
# from the first boot, created with an OS volume of its own that the delete takes along, so a
# rental leaves nothing billing. The image is the newest Ubuntu one with CUDA drivers that the
# instance type lists as supported, since a Blackwell card needs a newer driver than a Hopper one.

import json
import re
from time import monotonic
from typing import TYPE_CHECKING, ClassVar
from urllib.error import HTTPError

from .....core.errors import MissionError
from .....enums import Bound
from ..api import CloudApi, Machine
from ..core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

    from ....hosts.lease import Lease

_API = "https://api.verda.com/v1"
# Seconds before its expiry that a token is exchanged again.
_TOKEN_MARGIN_S = 60.0
# The statuses that end a machine's life; `running` is the one that takes ssh, and any other word
# is a boot still going, since Verda documents its list as open.
_GONE = frozenset(
    {"discontinued", "deleting", "error", "no_capacity", "installation_failed", "notfound"}
)
_IMAGE = re.compile(r"^ubuntu-(?P<ubuntu>\d+\.\d+)-cuda-(?P<cuda>\d+\.\d+)")


class VerdaApi(CloudApi):
    """Verda's instances, created with this machine's key and an OS volume of their own."""

    key_variables = ("VERDA_CLIENT_ID",)
    secret_variables: ClassVar[tuple[str, ...]] = ("VERDA_CLIENT_SECRET",)
    hint = "a Cloud API client id and secret from the Verda project's Keys page"
    exhausted = ("no capacity", "no_capacity", "not available", "unavailable")
    exhausted_statuses = {503: "verda answered 503: {reason}"}
    machine_url = _API + "/instances/{}"
    # The bearer token its client id and secret were last exchanged for, and when that lapses.
    _expires = 0.0
    _token = ""

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        status = str(found.get("status", ""))
        price = found.get("price_per_hour")
        return Machine(
            handle=str(found.get("id", "")),
            status="gone" if status in _GONE else "running" if status == "running" else "pending",
            host=str(found.get("ip") or ""),
            user="root",
            label=str(found.get("hostname") or ""),
            gpu=str(found.get("instance_type") or ""),
            usd_hr=float(price) if price is not None else None,
        )

    def _image(self, instance_type: str) -> str:
        """The newest Ubuntu image with CUDA drivers that `instance_type` supports."""
        listed = self._call("GET", url=f"{_API}/instance-types", answer=list[dict])
        supported = next(
            (
                kind.get("supported_os") or []
                for kind in listed
                if kind.get("instance_type") == instance_type
            ),
            [],
        )
        # A stable sort lists a plain image before its docker twin, which a tie in version keeps.
        plain_first = sorted(supported, key=lambda each: "docker" in each)
        if image := self._newest(plain_first, pattern=_IMAGE):
            return image
        raise MissionError(f"verda lists no Ubuntu CUDA image for {instance_type}")

    def bearer(self) -> str:
        if monotonic() >= self._expires:
            body: dict[str, JsonValue] = {
                "grant_type": "client_credentials",
                "client_id": self.key(),
                "client_secret": self._secret(self.secret_variables),
            }
            grant = json.loads(
                self._send("POST", url=f"{_API}/oauth2/token", body=body, headers={})
            )
            self._token = str(grant["access_token"])
            self._expires = monotonic() + float(grant.get("expires_in", 0)) - _TOKEN_MARGIN_S
        return self._token

    def create(
        self, offer: CatalogItem, *, lease: Lease, name: str, public_key: str, disk_gb: int
    ) -> str:
        del lease  # the prepaid balance caps Verda, which takes no deadline
        volume: dict[str, JsonValue] = {"name": f"{name}-os"[:63], "size": disk_gb}
        if offer.spot:
            volume["on_spot_discontinue"] = "delete_permanently"
        body: dict[str, JsonValue] = {
            "instance_type": offer.instance_name,
            "image": self._image(offer.instance_name),
            "ssh_key_ids": [self.registered(public_key)],
            "hostname": name[:63],
            "description": "mainboard",
            "location_code": offer.location,
            "is_spot": bool(offer.spot),
            "os_volume": volume,
        }
        try:
            return self._send("POST", url=f"{_API}/instances", body=body).decode().strip('"')
        except HTTPError as refused:
            self._exhausted(refused)
            raise

    def key(self) -> str:
        """The client id, refused unless its secret is set too."""
        self._secret(self.secret_variables)
        return super().key()

    def machines(self) -> list[Machine]:
        listed = self._call("GET", url=f"{_API}/instances", answer=list[dict])
        return [self._machine(each) for each in listed]

    def registered(self, public_key: str) -> str:
        """The id `public_key` is registered under on this account, registering it first."""
        known = self._call("GET", url=f"{_API}/ssh-keys", answer=list[dict])
        same = (
            each for each in known if self._same(str(each.get("key", "")), public_key=public_key)
        )
        if (match := next(same, None)) is not None:
            return str(match["id"])
        body: dict[str, JsonValue] = {"name": self._named(public_key), "key": public_key.strip()}
        return self._send("POST", url=f"{_API}/ssh-keys", body=body).decode().strip('"')

    def terminate(self, handle: str) -> None:
        """Delete `handle` with its OS volume, which would otherwise stay detached and billing."""
        found = self._fetched(self._locate(handle))
        if found is None:
            return
        volume = found.get("os_volume_id")
        self._ended(
            "PUT",
            url=f"{_API}/instances",
            body={
                "action": "delete",
                "id": handle,
                "volume_ids": [volume] if volume else [],
                "delete_permanently": True,
            },
        )


class VerdaBackend(CloudBackend):
    """Hold a Verda instance as an ssh host: `mb host hold verda --gpu-name B200 --for 2h`."""

    # Verda draws a prepaid balance and discontinues every instance at zero (docs.verda.com,
    # pricing and billing, 2026-10-20).
    bound = Bound.PREPAID
    name = "verda"
    catalog_name = "verda"
    client = VerdaApi
