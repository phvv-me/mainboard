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
from urllib.parse import quote
from urllib.request import Request

from ....core.errors import MissionError
from ..base import Credentials, http_transport
from ..exceptions import CapacityGone
from .api import CloudApi, Machine
from .core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

    from ..base import Transport

_API = "https://api.verda.com/v1"
# Seconds before its expiry that a token is exchanged again.
_TOKEN_MARGIN_S = 60.0
# The statuses that end a machine's life; `running` is the one that takes ssh, and any other word
# is a boot still going, since Verda documents its list as open.
_GONE = frozenset(
    {"discontinued", "deleting", "error", "no_capacity", "installation_failed", "notfound"}
)
_IMAGE = re.compile(r"^ubuntu-(\d+)\.(\d+)-cuda-(\d+)\.(\d+)")


class VerdaApi(CloudApi):
    """Verda's instances, created with this machine's key and an OS volume of their own."""

    key_variables = ("VERDA_CLIENT_ID",)
    secret_variables: ClassVar[tuple[str, ...]] = ("VERDA_CLIENT_SECRET",)
    hint = "a Cloud API client id and secret from the Verda project's Keys page"
    exhausted = ("no capacity", "no_capacity", "not available", "unavailable")

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

    @staticmethod
    def _rank(image: re.Match[str]) -> tuple[tuple[int, int], tuple[int, int], bool]:
        """How new `image` is: its CUDA, then its Ubuntu, a plain image beating its docker twin."""
        ubuntu_major, ubuntu_minor, cuda_major, cuda_minor = map(int, image.groups())
        return (cuda_major, cuda_minor), (ubuntu_major, ubuntu_minor), "docker" not in image.string

    def __init__(self, *, transport: Transport = http_transport) -> None:
        super().__init__(transport=transport)
        self._token = ""
        self._expires = 0.0

    def _exhausted(self, refused: HTTPError) -> None:
        if refused.code == 503:
            raise CapacityGone(f"verda answered 503: {refused.reason}") from refused
        super()._exhausted(refused)

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
        if images := [found for image in supported if (found := _IMAGE.match(image))]:
            return max(images, key=self._rank).string
        raise MissionError(f"verda lists no Ubuntu CUDA image for {instance_type}")

    def bearer(self) -> str:
        if monotonic() >= self._expires:
            request = Request(
                f"{_API}/oauth2/token",
                data=json.dumps(
                    {
                        "grant_type": "client_credentials",
                        "client_id": self.key(),
                        "client_secret": Credentials().key(*self.secret_variables, hint=self.hint),
                    }
                ).encode(),
                method="POST",
                headers={"Content-Type": "application/json", "User-Agent": "mainboard"},
            )
            grant = json.loads(self.transport(request))
            self._token = str(grant["access_token"])
            self._expires = monotonic() + float(grant.get("expires_in", 0)) - _TOKEN_MARGIN_S
        return self._token

    def create(self, offer: CatalogItem, *, name: str, public_key: str, disk_gb: int) -> str:
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
        Credentials().key(*self.secret_variables, hint=self.hint)
        return super().key()

    def machine(self, handle: str) -> Machine:
        found = self._fetched(f"{_API}/instances/{quote(handle, safe='')}")
        return Machine(handle=handle, status="gone") if found is None else self._machine(found)

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
        found = self._fetched(f"{_API}/instances/{quote(handle, safe='')}")
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

    name = "verda"
    catalog_name = "verda"
    client = VerdaApi
