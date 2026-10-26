# Lambda Cloud instances as held machines, through its REST API (`cloud.lambdalabs.com/api/v1`).
#
# A Lambda instance is a whole VM with ssh as `ubuntu` from the first boot, so nothing runs at
# start: this machine's key is registered once under a name derived from it, the instance is
# launched with that name, and mb's setup takes over once the address answers (the same four calls
# SkyPilot's Lambda provisioner makes: instance types, launch, instances, terminate).

import hashlib
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import quote

from .api import CloudApi, Machine
from .core import CloudBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

_API = "https://cloud.lambdalabs.com/api/v1"
# The status words of a Lambda instance, mapped onto a machine's three.
_STATUS = {"active": "running", "booting": "pending", "unhealthy": "pending"}


class LambdaApi(CloudApi):
    """Lambda Cloud's instances, launched with this machine's key registered under its name."""

    key_variables = ("LAMBDA_API_KEY", "LAMBDA_CLOUD_API_KEY")
    hint = "an API key from the Lambda Cloud dashboard"
    exhausted = ("capacity",)

    @staticmethod
    def _machine(found: Mapping) -> Machine:
        kind = found.get("instance_type") or {}
        cents = kind.get("price_cents_per_hour")
        return Machine(
            handle=str(found.get("id", "")),
            status=_STATUS.get(str(found.get("status", "")), "gone"),
            host=str(found.get("ip") or ""),
            user="ubuntu",
            label=str(found.get("name") or ""),
            gpu=str(kind.get("description", "")),
            usd_hr=cents / 100 if cents is not None else None,
        )

    def create(self, offer: CatalogItem, *, name: str, public_key: str, disk_gb: int) -> str:
        del disk_gb  # every Lambda instance type carries its own disk
        body: dict[str, JsonValue] = {
            "region_name": offer.location,
            "instance_type_name": offer.instance_name,
            "ssh_key_names": [self.registered(public_key)],
            "quantity": 1,
            "name": name[:64],
        }
        launched = self._launched(url=f"{_API}/instance-operations/launch", body=body)
        return str(launched["data"]["instance_ids"][0])

    def machine(self, handle: str) -> Machine:
        try:
            found = self._call(
                "GET", url=f"{_API}/instances/{quote(handle, safe='')}", answer=dict
            ).get("data", {})
        except HTTPError as error:
            if error.code == 404:
                return Machine(handle=handle, status="gone")
            raise
        return self._machine(found)

    def machines(self) -> list[Machine]:
        return [
            self._machine(each)
            for each in self._call("GET", url=f"{_API}/instances", answer=dict).get("data", [])
        ]

    def registered(self, public_key: str) -> str:
        """The name `public_key` is registered under on this account, registering it first."""
        name = "mainboard-" + hashlib.sha256(public_key.strip().encode()).hexdigest()[:12]
        keys = self._call("GET", url=f"{_API}/ssh-keys", answer=dict).get("data", [])
        if not any(key.get("name") == name for key in keys):
            self._call(
                "POST",
                url=f"{_API}/ssh-keys",
                answer=dict,
                body={"name": name, "public_key": public_key.strip()},
            )
        return name

    def terminate(self, handle: str) -> None:
        try:
            self._call(
                "POST",
                url=f"{_API}/instance-operations/terminate",
                answer=dict,
                body={"instance_ids": [handle]},
            )
        except HTTPError as error:
            if error.code not in {400, 404}:
                raise


class LambdaBackend(CloudBackend):
    """Hold a Lambda instance as an ssh host: `mb host hold lambda --gpu-name H100 --for 2h`."""

    name = "lambda"
    catalog_name = "lambdalabs"
    client = LambdaApi
