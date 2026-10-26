# RunPod pods as held machines, through RunPod's REST API (`rest.runpod.io/v1`).
#
# A pod is a container on a GPU host. It boots plain Ubuntu (the environment brings its own CUDA
# libraries) with a start command that installs and starts sshd and authorizes this machine's key,
# the way SkyPilot's RunPod provisioner does, then stays up; mb's setup does the rest over ssh.
# Port 22 is published on the host's public address under a mapped port.

import json
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import quote

from ...hosts.rentals import seeded
from .api import CloudApi, Machine
from .core import CloudBackend

_SSHD_INSTALL_COMMAND = (
    "command -v sshd >/dev/null || (apt-get update -qq && apt-get install -y -qq "
    "openssh-server curl ca-certificates >/dev/null)"
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

_API = "https://rest.runpod.io/v1"


class RunPodApi(CloudApi):
    """RunPod's pods, each a container that starts sshd with this machine's key."""

    key_variables = ("RUNPOD_API_KEY",)
    hint = "an API key from the RunPod console's settings"
    exhausted = ("no longer any instances available", "not enough", "unavailable", "no available")
    # The image a pod boots: plain Ubuntu, since the environment brings its own CUDA libraries
    # and only the host driver comes from the machine.
    image = "ubuntu:24.04"

    @staticmethod
    def _machine(pod: Mapping) -> Machine:
        desired = str(pod.get("desiredStatus", "")).upper()
        ports = pod.get("portMappings") or {}
        port = int(ports.get("22", 0) or 0)
        host = str(pod.get("publicIp") or "")
        status = (
            "gone"
            if desired in {"TERMINATED", "EXITED"}
            else "running"
            if desired == "RUNNING" and host and port
            else "pending"
        )
        gpu = pod.get("gpu") or {}
        return Machine(
            handle=str(pod.get("id", "")),
            status=status,
            host=host,
            port=port or 22,
            user="root",
            label=str(pod.get("name", "")),
            gpu=str(gpu.get("displayName", "") if isinstance(gpu, dict) else json.dumps(gpu)),
            usd_hr=pod.get("costPerHr"),
        )

    @staticmethod
    def _started(public_key: str) -> str:
        """The pod's start command: sshd installed and running with `public_key`, then held up."""
        return "\n".join(
            (
                "export DEBIAN_FRONTEND=noninteractive",
                _SSHD_INSTALL_COMMAND,
                "mkdir -p /run/sshd && ssh-keygen -A",
                seeded(public_key),
                "/usr/sbin/sshd",
                "sleep infinity",
            )
        )

    def create(self, offer: CatalogItem, *, name: str, public_key: str, disk_gb: int) -> str:
        body: dict[str, JsonValue] = {
            "name": name,
            "imageName": self.image,
            "gpuTypeIds": [offer.instance_name],
            "gpuCount": max(offer.gpu_count or 1, 1),
            "interruptible": bool(offer.spot),
            "containerDiskInGb": disk_gb,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": {"PUBLIC_KEY": public_key},
            "dockerStartCmd": ["bash", "-c", self._started(public_key)],
        }
        return str(self._launched(url=f"{_API}/pods", body=body)["id"])

    def machine(self, handle: str) -> Machine:
        try:
            pod = self._call("GET", url=f"{_API}/pods/{quote(handle, safe='')}", answer=dict)
        except HTTPError as error:
            if error.code == 404:
                return Machine(handle=handle, status="gone")
            raise
        return self._machine(pod)

    def machines(self) -> list[Machine]:
        return [
            self._machine(pod) for pod in self._call("GET", url=f"{_API}/pods", answer=list[dict])
        ]

    def terminate(self, handle: str) -> None:
        try:
            self._call("DELETE", url=f"{_API}/pods/{quote(handle, safe='')}", answer=dict)
        except HTTPError as error:
            if error.code != 404:
                raise


class RunPodBackend(CloudBackend):
    """Hold a RunPod pod as an ssh host: `mb host hold runpod --gpu-name "RTX 4090" --for 2h`."""

    name = "runpod"
    catalog_name = "runpod"
    client = RunPodApi
