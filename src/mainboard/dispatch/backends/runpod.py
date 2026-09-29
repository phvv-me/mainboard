# RunPod pods as held machines, through RunPod's REST API (`rest.runpod.io/v1`).
#
# A pod is a container on a GPU host. It boots plain Ubuntu (the environment brings its own CUDA
# libraries) with a start command that installs and starts sshd and authorizes this machine's key,
# the way SkyPilot's RunPod provisioner does, then stays up; mb's setup does the rest over ssh.
# Port 22 is published on the host's public address under a mapped port.

import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote

from ..rentals import seeded
from .cloud import CapacityGone, CloudBackend, Machine

_API = "https://rest.runpod.io/v1"
# What RunPod says when the chosen GPU has no machine left, in its refusals' wording.
_NO_CAPACITY = ("no longer any instances available", "not enough", "unavailable", "no available")


def started(public_key: str) -> str:
    """The pod's start command: sshd installed and running with `public_key`, then held up."""
    return "\n".join(
        (
            "export DEBIAN_FRONTEND=noninteractive",
            "command -v sshd >/dev/null || (apt-get update -qq && apt-get install -y -qq "
            "openssh-server curl ca-certificates >/dev/null)",
            "mkdir -p /run/sshd && ssh-keygen -A",
            seeded(public_key),
            "/usr/sbin/sshd",
            "sleep infinity",
        )
    )


class RunPodBackend(CloudBackend):
    """Hold a RunPod pod as an ssh host: `mb host hold runpod --gpu-name "RTX 4090" --for 2h`."""

    name = "runpod"
    catalog_name = "runpod"
    key_variables = ("RUNPOD_API_KEY",)

    def create(self, offer: Any, *, name: str, public_key: str, disk_gb: int) -> str:
        body = {
            "name": name,
            "imageName": self.image,
            "gpuTypeIds": [offer.instance_name],
            "gpuCount": max(offer.gpu_count or 1, 1),
            "interruptible": bool(offer.spot),
            "containerDiskInGb": disk_gb,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": {"PUBLIC_KEY": public_key},
            "dockerStartCmd": ["bash", "-c", started(public_key)],
        }
        try:
            created = self.call("POST", f"{_API}/pods", body)
        except HTTPError as refused:
            said = refused.read().decode(errors="replace")
            if any(word in said.lower() for word in _NO_CAPACITY):
                raise CapacityGone(said[:200]) from refused
            raise
        return str(created["id"])

    def machine(self, handle: str) -> Machine:
        try:
            pod = self.call("GET", f"{_API}/pods/{quote(handle, safe='')}")
        except HTTPError as error:
            if error.code == 404:
                return Machine(handle=handle, status="gone")
            raise
        return self._machine(pod)

    def machines(self) -> list[Machine]:
        return [self._machine(pod) for pod in self.call("GET", f"{_API}/pods") or []]

    def terminate(self, handle: str) -> None:
        try:
            self.call("DELETE", f"{_API}/pods/{quote(handle, safe='')}")
        except HTTPError as error:
            if error.code != 404:
                raise

    @staticmethod
    def _machine(pod: dict) -> Machine:
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
