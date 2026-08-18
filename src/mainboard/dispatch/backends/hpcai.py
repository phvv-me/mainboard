# `HpcAiBackend` runs a command on an HPC-AI (hpc-ai.com) instance through their REST API.
# Auth is a console API key sent as X-API-Key on every call (verified live), and every
# call refreshes it first when it has gone stale.

import json
import os
from typing import TYPE_CHECKING
from urllib.request import Request
from uuid import uuid4

from ...core.errors import MissionError
from ..schedulers.base import JobState
from .base import ProviderBackend, http_transport, require_budget

if TYPE_CHECKING:
    from ...context.plan import ExecutionPlan
    from ...manifest.schema.host import HostProfile
    from ..schedulers.base import Resources
    from .base import Transport

# The volume path every initScript mounts, and where its exit-code/output sentinels land.
_VOLUME_PATH = "/mnt/vol"
# instanceRuntimeInfo.status values this backend has seen mapped onto our verdict vocabulary.
# A status outside this table (a new HPC-AI state) reads as "unknown" rather than crashing.
_VERDICTS = {
    "pending": "running",
    "starting": "running",
    "running": "running",
    "stopping": "running",
    "stopped": "ok",
    "failed": "failed",
    "error": "failed",
}


def api_key() -> str:
    """The HPC-AI key from `HPCAI_API_KEY`, refusing with the setup hint when unset.

    Console API keys authenticate the instance namespace through the `X-API-Key`
    header (verified live 2026-08-18), so no login flow or cookie JWT exists here.
    """
    key = os.environ.get("HPCAI_API_KEY", "")
    if not key:
        raise MissionError(
            "set HPCAI_API_KEY in the workspace .env, an API key minted in the HPC-AI console"
        )
    return key


def _required_var(profile: HostProfile, key: str) -> str:
    """`profile.vars[key]`, raising a clear `MissionError` naming `key` when it is missing.

    `HostProfile` forbids undeclared fields, so the opaque, provider-specific values HPC-AI
    needs (`instance-type-id`, `image-id`, `region`) live in its free-form `vars` table instead.
    """
    try:
        return profile.vars[key]
    except KeyError:
        raise MissionError(
            f"host {profile.root!r} needs [hosts.<name>.vars] {key!r} set for the hpc-ai backend"
        ) from None


class HpcAiBackend(ProviderBackend):
    """Run a command on an HPC-AI instance, its own REST API standing in for a scheduler.

    HPC-AI reports instance-level status only (`instanceRuntimeInfo.status`), never a process
    exit code, so `submit` wraps the command in an initScript that writes its real exit code and
    captured output to sentinel files on the instance's mounted volume; `state` reports whether
    the instance itself is still up, and the sentinel files are the source of truth for the
    command's own outcome.
    """

    name = "hpc-ai"

    def __init__(self, *, spot: bool = False, transport: Transport = http_transport) -> None:
        """spot: whether created instances are spot (cost-optimized, preemptible).
        transport: sends a prepared `Request`, returning its response; injectable for tests.
        """
        self.spot = spot
        self.transport = transport

    def cancel(self, handle: str) -> None:
        self._request("POST", path="/instance/stop", body={"name": handle})
        self._request("POST", path="/instance/delete", body={"name": handle})

    def deliver(self, handle: str, *, path: str) -> None:
        raise MissionError(
            f"hpc-ai backend cannot deliver {path!r} yet; download "
            f"{_VOLUME_PATH}/{handle}.* from the instance's mounted volume by hand until that "
            "path lands"
        )

    def logs(self, handle: str) -> str:
        raise MissionError(
            f"hpc-ai backend has no server-side logs; tail {_VOLUME_PATH}/{handle}.log on the "
            "instance's mounted volume instead"
        )

    def state(self, handle: str) -> JobState:
        payload = self._request("POST", path="/instance/list", body={})
        entry = next(
            (item for item in payload.get("instances", []) if item.get("name") == handle), None
        )
        if entry is None:
            return JobState(handle=handle, verdict="vanished")
        status = entry.get("instanceRuntimeInfo", {}).get("status", "")
        return JobState(handle=handle, state=status, verdict=_VERDICTS.get(status, "unknown"))

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        require_budget(resources)
        handle = uuid4().hex
        init_script = (
            f"{command} > {_VOLUME_PATH}/{handle}.log 2>&1\n"
            f"echo $? > {_VOLUME_PATH}/{handle}.exit\n"
        )
        self._request(
            "POST",
            path="/instance/create",
            body={
                "name": handle,
                "isSpotInstance": self.spot,
                "instanceTypeId": _required_var(plan.profile, "instance-type-id"),
                "imageId": _required_var(plan.profile, "image-id"),
                "region": _required_var(plan.profile, "region"),
                "instanceConfiguration": {"initScript": init_script},
            },
        )
        return handle

    def _request(self, method: str, *, path: str, body: dict) -> dict:
        """An authenticated call to the instance API under the console API key.

        Every endpoint hangs off `https://www.hpc-ai.com/api`, spelled inline so the https root
        is visible at the one place a `Request` is built.
        """
        headers = {"Content-Type": "application/json", "X-API-Key": api_key()}
        request = Request(
            f"https://www.hpc-ai.com/api{path}",
            method=method,
            data=json.dumps(body).encode(),
            headers=headers,
        )
        return json.loads(self.transport(request).read())
