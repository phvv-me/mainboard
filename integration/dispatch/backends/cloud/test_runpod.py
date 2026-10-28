"""RunPod pods against the API's recorded answers: what each call sends, and what a pod reads as,
without a key or a bill. A live rent needs the owner's key (`host hold`)."""

import pytest

from mainboard.dispatch.backends import RunPodBackend
from mainboard.dispatch.backends.cloud.core import _WAITER
from mainboard.dispatch.backends.cloud.runpod import RunPodApi
from mainboard.dispatch.backends.exceptions import CapacityGone
from mainboard.dispatch.hosts import rentals

from ..builders import offer, refusal
from ..fakes import Recorded


def test_runpod_creates_a_pod_that_opens_ssh_with_the_key() -> None:
    wire = Recorded({("POST", "/pods"): {"id": "pod-1"}})
    handle = RunPodApi(transport=wire).create(
        offer(instance_name="NVIDIA GeForce RTX 4090"),
        name="hold-a",
        public_key="ssh-ed25519 K",
        disk_gb=80,
    )
    _, url, body = wire.sent[0]
    assert handle == "pod-1" and url == "https://rest.runpod.io/v1/pods"
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"] and body["ports"] == ["22/tcp"]
    assert body["cloudType"] == "SECURE" and body["minDownloadMbps"] == 500
    backend = RunPodBackend(transport=wire)
    assert all(backend.rentable(offer(location=place)) for place in ("US-TX-3", "EUR-IS-1"))
    assert not backend.rentable(offer(location="CA"))
    assert "sshd" in body["dockerStartCmd"][-1] and "ssh-ed25519 K" in body["dockerStartCmd"][-1]


def test_runpod_reads_a_running_pod_by_its_mapped_ssh_port() -> None:
    pod = {
        "id": "pod-1",
        "desiredStatus": "RUNNING",
        "publicIp": "1.2.3.4",
        "portMappings": {"22": 40022},
    }
    found = RunPodApi(transport=Recorded({("GET", "/pods/pod-1"): pod})).machine("pod-1")
    assert (found.status, found.host, found.port, found.user) == (
        "running",
        "1.2.3.4",
        40022,
        "root",
    )


def test_runpod_out_of_capacity_is_a_retry_not_a_failure() -> None:
    wire = Recorded(
        {("POST", "/pods"): refusal(400, "There are no longer any instances available")}
    )
    with pytest.raises(CapacityGone):
        RunPodApi(transport=wire).create(offer(), name="n", public_key="k", disk_gb=10)
    assert [method for method, _, _ in wire.sent] == ["POST"]


def test_a_gone_pod_reads_as_gone_and_ends_quietly() -> None:
    wire = Recorded(
        {("GET", "/pods/p"): refusal(404, "{}"), ("DELETE", "/pods/p"): refusal(404, "{}")}
    )
    api = RunPodApi(transport=wire)
    assert api.machine("p").status == "gone"
    api.terminate("p")


def test_a_cloud_job_reads_its_verdict_from_the_exit_file_the_waiter_leaves(monkeypatch) -> None:
    running = {
        "id": "pod-1",
        "desiredStatus": "RUNNING",
        "publicIp": "1.2.3.4",
        "portMappings": {"22": 1},
    }
    backend = RunPodBackend(transport=Recorded({("GET", "/pods/pod-1"): running}))
    monkeypatch.setattr(backend, "_read", lambda handle, path: "")
    assert backend.state("pod-1").verdict == "running"
    monkeypatch.setattr(backend, "_read", lambda handle, path: "3\n")
    finished = backend.state("pod-1")
    assert (finished.verdict, finished.exit_code) == ("failed", 3)
    assert rentals.Rental.LAUNCH in _WAITER and "/tmp/mainboard.exit" in _WAITER
