"""The cloud backends against their APIs' recorded answers: what each call sends, and what a
machine reads as, without a key or a bill. A live rent needs the owner's key (`host hold`)."""

import io
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from mainboard.dispatch.backends import LambdaBackend, RunPodBackend
from mainboard.dispatch.backends.cloud import CapacityGone, card


class Recorded:
    """A transport answering each (method, path suffix) from a table, keeping every request."""

    def __init__(self, answers: dict[tuple[str, str], object]) -> None:
        self.answers = answers
        self.sent: list[tuple[str, str, dict | None]] = []

    def __call__(self, request):
        body = json.loads(request.data) if request.data else None
        self.sent.append((request.get_method(), request.full_url, body))
        for (method, suffix), answer in self.answers.items():
            if request.get_method() == method and request.full_url.endswith(suffix):
                if isinstance(answer, HTTPError):
                    raise answer
                return SimpleNamespace(status=200, read=lambda a=answer: json.dumps(a).encode())
        raise AssertionError(f"unexpected {request.get_method()} {request.full_url}")


def refusal(code: int, said: str) -> HTTPError:
    return HTTPError("https://x", code, said, {}, io.BytesIO(said.encode()))


@pytest.fixture(autouse=True)
def keys(monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test-runpod")
    monkeypatch.setenv("LAMBDA_API_KEY", "test-lambda")


def offer(**fields) -> SimpleNamespace:
    return SimpleNamespace(
        **{"instance_name": "x", "location": "r", "gpu_count": 1, "spot": False, **fields}
    )


def test_a_card_reads_the_same_however_it_is_spelled() -> None:
    assert card("NVIDIA_GeForce_RTX_4090") == card("rtx 4090") == "RTX4090"


def test_runpod_creates_a_pod_that_opens_ssh_with_the_key() -> None:
    wire = Recorded({("POST", "/pods"): {"id": "pod-1"}})
    handle = RunPodBackend(transport=wire).create(
        offer(instance_name="NVIDIA GeForce RTX 4090"),
        name="hold-a",
        public_key="ssh-ed25519 K",
        disk_gb=80,
    )
    _, url, body = wire.sent[0]
    assert handle == "pod-1" and url == "https://rest.runpod.io/v1/pods"
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"] and body["ports"] == ["22/tcp"]
    assert "sshd" in body["dockerStartCmd"][-1] and "ssh-ed25519 K" in body["dockerStartCmd"][-1]


def test_runpod_reads_a_running_pod_by_its_mapped_ssh_port() -> None:
    pod = {
        "id": "pod-1",
        "desiredStatus": "RUNNING",
        "publicIp": "1.2.3.4",
        "portMappings": {"22": 40022},
    }
    found = RunPodBackend(transport=Recorded({("GET", "/pods/pod-1"): pod})).machine("pod-1")
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
        RunPodBackend(transport=wire).create(offer(), name="n", public_key="k", disk_gb=10)


def test_a_gone_pod_reads_as_gone_and_ends_quietly() -> None:
    wire = Recorded(
        {("GET", "/pods/p"): refusal(404, "{}"), ("DELETE", "/pods/p"): refusal(404, "{}")}
    )
    backend = RunPodBackend(transport=wire)
    assert backend.machine("p").status == "gone"
    backend.terminate("p")


def test_lambda_registers_the_key_once_then_launches_with_it() -> None:
    wire = Recorded(
        {
            ("GET", "/ssh-keys"): {"data": []},
            ("POST", "/ssh-keys"): {"data": {}},
            ("POST", "/instance-operations/launch"): {"data": {"instance_ids": ["i-1"]}},
        }
    )
    handle = LambdaBackend(transport=wire).create(
        offer(instance_name="gpu_1x_h100_pcie", location="us-east-1"),
        name="hold-b",
        public_key="ssh-ed25519 K",
        disk_gb=0,
    )
    launch = next(body for method, url, body in wire.sent if url.endswith("/launch"))
    registered = next(
        body for method, url, body in wire.sent if method == "POST" and url.endswith("/ssh-keys")
    )
    assert handle == "i-1" and launch["ssh_key_names"] == [registered["name"]]
    assert (
        launch["instance_type_name"] == "gpu_1x_h100_pcie" and launch["region_name"] == "us-east-1"
    )


def test_lambda_reads_an_active_instance_as_ubuntu_over_ssh() -> None:
    active = {
        "data": {
            "id": "i-1",
            "status": "active",
            "ip": "5.6.7.8",
            "instance_type": {"price_cents_per_hour": 249},
        }
    }
    found = LambdaBackend(transport=Recorded({("GET", "/instances/i-1"): active})).machine("i-1")
    assert (found.status, found.host, found.user, found.usd_hr) == (
        "running",
        "5.6.7.8",
        "ubuntu",
        2.49,
    )


def test_lambda_insufficient_capacity_is_a_retry() -> None:
    wire = Recorded(
        {
            ("GET", "/ssh-keys"): {"data": [{"name": "x"}]},
            ("POST", "/ssh-keys"): {"data": {}},
            ("POST", "/instance-operations/launch"): refusal(
                400, '{"error": {"code": "instance-operations/launch/insufficient-capacity"}}'
            ),
        }
    )
    with pytest.raises(CapacityGone):
        LambdaBackend(transport=wire).create(offer(), name="n", public_key="k", disk_gb=0)


def test_offers_list_every_cloud_and_mark_what_mb_rents(mb) -> None:
    ran = mb("host", "offers", "RTX 4090", "--limit", "40", timeout=300)
    assert ran.code == 0, ran.said
    rows = ran.out.splitlines()
    assert rows[0].startswith("provider") and any("vast" in row for row in rows[1:])
