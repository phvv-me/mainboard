"""DigitalOcean GPU Droplets against the API's recorded answers, without a key or a bill."""

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends.cloud.digitalocean import DigitalOceanApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import offer, refusal
from ..constants import SSH_FINGERPRINT, SSH_KEY
from ..fakes import Recorded

_KNOWN = ("GET", f"/account/keys/{SSH_FINGERPRINT}")


@pytest.mark.parametrize(
    ("plan", "image"),
    [
        pytest.param("gpu-h100x1-80gb", "gpu-h100x1-base", id="an-h100-plan-keeps-its-count"),
        pytest.param("gpu-h200x8-1128gb", "gpu-h200x8-base", id="an-h200-plan-keeps-its-count"),
        pytest.param("gpu-mi300x1-192gb", "gpu-amd-base", id="an-amd-plan-takes-the-amd-image"),
    ],
)
def test_a_gpu_plan_boots_from_its_ai_ml_ready_base_image(plan: str, image: str) -> None:
    wire = Recorded({_KNOWN: {}, ("POST", "/droplets"): {"droplet": {"id": 7}}})
    handle = DigitalOceanApi(transport=wire).create(
        offer(instance_name=plan, location="tor1"), name="hold", public_key=SSH_KEY, disk_gb=0
    )
    body = wire.sent[-1][2]
    assert handle == "7" and (body["size"], body["image"], body["region"]) == (plan, image, "tor1")
    assert body["ssh_keys"] == [SSH_FINGERPRINT]


def test_digitalocean_registers_a_key_it_does_not_know_by_its_fingerprint() -> None:
    wire = Recorded({_KNOWN: refusal(404, "{}"), ("POST", "/account/keys"): {"ssh_key": {}}})
    registered = DigitalOceanApi(transport=wire).registered(SSH_KEY)
    method, _, posted = wire.sent[-1]
    assert registered == SSH_FINGERPRINT and method == "POST" and posted["public_key"] == SSH_KEY


def test_digitalocean_reads_an_active_droplet_by_its_public_address() -> None:
    active = {
        "droplet": {
            "id": 7,
            "status": "active",
            "networks": {
                "v4": [
                    {"type": "private", "ip_address": "10.0.0.2"},
                    {"type": "public", "ip_address": "5.6.7.8"},
                ]
            },
            "size": {"price_hourly": 6.74, "gpu_info": {"model": "nvidia_h100"}},
        }
    }
    wire = Recorded(
        {("GET", "/droplets/7"): active, ("DELETE", "/droplets/8"): refusal(404, "{}")}
    )
    api = DigitalOceanApi(transport=wire)
    found = api.machine("7")
    api.terminate("8")
    assert (found.status, found.host, found.user, found.usd_hr, found.gpu) == (
        "running",
        "5.6.7.8",
        "root",
        6.74,
        "nvidia_h100",
    )


def test_digitalocean_capacity_refusal_is_a_retry() -> None:
    wire = Recorded({_KNOWN: {}, ("POST", "/droplets"): refusal(422, "no capacity in tor1")})
    with pytest.raises(CapacityGone):
        DigitalOceanApi(transport=wire).create(
            offer(instance_name="gpu-h100x1-80gb"), name="n", public_key=SSH_KEY, disk_gb=0
        )
    assert [method for method, _, _ in wire.sent] == ["GET", "POST"]


def test_digitalocean_refuses_a_plan_that_is_not_a_gpu_plan_before_any_call() -> None:
    wire = Recorded({})
    with pytest.raises(MissionError, match="not a GPU plan"):
        DigitalOceanApi(transport=wire).create(
            offer(instance_name="s-2vcpu-4gb"), name="n", public_key=SSH_KEY, disk_gb=0
        )
    assert wire.sent == []
