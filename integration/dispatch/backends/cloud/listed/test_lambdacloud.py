"""Lambda Cloud instances against the API's recorded answers, without a key or a bill."""

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import LambdaBackend, ProviderBackend
from mainboard.dispatch.backends.cloud.listed.lambdacloud import LambdaApi
from mainboard.dispatch.backends.exceptions import CapacityGone
from mainboard.enums import Bound

from ....hosts.builders import lease
from ...builders import offer, refusal
from ...fakes import Recorded


def test_lambda_and_every_card_billed_provider_rent_only_behind_a_declared_card_limit() -> None:
    caps = [ProviderBackend.find(kind).cap(card_usd=40.0) for kind in ProviderBackend.names()]
    with pytest.raises(MissionError, match=r"lambda bills a card .*serve grant card lambda --usd"):
        LambdaBackend.cap(card_usd=0.0)
    assert {cap.provider: (cap.bound, cap.limit_usd) for cap in caps} == {
        "digitalocean": (Bound.CARD, 40.0),
        "hotaisle": (Bound.PREPAID, 0.0),
        "hpc-ai": (Bound.PREPAID, 0.0),
        "hyperstack": (Bound.PREPAID, 0.0),
        "lambda": (Bound.CARD, 40.0),
        "modal": (Bound.THRESHOLD, 0.0),
        "runpod": (Bound.PREPAID, 0.0),
        "shadeform": (Bound.THRESHOLD, 0.0),
        "vast": (Bound.PREPAID, 0.0),
        "verda": (Bound.PREPAID, 0.0),
        "vultr": (Bound.CARD, 40.0),
    }


def test_lambda_registers_the_key_once_then_launches_with_it() -> None:
    wire = Recorded(
        {
            ("GET", "/ssh-keys"): {"data": []},
            ("POST", "/ssh-keys"): {"data": {}},
            ("POST", "/instance-operations/launch"): {"data": {"instance_ids": ["i-1"]}},
        }
    )
    handle = LambdaApi(transport=wire).create(
        offer(instance_name="gpu_1x_h100_pcie", location="us-east-1"),
        lease=lease(),
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
    found = LambdaApi(transport=Recorded({("GET", "/instances/i-1"): active})).machine("i-1")
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
        LambdaApi(transport=wire).create(
            offer(), lease=lease(), name="n", public_key="k", disk_gb=0
        )
    assert wire.sent[-1][1].endswith("/instance-operations/launch")
