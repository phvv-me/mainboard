"""Lambda Cloud instances against the API's recorded answers, without a key or a bill."""

import pytest

from mainboard.dispatch.backends.cloud.lambdacloud import LambdaApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import offer, refusal
from ..fakes import Recorded


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
        LambdaApi(transport=wire).create(offer(), name="n", public_key="k", disk_gb=0)
    assert wire.sent[-1][1].endswith("/instance-operations/launch")
