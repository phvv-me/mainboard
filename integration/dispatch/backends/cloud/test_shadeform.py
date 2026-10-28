"""Shadeform's catalog and instances against the API's recorded answers, without a key."""

import pytest

from mainboard.dispatch.backends import ShadeformBackend
from mainboard.dispatch.backends.cloud.shadeform import ShadeformApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import refusal
from ..constants import SSH_KEY
from ..fakes import Recorded

_TYPES = ("GET", "/instances/types")


def _kind(*, gpu: str, count: int, cents: int, regions: list[dict]) -> dict:
    return {
        "cloud": "lambda",
        "shade_instance_type": f"{count}x{gpu}",
        "cloud_instance_type": "x",
        "gpu_type": gpu,
        "num_gpus": count,
        "hourly_price": cents,
        "vcpus": 16,
        "memory_in_gb": 200,
        "storage_in_gb": 512,
        "boot_time": {"min_boot_in_sec": 120, "max_boot_in_sec": 600},
        "configuration": {
            "gpu_type": gpu,
            "gpu_manufacturer": "nvidia",
            "vram_per_gpu_in_gb": 80,
            "os_options": [
                "ubuntu20.04_cuda12.2_shade_os",
                "ubuntu22.04_cuda12.8_shade_os",
                "ubuntu22.04_cuda12.2_shade_os",
                "docker_only",
            ],
        },
        "availability": regions,
    }


def _where(
    region: str, *, available: bool = True, rental: str = "on_demand", price: str = ""
) -> dict:
    return {
        "region": region,
        "available": available,
        "rental_type": rental,
        "display_name": region,
        "hourly_price": price,
    }


_LISTING = {
    "instance_types": [
        _kind(
            gpu="H100",
            count=1,
            cents=329,
            regions=[_where("us-east-1"), _where("us-west-1", available=False)],
        ),
        _kind(
            gpu="A100_80G",
            count=8,
            cents=1200,
            regions=[_where("eu-1", rental="spot", price="4.5")],
        ),
    ]
}


def test_only_available_regions_are_offered_under_the_cloud_that_owns_the_hardware() -> None:
    offers = ShadeformApi(transport=Recorded({_TYPES: _LISTING})).offers()
    assert [(each.location, each.price, each.spot) for each in offers] == [
        ("lambda/us-east-1", 3.29, False),
        ("lambda/eu-1", 4.5, True),
    ]
    assert offers[1].gpu_name == "A100" and offers[0].provider_data["boot_max_s"] == 600


def test_a_cloud_with_no_gpuhunt_catalog_is_filtered_from_its_own_listing() -> None:
    backend = ShadeformBackend(transport=Recorded({_TYPES: _LISTING}))
    assert [each.region for each in backend.catalog(gpu_name="H100")] == ["lambda/us-east-1"]
    assert backend.catalog(gpu_name="H100", gpus=8) == []


def test_a_machine_is_created_in_its_cloud_with_the_newest_cuda_image_and_the_key_by_id() -> None:
    wire = Recorded(
        {
            _TYPES: _LISTING,
            ("GET", "/sshkeys"): {"ssh_keys": [{"id": "k-1", "public_key": SSH_KEY}]},
            ("POST", "/instances/create"): {"id": "m-1"},
        }
    )
    api = ShadeformApi(transport=wire)
    handle = api.create(api.offers()[0], name="hold", public_key=SSH_KEY, disk_gb=0)
    body = wire.sent[-1][2]
    assert handle == "m-1" and (body["cloud"], body["region"]) == ("lambda", "us-east-1")
    assert (body["os"], body["ssh_key_id"], body["rental_type"]) == (
        "ubuntu22.04_cuda12.8_shade_os",
        "k-1",
        "on_demand",
    )


def test_shadeform_registers_a_key_it_does_not_know() -> None:
    wire = Recorded(
        {("GET", "/sshkeys"): {"ssh_keys": None}, ("POST", "/sshkeys/add"): {"id": "new"}}
    )
    assert ShadeformApi(transport=wire).registered(SSH_KEY) == "new"
    assert wire.sent[-1][2]["public_key"] == SSH_KEY


def test_a_refused_type_is_a_retry() -> None:
    wire = Recorded(
        {
            _TYPES: _LISTING,
            ("GET", "/sshkeys"): {"ssh_keys": [{"id": "k", "public_key": SSH_KEY}]},
            ("POST", "/instances/create"): refusal(400, "instance type not available"),
        }
    )
    api = ShadeformApi(transport=wire)
    with pytest.raises(CapacityGone):
        api.create(api.offers()[0], name="n", public_key=SSH_KEY, disk_gb=0)
    assert wire.sent[-1][1].endswith("/instances/create")


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        pytest.param({"status": "pending", "ip": ""}, "pending", id="creating"),
        pytest.param({"status": "active", "ip": "5.6.7.8"}, "running", id="active"),
        pytest.param({"status": "deleted", "ip": "5.6.7.8"}, "gone", id="deleted"),
    ],
)
def test_an_instance_takes_ssh_only_once_active_with_an_address(answer: dict, status: str) -> None:
    info = {"id": "m-1", "ssh_user": "shadeform", "ssh_port": 22, "hourly_price": 329, **answer}
    wire = Recorded({("GET", "/instances/m-1/info"): info})
    found = ShadeformApi(transport=wire).machine("m-1")
    assert (found.status, found.user, found.usd_hr) == (status, "shadeform", 3.29)


def test_a_deleted_instance_is_gone_and_ending_it_again_tolerates_that() -> None:
    wire = Recorded(
        {
            ("GET", "/instances/x/info"): refusal(404, "{}"),
            ("POST", "/instances/x/delete"): refusal(404, "{}"),
        }
    )
    api = ShadeformApi(transport=wire)
    api.terminate("x")
    assert api.machine("x").status == "gone"
