"""Hyperstack's catalog and virtual machines against the API's recorded answers, without a key."""

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import HyperstackBackend
from mainboard.dispatch.backends.cloud.hyperstack import HyperstackApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import offer, refusal
from ..constants import SSH_KEY
from ..fakes import Recorded

_REGION = "CANADA-1"
_IMAGES_AT = ("GET", f"region={_REGION}")


def _flavor(name: str, *, gpu: str, count: int, stock: bool) -> dict:
    return {
        "name": name,
        "cpu": 28,
        "ram": 180,
        "disk": 100,
        "ephemeral": 750,
        "gpu": gpu,
        "gpu_count": count,
        "stock_available": stock,
    }


_FLAVORS = {
    "data": [
        {
            "gpu": "H100-80G-PCIe",
            "region_name": _REGION,
            "flavors": [
                _flavor("n3-H100x1", gpu="H100-80G-PCIe", count=1, stock=True),
                _flavor("n3-H100x8", gpu="H100-80G-PCIe", count=8, stock=False),
                _flavor("n1-cpu-small", gpu="", count=0, stock=True),
            ],
        },
        {
            "gpu": "RTX-A6000",
            "region_name": "NORWAY-1",
            "flavors": [_flavor("n1-A6000x2", gpu="RTX-A6000", count=2, stock=True)],
        },
    ]
}
_PRICEBOOK = [
    {"name": "H100-80G-PCIe", "value": "2.500000000"},
    {"name": "RTX-A6000", "value": "0.500000000"},
    {"name": "vCPU", "value": "0E-9"},
]
_IMAGES = {
    "images": [
        {
            "region_name": _REGION,
            "type": "Ubuntu",
            "images": [
                {"name": "Ubuntu Server 22.04 LTS R535 CUDA 12.2"},
                {"name": "Ubuntu Server 22.04 LTS R570 CUDA 12.8"},
                {"name": "Ubuntu Server 22.04 LTS R535"},
            ],
        }
    ]
}
_LISTINGS: dict[tuple[str, str], object] = {
    ("GET", "/core/flavors"): _FLAVORS,
    ("GET", "/pricebook"): _PRICEBOOK,
}


def test_a_flavor_in_stock_is_priced_by_its_gpu_model_times_its_count() -> None:
    offers = HyperstackApi(transport=Recorded(_LISTINGS)).offers()
    assert [
        (each.instance_name, each.location, each.gpu_name, each.gpu_count, each.price)
        for each in offers
    ] == [
        ("n3-H100x1", "CANADA-1", "H100", 1, 2.5),
        ("n1-A6000x2", "NORWAY-1", "A6000", 2, 1.0),
    ]


def test_a_cloud_with_no_gpuhunt_catalog_is_filtered_from_its_own_listing() -> None:
    backend = HyperstackBackend(transport=Recorded(_LISTINGS))
    assert [each.region for each in backend.catalog(gpu_name="A6000")] == ["NORWAY-1"]


def test_a_machine_boots_the_newest_cuda_image_in_a_region_environment_with_the_key() -> None:
    wire = Recorded(
        {
            ("POST", "/core/environments"): {"status": True},
            ("POST", "/core/keypairs"): {"status": True},
            _IMAGES_AT: _IMAGES,
            ("POST", "/core/virtual-machines"): {"instances": [{"id": 123}]},
        }
    )
    handle = HyperstackApi(transport=wire).create(
        offer(instance_name="n3-H100x1", location=_REGION),
        name="hold",
        public_key=SSH_KEY,
        disk_gb=0,
    )
    environment, keypair, _, vm = (body for _, _, body in wire.sent)
    assert handle == "123" and environment == {"name": "mainboard-canada-1", "region": "CANADA-1"}
    assert keypair["environment_name"] == "mainboard-canada-1" and keypair["public_key"] == SSH_KEY
    assert vm["image_name"] == "Ubuntu Server 22.04 LTS R570 CUDA 12.8"
    assert (vm["flavor_name"], vm["key_name"], vm["assign_floating_ip"]) == (
        "n3-H100x1",
        keypair["name"],
        True,
    )


def test_an_environment_and_keypair_that_already_exist_are_reused() -> None:
    taken = refusal(400, '{"message": "Environment already exists"}')
    wire = Recorded(
        {
            ("POST", "/core/environments"): taken,
            ("POST", "/core/keypairs"): refusal(409, "keypair name already in use"),
            _IMAGES_AT: _IMAGES,
            ("POST", "/core/virtual-machines"): {"instances": [{"id": 1}]},
        }
    )
    HyperstackApi(transport=wire).create(
        offer(instance_name="n3-H100x1", location=_REGION),
        name="hold",
        public_key=SSH_KEY,
        disk_gb=0,
    )
    assert wire.sent[-1][0] == "POST" and wire.sent[-1][1].endswith("/core/virtual-machines")


def test_any_other_refusal_to_make_an_environment_propagates() -> None:
    wire = Recorded({("POST", "/core/environments"): refusal(401, "bad api key")})
    with pytest.raises(Exception, match="401"):
        HyperstackApi(transport=wire).create(
            offer(instance_name="n3-H100x1", location=_REGION),
            name="n",
            public_key=SSH_KEY,
            disk_gb=0,
        )
    assert len(wire.sent) == 1


def test_a_flavor_out_of_stock_is_a_retry() -> None:
    wire = Recorded(
        {
            ("POST", "/core/environments"): {},
            ("POST", "/core/keypairs"): {},
            _IMAGES_AT: _IMAGES,
            ("POST", "/core/virtual-machines"): refusal(400, "Not enough stock for n3-H100x1"),
        }
    )
    with pytest.raises(CapacityGone):
        HyperstackApi(transport=wire).create(
            offer(instance_name="n3-H100x1", location=_REGION),
            name="n",
            public_key=SSH_KEY,
            disk_gb=0,
        )
    assert wire.sent[-1][1].endswith("/core/virtual-machines")


def test_a_region_with_no_cuda_image_is_refused() -> None:
    wire = Recorded(
        {
            ("POST", "/core/environments"): {},
            ("POST", "/core/keypairs"): {},
            _IMAGES_AT: {"images": []},
        }
    )
    with pytest.raises(MissionError, match="no Ubuntu CUDA image"):
        HyperstackApi(transport=wire).create(
            offer(instance_name="n3-H100x1", location=_REGION),
            name="n",
            public_key=SSH_KEY,
            disk_gb=0,
        )
    assert [method for method, _, _ in wire.sent] == ["POST", "POST", "GET"]


def test_hyperstack_authenticates_by_an_api_key_header_not_a_bearer() -> None:
    seen: list[tuple[str | None, str | None]] = []

    def wire(request) -> bytes:
        seen.append((request.get_header("Api_key"), request.get_header("Authorization")))
        return b"{}"

    HyperstackApi(transport=wire).machines()
    assert seen == [("test-key", None)]


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        pytest.param({"status": "CREATING", "floating_ip": None}, "pending", id="creating"),
        pytest.param({"status": "ACTIVE", "floating_ip": None}, "pending", id="no-address-yet"),
        pytest.param({"status": "ACTIVE", "floating_ip": "5.6.7.8"}, "running", id="active"),
        pytest.param({"status": "ERROR", "floating_ip": None}, "gone", id="failed"),
    ],
)
def test_a_machine_takes_ssh_once_active_with_a_floating_address(
    answer: dict, status: str
) -> None:
    wire = Recorded({("GET", "/core/virtual-machines/9"): {"instance": {"id": 9, **answer}}})
    found = HyperstackApi(transport=wire).machine("9")
    assert (found.status, found.user) == (status, "ubuntu")


def test_a_deleted_machine_is_gone_and_ending_it_again_tolerates_that() -> None:
    wire = Recorded(
        {
            ("GET", "/core/virtual-machines/9"): refusal(404, "{}"),
            ("DELETE", "/core/virtual-machines/9"): refusal(404, "{}"),
        }
    )
    api = HyperstackApi(transport=wire)
    api.terminate("9")
    assert api.machine("9").status == "gone"
