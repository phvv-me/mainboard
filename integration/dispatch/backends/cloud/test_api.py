"""What every cloud's API shares: where a handle is read, and which image a machine boots."""

import pytest

from mainboard.dispatch.backends.cloud.api import CloudApi
from mainboard.dispatch.backends.cloud.curated.hotaisle import HotAisleApi
from mainboard.dispatch.backends.cloud.curated.hyperstack import HyperstackApi
from mainboard.dispatch.backends.cloud.curated.runpod import RunPodApi
from mainboard.dispatch.backends.cloud.curated.shadeform import ShadeformApi
from mainboard.dispatch.backends.cloud.listed.digitalocean import DigitalOceanApi
from mainboard.dispatch.backends.cloud.listed.verda import VerdaApi

from ...hosts.builders import lease
from ..builders import offer, refusal
from ..constants import SSH_KEY
from ..fakes import Recorded

_TOKEN = ("POST", "/oauth2/token")


@pytest.mark.parametrize(
    ("api", "tail"),
    [
        pytest.param(HotAisleApi, "/virtual_machines/a%2Fb%20c/", id="hotaisle"),
        pytest.param(HyperstackApi, "/core/virtual-machines/a%2Fb%20c", id="hyperstack"),
        pytest.param(RunPodApi, "/pods/a%2Fb%20c", id="runpod"),
        pytest.param(ShadeformApi, "/instances/a%2Fb%20c/info", id="shadeform"),
        pytest.param(DigitalOceanApi, "/droplets/a%2Fb%20c", id="digitalocean"),
        pytest.param(VerdaApi, "/instances/a%2Fb%20c", id="verda"),
    ],
)
def test_a_handle_is_one_quoted_segment_of_the_url_its_machine_is_read_at(
    api: type[CloudApi], tail: str
) -> None:
    wire = Recorded({("GET", tail): refusal(404, "{}"), _TOKEN: {"access_token": "t"}})
    assert api(transport=wire).machine("a/b c").status == "gone"


def test_a_newer_cuda_beats_a_newer_ubuntu_and_versions_compare_as_numbers() -> None:
    supported = ["ubuntu-24.04-cuda-12.9-open", "ubuntu-22.04-cuda-12.10-open"]
    kinds = [{"instance_type": "t", "supported_os": supported}]
    wire = Recorded(
        {
            _TOKEN: {"access_token": "t"},
            ("GET", "/ssh-keys"): [{"id": "k", "key": SSH_KEY}],
            ("GET", "/instance-types"): kinds,
            ("POST", "/instances"): "i-1",
        }
    )
    VerdaApi(transport=wire).create(
        offer(instance_name="t"), lease=lease(), name="n", public_key=SSH_KEY, disk_gb=50
    )
    assert wire.sent[-1][2]["image"] == supported[1]
