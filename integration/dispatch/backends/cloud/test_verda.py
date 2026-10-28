"""Verda instances against the API's recorded answers, without a key or a bill."""

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends.cloud.verda import VerdaApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import offer, refusal
from ..constants import SSH_KEY
from ..fakes import Recorded

_TOKEN = {"access_token": "tok", "expires_in": 3600}
_TYPES = [
    {
        "instance_type": "1B200.30V",
        "supported_os": [
            "ubuntu-22.04-cuda-12.8-open-docker",
            "ubuntu-24.04-cuda-13.0-open-docker",
            "ubuntu-24.04-cuda-13.0-open",
            "ubuntu-24.04",
        ],
    }
]


def test_verda_exchanges_its_client_pair_once_and_boots_the_newest_cuda_image() -> None:
    wire = Recorded(
        {
            ("POST", "/oauth2/token"): _TOKEN,
            ("GET", "/ssh-keys"): [],
            ("POST", "/ssh-keys"): "key-1",
            ("GET", "/instance-types"): _TYPES,
            ("POST", "/instances"): "inst-1",
        }
    )
    handle = VerdaApi(transport=wire).create(
        offer(instance_name="1B200.30V", location="FIN-03"),
        name="hold-a",
        public_key=SSH_KEY,
        disk_gb=100,
    )
    created = next(body for method, url, body in wire.sent if url.endswith("/v1/instances"))
    assert handle == "inst-1" and created["image"] == "ubuntu-24.04-cuda-13.0-open"
    assert created["ssh_key_ids"] == ["key-1"] and created["location_code"] == "FIN-03"
    assert sum(url.endswith("/oauth2/token") for _, url, _ in wire.sent) == 1


def test_verda_reads_a_running_instance_and_ends_it_with_its_os_volume() -> None:
    running = {
        "id": "i-1",
        "status": "running",
        "ip": "1.2.3.4",
        "os_volume_id": "v-1",
        "price_per_hour": "3.17",
    }
    wire = Recorded(
        {
            ("POST", "/oauth2/token"): _TOKEN,
            ("GET", "/instances/i-1"): running,
            ("PUT", "/instances"): {},
        }
    )
    api = VerdaApi(transport=wire)
    found = api.machine("i-1")
    api.terminate("i-1")
    ended = next(body for method, _, body in wire.sent if method == "PUT")
    assert (found.status, found.host, found.user, found.usd_hr) == (
        "running",
        "1.2.3.4",
        "root",
        3.17,
    )
    assert ended == {
        "action": "delete",
        "id": "i-1",
        "volume_ids": ["v-1"],
        "delete_permanently": True,
    }


def test_verda_no_capacity_is_a_retry_and_a_failed_boot_reads_as_gone() -> None:
    wire = Recorded(
        {
            ("POST", "/oauth2/token"): _TOKEN,
            ("GET", "/ssh-keys"): [{"id": "key-1", "key": SSH_KEY}],
            ("GET", "/instance-types"): _TYPES,
            ("POST", "/instances"): refusal(503, "no capacity"),
            ("GET", "/instances/i-2"): {"id": "i-2", "status": "no_capacity"},
            ("GET", "/instances/i-3"): {"id": "i-3", "status": "validating"},
        }
    )
    api = VerdaApi(transport=wire)
    with pytest.raises(CapacityGone):
        api.create(offer(instance_name="1B200.30V"), name="n", public_key=SSH_KEY, disk_gb=50)
    assert (api.machine("i-2").status, api.machine("i-3").status) == ("gone", "pending")


def test_verda_refuses_a_client_id_whose_secret_is_not_set(monkeypatch) -> None:
    monkeypatch.delenv("VERDA_CLIENT_SECRET")
    wire = Recorded({})
    with pytest.raises(MissionError):
        VerdaApi(transport=wire).key()
    assert wire.sent == []
