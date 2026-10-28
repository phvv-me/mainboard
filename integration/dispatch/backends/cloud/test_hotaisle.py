"""Hot Aisle virtual machines against the API's recorded answers, without a key."""

import pytest
from gpuhunt import CatalogItem
from gpuhunt._internal.models import JSONObject

from mainboard.dispatch.backends import HotAisleBackend
from mainboard.dispatch.backends.cloud.hotaisle import HotAisleApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import offer, refusal
from ..constants import SSH_KEY
from ..fakes import Recorded

_KEYS = ("GET", "/user/ssh_keys/")
_VMS = "/teams/test-key/virtual_machines/"
_SPECS: JSONObject = {
    "cpu_cores": 13,
    "ram_capacity": 240 * 1024**3,
    "disk_capacity": 12 * 1024**4,
    "gpus": [{"count": 1, "manufacturer": "AMD", "model": "MI300X"}],
}


def _row(specs: str) -> CatalogItem:
    """The MI300X row gpuhunt lists, keeping its availability specs under `specs`."""
    row = offer(instance_name="vm-mi300x-1", location="us-michigan-1")
    data: JSONObject = {specs: _SPECS}
    row.provider_data = data
    return row


def test_a_machine_is_created_from_the_specs_its_availability_entry_published() -> None:
    wire = Recorded(
        {_KEYS: [], ("POST", "/user/ssh_keys/"): {}, ("POST", _VMS): {"deployment_id": "d-1"}}
    )
    handle = HotAisleApi(transport=wire).create(
        _row("vm_specs"), name="hold", public_key=SSH_KEY, disk_gb=0
    )
    assert handle == "d-1" and wire.sent[-1][2] == _SPECS
    assert wire.sent[-1][1] == f"https://admin.hotaisle.app/api{_VMS}"


def test_hot_aisle_sends_its_token_scheme_rather_than_a_bearer() -> None:
    seen: list[str] = []

    def wire(request) -> bytes:
        seen.append(request.get_header("Authorization"))
        return b"null"

    HotAisleApi(transport=wire).machines()
    assert seen == ["Token test-key"]


def test_a_key_already_on_the_user_is_not_registered_again() -> None:
    wire = Recorded({_KEYS: [{"public_key": SSH_KEY}], ("POST", _VMS): {"deployment_id": "d-1"}})
    HotAisleApi(transport=wire).create(
        _row("vm_specs"), name="hold", public_key=SSH_KEY, disk_gb=0
    )
    assert [method for method, _, _ in wire.sent] == ["GET", "POST"]


def test_a_missing_machine_type_is_a_retry() -> None:
    wire = Recorded({_KEYS: [{"public_key": SSH_KEY}], ("POST", _VMS): refusal(404, "{}")})
    with pytest.raises(CapacityGone):
        HotAisleApi(transport=wire).create(
            _row("vm_specs"), name="n", public_key=SSH_KEY, disk_gb=0
        )
    assert wire.sent[-1][0] == "POST"


def test_a_ready_machine_is_reached_through_its_nat_port_as_the_hotaisle_user() -> None:
    ready = {
        "deployment_id": "d-1",
        "status": "ready",
        "name": "vm-01",
        "ip_address": "10.0.0.5",
        "ssh_access": {"ip_address": "203.0.113.10", "port": 2222},
        "gpus": [{"count": 1, "model": "MI300X"}],
    }
    wire = Recorded({("GET", f"{_VMS}d-1/"): ready, ("GET", f"{_VMS}d-2/"): refusal(404, "{}")})
    api = HotAisleApi(transport=wire)
    found = api.machine("d-1")
    assert (found.status, found.host, found.port, found.user, found.gpu) == (
        "running",
        "203.0.113.10",
        2222,
        "hotaisle",
        "MI300X",
    )
    assert api.machine("d-2").status == "gone"


def test_hot_aisle_lists_nothing_as_null_and_deletes_with_force() -> None:
    wire = Recorded({("GET", _VMS): None, ("DELETE", f"{_VMS}d-1/?force=true"): {}})
    api = HotAisleApi(transport=wire)
    assert api.machines() == []
    api.terminate("d-1")
    assert wire.sent[-1][0] == "DELETE"


def test_hot_aisle_rents_virtual_machines_and_not_bare_metal() -> None:
    backend = HotAisleBackend(transport=Recorded({}))
    assert backend.rentable(_row("vm_specs")) and not backend.rentable(_row("bare_metal_specs"))
