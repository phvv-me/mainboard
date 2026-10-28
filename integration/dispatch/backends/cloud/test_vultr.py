"""Vultr instances and bare metal servers against the API's recorded answers, without a key."""

import pytest

from mainboard.dispatch.backends.cloud.vultr import VultrApi
from mainboard.dispatch.backends.exceptions import CapacityGone

from ..builders import offer, refusal
from ..constants import SSH_KEY
from ..fakes import Recorded

_KEYS = ("GET", "/ssh-keys?per_page=500")
_KNOWN = {"ssh_keys": [{"id": "key-1", "ssh_key": SSH_KEY}]}


@pytest.mark.parametrize(
    ("plan", "path", "answer"),
    [
        pytest.param(
            "vcg-l40s-16c-180g-48vram",
            "/instances",
            {"instance": {"id": "i-1"}},
            id="a-cloud-gpu-plan-is-an-instance",
        ),
        pytest.param(
            "vbm-72c-480gb-gh200-gpu",
            "/bare-metals",
            {"bare_metal": {"id": "i-1"}},
            id="a-bare-metal-plan-is-a-bare-metal-server",
        ),
    ],
)
def test_a_plan_is_created_where_its_kind_is_sold_with_the_known_key(
    plan: str, path: str, answer: dict
) -> None:
    wire = Recorded({_KEYS: _KNOWN, ("POST", path): answer})
    handle = VultrApi(transport=wire).create(
        offer(instance_name=plan, location="ewr"), name="hold", public_key=SSH_KEY, disk_gb=0
    )
    body = wire.sent[-1][2]
    assert handle == "i-1" and wire.sent[-1][1].endswith(path)
    assert (body["plan"], body["region"], body["sshkey_id"]) == (plan, "ewr", ["key-1"])


def test_vultr_registers_a_key_it_does_not_know() -> None:
    wire = Recorded({_KEYS: {"ssh_keys": []}, ("POST", "/ssh-keys"): {"ssh_key": {"id": "new"}}})
    registered = VultrApi(transport=wire).registered(SSH_KEY)
    assert registered == "new" and wire.sent[-1][2]["ssh_key"] == SSH_KEY


def test_vultr_looks_a_handle_up_as_an_instance_then_as_a_bare_metal_server() -> None:
    metal = {"bare_metal": {"id": "b-1", "status": "active", "main_ip": "5.6.7.8", "plan": "vbm"}}
    wire = Recorded(
        {("GET", "/instances/b-1"): refusal(404, "{}"), ("GET", "/bare-metals/b-1"): metal}
    )
    found = VultrApi(transport=wire).machine("b-1")
    assert (found.status, found.host, found.user) == ("running", "5.6.7.8", "root")


@pytest.mark.parametrize(
    ("answer", "status", "host"),
    [
        pytest.param({"status": "pending", "main_ip": "0.0.0.0"}, "pending", "", id="unassigned"),
        pytest.param(
            {"status": "active", "server_status": "installingbooting", "main_ip": "1.2.3.4"},
            "pending",
            "1.2.3.4",
            id="still-installing",
        ),
        pytest.param(
            {"status": "active", "server_status": "ok", "main_ip": "1.2.3.4"},
            "running",
            "1.2.3.4",
            id="booted",
        ),
    ],
)
def test_an_instance_takes_ssh_once_active_and_ok_with_an_address(
    answer: dict, status: str, host: str
) -> None:
    wire = Recorded({("GET", "/instances/i-1"): {"instance": {"id": "i-1", **answer}}})
    found = VultrApi(transport=wire).machine("i-1")
    assert (found.status, found.host) == (status, host)


def test_a_handle_neither_kind_knows_is_gone_and_ending_it_tolerates_that() -> None:
    gone = refusal(404, "{}")
    wire = Recorded(
        {
            ("GET", "/instances/x"): gone,
            ("GET", "/bare-metals/x"): gone,
            ("DELETE", "/instances/x"): gone,
            ("DELETE", "/bare-metals/x"): gone,
        }
    )
    api = VultrApi(transport=wire)
    api.terminate("x")
    assert api.machine("x").status == "gone"


def test_vultr_plan_that_is_not_available_is_a_retry() -> None:
    wire = Recorded(
        {_KEYS: _KNOWN, ("POST", "/instances"): refusal(400, "this plan is not available here")}
    )
    with pytest.raises(CapacityGone):
        VultrApi(transport=wire).create(
            offer(instance_name="vcg-l40s-16c-180g-48vram"),
            name="n",
            public_key=SSH_KEY,
            disk_gb=0,
        )
    assert wire.sent[-1][0] == "POST"
