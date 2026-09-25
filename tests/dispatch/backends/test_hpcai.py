from contextlib import nullcontext
from pathlib import Path
from urllib.error import HTTPError

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import Capability, Delivery, HpcAiBackend, LogSource, api_key
from mainboard.dispatch.evidence import framing, staging
from mainboard.dispatch.rentals import waiting
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import HostProfile

from ..support import Naps, created_request
from .support import FakeTransport, Reply, hpc_ai_backend, plan, refused

_STOP = "https://www.hpc-ai.com/api/instance/stop"
_TERMINATE = "https://www.hpc-ai.com/api/instance/terminate"
_LIST = "https://www.hpc-ai.com/api/instance/list"

# The `[hosts.<name>.vars]` table their create endpoint needs, and the id it answers with.
_VARS = {"instance-type-id": "t1", "image-id": "i1", "region": "r1"}
_CREATED = {"instanceId": "notebook-42"}

# Their published `instanceRuntimeInfo.status` set as they spell it, onto our verdicts. The last
# two are not theirs: a state added later and a row carrying none read as unknown.
_VERDICTS = {
    "Initializing": "running",
    "PullingImage": "running",
    "Starting": "running",
    "Restarting": "running",
    "Running": "running",
    "Stopping": "running",
    "Stopped": "ok",
    "Archived": "ok",
    "Released": "vanished",
    "StartingFailed": "failed",
    "InitializationFailed": "failed",
    "Mystery": "unknown",
    "": "unknown",
}


def hpc_ai_plan(variables: dict[str, str] | None = None):
    """An `ExecutionPlan` whose profile is `kind="hpc-ai"`, with the given `[vars]` table."""
    return plan(
        profile=HostProfile(
            kind="hpc-ai", root="/repo", sync={"include": ["src"]}, vars=variables or {}
        )
    )


def authed_backend(*responses: Reply, spot: bool = False) -> HpcAiBackend:
    """An `HpcAiBackend` replaying `responses`, keyed by the env fixture."""
    return hpc_ai_backend(transport=FakeTransport(*responses), spot=spot)


def listing(*instances: dict, total: int | None = None) -> dict:
    """One `/instance/list` page carrying `instances`, its pager reporting `total`."""
    return {
        "instances": list(instances),
        "pager": {
            "currentPage": 1,
            "pageSize": 50,
            "totalEntries": len(instances) if total is None else total,
        },
    }


def listed(handle: str, status: str) -> dict:
    """One instance row as `/instance/list` nests it."""
    return {
        "instanceMetadata": {"instanceId": handle},
        "instanceRuntimeInfo": {"status": status},
    }


def priced(identifier: str, *, usd_hr: float, stock: str) -> dict:
    """One `instanceTypeInfos` entry priced per hour and per week, only the hourly one counting."""
    return {
        "instanceTypeId": identifier,
        "gpuNum": 8,
        "price": [
            {"chargeMode": "perWeek", "price": usd_hr * 168},
            {"chargeMode": "perHour", "price": usd_hr},
        ],
        "stockStatus": stock,
    }


@pytest.fixture(autouse=True)
def _key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "key-123")


def test_api_key_reads_the_env_and_refuses_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert api_key() == "key-123"
    monkeypatch.delenv("HPCAI_API_KEY")
    with pytest.raises(MissionError, match="HPCAI_API_KEY"):
        api_key()


@pytest.mark.parametrize(
    "missing",
    [None, "instance-type-id", "image-id", "region"],
    ids=["a-budget-nobody-set", "instance-type-id", "image-id", "region"],
)
def test_submit_refuses_before_any_network_call_when_the_request_is_incomplete(
    missing: str | None,
) -> None:
    """`HostProfile` forbids undeclared fields, so the opaque provider ids live in `vars`."""
    variables = {key: value for key, value in _VARS.items() if key != missing}
    backend = authed_backend()
    with pytest.raises(MissionError, match=missing or "max-usd"):
        backend.submit(
            hpc_ai_plan(variables),
            "echo hi",
            Resources(max_usd=1.0) if missing else Resources(),
            allocation=created_request(),
        )
    assert backend.transport.calls == []


@pytest.mark.parametrize("spot", [False, True])
def test_submit_posts_every_field_their_create_validator_calls_required(spot: bool) -> None:
    """The handle is the provider's `instanceId`, and the initScript writes the sentinel pair."""
    backend = authed_backend(_CREATED, spot=spot)
    handle = backend.submit(
        hpc_ai_plan(_VARS), "python train.py", Resources(max_usd=5.0), allocation=created_request()
    )
    assert handle == "notebook-42"
    (request,) = backend.transport.calls
    assert request.full_url == "https://www.hpc-ai.com/api/instance/create"
    assert request.get_header("X-api-key") == "key-123"
    (body,) = backend.transport.bodies
    assert body.pop("name").startswith("mainboard-")
    assert body == {
        "isSpotInstance": spot,
        "instanceTypeId": "t1",
        "imageId": "i1",
        "region": "r1",
        "billing": {"chargeMode": "perHour", "duration": 1},
        "remoteStorages": [],
        "instanceConfiguration": {
            "enableCommonData": False,
            "enableDocker": False,
            "initScript": (
                "mkdir -p /root/dataDisk\n"
                f"{{ {staging()}\npython train.py\nstatus=$?\n{framing()}\n"
                "\n} > /root/dataDisk/mainboard.log 2>&1\n"
                "echo $status > /root/dataDisk/mainboard.exit\n"
            ),
        },
        "nodePorts": [],
    }


@pytest.mark.parametrize(
    ("pages", "verdict", "asked"),
    [
        pytest.param((listing(),), "vanished", 1, id="a-listing-with-nothing-in-it"),
        pytest.param(
            (listing(listed("other", "Running")),),
            "vanished",
            1,
            id="a-page-that-already-runs-past-the-total",
        ),
        pytest.param(
            (
                listing(listed("other", "Running"), total=51),
                listing(listed("h1", "Running"), total=51),
            ),
            "running",
            2,
            id="a-handle-that-turns-up-on-the-second-page",
        ),
    ],
)
def test_state_walks_the_pager_until_the_handle_turns_up_or_the_total_runs_out(
    pages: tuple[dict, ...], verdict: str, asked: int
) -> None:
    """`/instance/list` answers 500 for a request carrying no pager, so the page size is ours."""
    backend = authed_backend(*pages)
    assert backend.state("h1").verdict == verdict
    assert backend.transport.urls == [_LIST] * asked
    assert backend.transport.bodies[0] == {"pager": {"currentPage": 1, "pageSize": 50}}
    assert [body["pager"]["currentPage"] for body in backend.transport.bodies] == [
        page for page in range(1, asked + 1)
    ]


def test_state_maps_every_camel_case_runtime_status_onto_one_of_our_verdicts() -> None:
    """Their states are camel case and ours are one word, and one they add later is unknown."""
    states = {
        status: authed_backend(listing(listed("h1", status))).state("h1") for status in _VERDICTS
    }
    assert {status: state.verdict for status, state in states.items()} == _VERDICTS
    assert {status: state.state for status, state in states.items()} == {
        status: status for status in _VERDICTS
    }


def test_catalog_flattens_the_console_feed_to_one_row_per_type_per_region() -> None:
    """Rows land in stock first and cheapest first, a type quoting no hourly rate as unpriced."""
    backend = authed_backend(
        {
            "instanceInfos": [
                {
                    "gpuName": "RTX-4090",
                    "regionInfos": [
                        {
                            "regionName": "eu-west-1",
                            "regionId": "r-eu",
                            "instanceTypeInfos": [priced("t-eu", usd_hr=4.0, stock="OutOfStock")],
                        },
                        {
                            "regionName": "us-west-1",
                            "regionId": "r-us",
                            "instanceTypeInfos": [priced("t-us", usd_hr=5.0, stock="InStock")],
                        },
                    ],
                },
                {
                    "regionInfos": [
                        {
                            "instanceTypeInfos": [
                                {"instanceTypeId": "t-mute", "price": [{"chargeMode": "perMonth"}]}
                            ]
                        }
                    ]
                },
            ]
        }
    )
    rows = backend.catalog()
    assert [row["instance_type_id"] for row in rows] == ["t-us", "t-eu", "t-mute"]
    assert rows[0] == {
        "gpu": "RTX-4090",
        "gpus": 8,
        "usd_hr": pytest.approx(5.0),
        "region": "us-west-1",
        "region_id": "r-us",
        "instance_type_id": "t-us",
        "in_stock": True,
    }
    assert rows[2] == {
        "gpu": "",
        "gpus": 0,
        "usd_hr": None,
        "region": "",
        "region_id": "",
        "instance_type_id": "t-mute",
        "in_stock": False,
    }
    assert backend.transport.urls == ["https://www.hpc-ai.com/api/resource/user/instance/list"]
    assert authed_backend({}).catalog() == []


@pytest.mark.parametrize(
    ("price", "hourly"),
    [
        pytest.param({"chargeMode": "perHour"}, None, id="no-price-at-all"),
        pytest.param({"chargeMode": "perHour", "price": None}, None, id="a-null-price"),
        pytest.param({"chargeMode": "perHour", "price": 0}, 0.0, id="a-real-zero"),
    ],
)
def test_an_unpriced_hourly_offer_is_unknown_not_free(price: dict, hourly: float | None) -> None:
    assert HpcAiBackend._hourly({"price": [price]}) == hourly


@pytest.mark.parametrize(
    ("stop", "terminate", "refuses"),
    [
        pytest.param({}, {}, False, id="a-running-instance"),
        pytest.param(refused(400, _STOP), {}, False, id="an-instance-that-was-already-stopped"),
        pytest.param(
            refused(400, _STOP),
            refused(404, _TERMINATE),
            False,
            id="an-instance-hpc-ai-has-already-forgotten",
        ),
        pytest.param({}, refused(401, _TERMINATE), True, id="a-terminate-really-refused"),
    ],
)
def test_cancel_stops_then_terminates_whatever_the_stop_answered(
    stop: Reply, terminate: Reply, refuses: bool
) -> None:
    """A refused stop never blocks the terminate, and a 404 terminate is the state asked for.

    Any other refusal of the terminate is a real fault, since the meter is still running.
    """
    backend = authed_backend(stop, terminate)
    with pytest.raises(HTTPError) if refuses else nullcontext():
        backend.cancel("notebook-42")
    assert backend.transport.urls == [_STOP, _TERMINATE]
    assert backend.transport.bodies == [{"instanceId": "notebook-42"}] * 2


def test_standing_reads_the_balance_under_the_console_key_and_names_it_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`balance` is the pot left after vouchers and credits; no rate without an instance type."""
    backend = authed_backend({"balance": 100, "availableVoucherAmount": 5})
    standing = backend.standing()
    assert (standing.keyed, standing.credit_usd, standing.usd_hr) == (True, 100.0, None)
    (request,) = backend.transport.calls
    assert request.full_url == "https://www.hpc-ai.com/api/balance"
    assert request.headers["X-api-key"] == "key-123"
    monkeypatch.delenv("HPCAI_API_KEY")
    unkeyed = authed_backend()
    absent = unkeyed.standing()
    assert absent.keyed is False
    assert "HPCAI_API_KEY" in absent.note
    assert unkeyed.transport.calls == []


@pytest.mark.parametrize(
    ("capability", "line"),
    [
        pytest.param(
            LogSource,
            "hpc-ai backend has no server-side logs; read /root/dataDisk/mainboard.log on "
            "instance h1 over ssh instead",
            id="the-log-gap",
        ),
        pytest.param(
            Delivery,
            "hpc-ai backend cannot deliver 'out/results.json' yet; download "
            "/root/dataDisk/mainboard.* from instance h1 over ssh until that path lands",
            id="the-delivery-gap",
        ),
    ],
)
def test_a_declared_gap_names_the_sentinel_path_to_read_by_hand_instead(
    capability: type[Capability], line: str
) -> None:
    """Both gaps point at the sentinel files on the instance's own disk."""
    advice = authed_backend().refusal(capability, handle="h1", path="out/results.json")
    assert advice == line


def reachable_row(handle: str, *, port: int = 30022) -> dict:
    """One instance row for a machine up and publishing the ssh line its console prints."""
    row = listed(handle, "Running")
    row["instanceMetadata"]["instanceUsername"] = "ubuntu"
    row["instanceSpecInfo"] = {
        "regionInfo": {"sshAddress": "gpu.hpc-ai.com"},
        "nodePorts": [{"port": 8888, "nodePort": 30888}, {"port": 22, "nodePort": port}],
    }
    return row


@pytest.mark.usefixtures("answering")
def test_a_rental_is_created_waiting_for_a_landing_and_read_back_off_its_ssh_line(
    home_key: Path,
) -> None:
    """The initScript waits for the landing inside the same sentinel redirect, and ssh reaches
    the instance at the address, mapped port and login their console's `ssh -p` line uses.
    """
    backend = authed_backend(_CREATED, listing(reachable_row("notebook-42")))
    rental = backend.rent(
        hpc_ai_plan(_VARS),
        Resources(max_usd=1.0, walltime="00:30:00"),
        allocation=created_request(),
    )
    created, _ = backend.transport.bodies
    script = created["instanceConfiguration"]["initScript"]
    assert waiting() in script and "python" not in script
    assert script.endswith(
        "} > /root/dataDisk/mainboard.log 2>&1\necho $status > /root/dataDisk/mainboard.exit\n"
    )
    assert rental.handle == "notebook-42"
    assert rental.endpoint.destination == "ubuntu@gpu.hpc-ai.com"
    assert (rental.endpoint.port, rental.endpoint.identity) == (30022, home_key.as_posix())


@pytest.mark.usefixtures("home_key")
def test_an_instance_that_publishes_no_ssh_endpoint_is_terminated_and_names_the_console_keys() -> (
    None
):
    """HPC-AI attaches no key at create time, so the missing one is almost always the account's."""
    naps = Naps()
    pages = [listing(listed("notebook-42", "PullingImage"))] * 90
    backend = hpc_ai_backend(transport=FakeTransport(_CREATED, *pages, {}, {}), naps=naps)
    with pytest.raises(MissionError, match="ssh keys in the HPC-AI console"):
        backend.rent(
            hpc_ai_plan(_VARS),
            Resources(max_usd=1.0, walltime="00:30:00"),
            allocation=created_request(),
        )
    assert backend.transport.urls[-1] == _TERMINATE
    assert naps.waited == [10.0] * 90


def test_rentals_walk_every_listing_page_whoever_created_the_instance() -> None:
    """Each listed instance is a rental, its label read from the name it was created under."""
    named = {**listed("n1", "Running"), "instanceMetadata": {"instanceId": "n1", "name": "hold"}}
    backend = authed_backend(listing(named, total=51), listing({}, total=51))
    first, second = backend.rentals()
    assert (first.handle, first.label, first.status) == ("n1", "hold", "Running")
    assert (second.handle, second.label, second.status) == ("", "", "")
