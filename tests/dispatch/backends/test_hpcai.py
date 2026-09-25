from contextlib import nullcontext
from pathlib import Path
from urllib.error import HTTPError

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import Capability, Delivery, HpcAiBackend, LogSource, api_key
from mainboard.dispatch.backends import hpcai as hpcai_module
from mainboard.dispatch.evidence import framing, staging
from mainboard.dispatch.rentals import waiting
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import HostProfile

from ..support import Naps, created_request, keypair
from .support import FakeTransport, Reply, hpc_ai_backend, plan, refused

# The API root every refusal a test queues is attributed to.
_API = "https://www.hpc-ai.com/api/instance/stop"
_LIST = "https://www.hpc-ai.com/api/instance/list"

# The `[hosts.<name>.vars]` table their create endpoint needs, and the id it answers with.
_VARS = {"instance-type-id": "t1", "image-id": "i1", "region": "r1"}
_CREATED = {"instanceId": "notebook-42"}

# Their published `instanceRuntimeInfo.status` set spelled the way they spell it, mapped onto our
# one-word verdicts. The last two are not theirs: a state they add later, and a row that carries
# no status at all, both of which must read as unknown rather than crash a sweep.
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
    """An `HpcAiBackend` over a queued-response transport, key auth from the env fixture."""
    return hpc_ai_backend(transport=FakeTransport(*responses), spot=spot)


def listing(*instances: dict, total: int | None = None) -> dict:
    """One `/instance/list` page carrying `instances`, its pager reporting `total` entries."""
    return {
        "instances": list(instances),
        "pager": {
            "currentPage": 1,
            "pageSize": 50,
            "totalEntries": len(instances) if total is None else total,
        },
    }


def listed(handle: str, status: str) -> dict:
    """One instance row as `/instance/list` nests it, id under metadata, status under runtime."""
    return {
        "instanceMetadata": {"instanceId": handle},
        "instanceRuntimeInfo": {"status": status},
    }


def priced(identifier: str, *, usd_hr: float, stock: str) -> dict:
    """One `instanceTypeInfos` entry, priced per hour and per week so only the hourly one counts.

    identifier: the instance-type id the row publishes.
    usd_hr: the hourly rate, from which the weekly one is derived.
    stock: the `stockStatus` string, `InStock` being the only rentable one.
    """
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
    """A body missing `billing` or `nodePorts` is rejected outright rather than defaulted.

    The returned handle is the provider's own `instanceId`, which is what every later call
    addresses the rental by, and the initScript is the only place the command's real exit code
    and captured output can land, since HPC-AI reports instance status and nothing finer.
    """
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
    """The catalog rows carry exactly what a hosts table needs.

    The launch-form feed is the only place HPC-AI publishes a price, a GPU name or a stock
    count, so the rows land in-stock first and cheapest first, and a type quoting no hourly
    rate reads as unpriced rather than crashing.
    """
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
    "price", [{"chargeMode": "perHour"}, {"chargeMode": "perHour", "price": None}]
)
def test_an_unpriced_hourly_offer_is_unknown_not_free(price: dict) -> None:
    assert HpcAiBackend._hourly({"price": [price]}) is None
    assert HpcAiBackend._hourly({"price": [{"chargeMode": "perHour", "price": 0}]}) == 0.0


def test_cancel_stops_then_terminates_the_instance_by_id() -> None:
    backend = authed_backend({}, {})
    backend.cancel("notebook-42")
    assert backend.transport.urls == [_API, "https://www.hpc-ai.com/api/instance/terminate"]
    assert backend.transport.bodies == [{"instanceId": "notebook-42"}] * 2


@pytest.mark.parametrize(
    ("stop", "terminate", "refuses"),
    [
        pytest.param(refused(400, _API), {}, False, id="an-instance-that-was-already-stopped"),
        pytest.param(
            refused(400, _API),
            refused(404, _API),
            False,
            id="an-instance-hpc-ai-has-already-forgotten",
        ),
        pytest.param({}, refused(401, _API), True, id="a-terminate-the-provider-really-refuses"),
    ],
)
def test_cancel_ends_the_billing_whatever_the_stop_before_it_answered(
    stop: Reply, terminate: Reply, refuses: bool
) -> None:
    """A sweep cancels every run it settles, so the same instance is cancelled more than once.

    Stopping is preparation rather than the point, so its refusal never blocks the terminate that
    ends the billing, and a terminate answered with a 404 has reached the state it asked for. Any
    other refusal of the terminate is a real fault, since the meter is still running.
    """
    backend = authed_backend(stop, terminate)
    with pytest.raises(HTTPError) if refuses else nullcontext():
        backend.cancel("notebook-42")
    assert backend.transport.urls[-1] == "https://www.hpc-ai.com/api/instance/terminate"


def test_standing_reads_the_balance_under_the_console_key_and_names_it_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standing reads the balance under the console's own key.

    `balance` is the pot left once vouchers and monthly credits have been spent first, and no
    rate rides along since the provider quotes no price without an instance type in hand.
    """
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
    """An instance's output never leaves its own disk, so both gaps point at the same files."""
    advice = authed_backend().refusal(capability, handle="h1", path="out/results.json")
    assert advice == line


def reachable_row(handle: str, *, port: int = 30022) -> dict:
    """One instance row for a machine that is up and publishes the ssh line its console prints."""
    row = listed(handle, "Running")
    row["instanceMetadata"]["instanceUsername"] = "ubuntu"
    row["instanceSpecInfo"] = {
        "regionInfo": {"sshAddress": "gpu.hpc-ai.com"},
        "nodePorts": [{"port": 8888, "nodePort": 30888}, {"port": 22, "nodePort": port}],
    }
    return row


def test_a_rental_is_created_waiting_for_a_landing_and_read_back_off_its_ssh_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare instance has no workspace, no tool and no environment, so the job cannot start yet.

    The initScript holds the machine still until the dispatch has put all three on it, and the
    sentinel pair on the data disk stays exactly where it was, since that is the only place an
    instance's own output can land. Where ssh reaches it is the address, the mapped port and the
    login their console builds its own `ssh -p` line from.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    key = keypair(tmp_path)
    monkeypatch.setattr(hpcai_module, "reachable", lambda endpoint, *, sleeper: endpoint)
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
    assert (rental.endpoint.port, rental.endpoint.identity) == (30022, key.as_posix())


def test_an_instance_that_publishes_no_ssh_endpoint_is_terminated_and_names_the_console_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HPC-AI attaches no key at create time, so the missing one is almost always the account's."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    keypair(tmp_path)
    naps = Naps()
    pages = [listing(listed("notebook-42", "PullingImage"))] * 90
    backend = hpc_ai_backend(transport=FakeTransport(_CREATED, *pages, {}, {}), naps=naps)
    with pytest.raises(MissionError, match="ssh keys in the HPC-AI console"):
        backend.rent(
            hpc_ai_plan(_VARS),
            Resources(max_usd=1.0, walltime="00:30:00"),
            allocation=created_request(),
        )
    assert backend.transport.urls[-1] == "https://www.hpc-ai.com/api/instance/terminate"
    assert naps.waited == [10.0] * 90


def test_rentals_walk_every_listing_page_whoever_created_the_instance() -> None:
    """Each listed instance is a rental, its label read from the name it was created under."""
    named = {**listed("n1", "Running"), "instanceMetadata": {"instanceId": "n1", "name": "hold"}}
    backend = authed_backend(listing(named, total=51), listing({}, total=51))
    first, second = backend.rentals()
    assert (first.handle, first.label, first.status) == ("n1", "hold", "Running")
    assert (second.handle, second.label, second.status) == ("", "", "")
