import json

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import (
    Delivery,
    HpcAiBackend,
    LogSource,
    api_key,
    http_transport,
)
from mainboard.dispatch.schedulers import Resources
from mainboard.manifest import HostProfile

from .conftest import FakeTransport, hpc_ai_backend, plan


def hpc_ai_plan(vars: dict[str, str] | None = None):
    """An `ExecutionPlan` whose profile is `kind="hpc-ai"`, with the given `[vars]` table."""
    return plan(
        profile=HostProfile(
            kind="hpc-ai", root="/repo", sync={"include": ["src"]}, vars=vars or {}
        )
    )


def authed_backend(*responses: dict) -> HpcAiBackend:
    """An `HpcAiBackend` over a queued-response transport, key auth from the env fixture."""
    return hpc_ai_backend(transport=FakeTransport(*responses))


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


def bodies(backend: HpcAiBackend) -> list[dict]:
    """The JSON body of every request the backend's fake transport recorded."""
    return [json.loads(call.data) for call in backend.transport.calls]


_VARS = {"instance-type-id": "t1", "image-id": "i1", "region": "r1"}
_CREATED = {"instanceId": "notebook-42"}


@pytest.fixture(autouse=True)
def _key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "key-123")


# --- api_key ---


def test_api_key_reads_the_env_and_refuses_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "key-123")
    assert api_key() == "key-123"
    monkeypatch.delenv("HPCAI_API_KEY")
    with pytest.raises(MissionError, match="HPCAI_API_KEY"):
        api_key()


# --- submit ---


def test_submit_refuses_before_any_network_call_when_budget_is_unset() -> None:
    transport = FakeTransport()
    backend = hpc_ai_backend(transport=transport)
    with pytest.raises(MissionError, match="max-usd"):
        backend.submit(hpc_ai_plan(), "echo hi", Resources())
    assert transport.calls == []


@pytest.mark.parametrize("missing", ["instance-type-id", "image-id", "region"])
def test_submit_raises_naming_the_missing_var(missing: str) -> None:
    variables = dict(_VARS)
    del variables[missing]
    transport = FakeTransport()
    backend = hpc_ai_backend(transport=transport)
    with pytest.raises(MissionError, match=missing):
        backend.submit(hpc_ai_plan(variables), "echo hi", Resources(max_usd=1.0))
    assert transport.calls == []


def test_submit_posts_an_authed_create_request_and_returns_the_provider_instance_id() -> None:
    backend = authed_backend(_CREATED)
    backend.spot = True
    handle = backend.submit(hpc_ai_plan(_VARS), "python train.py", Resources(max_usd=5.0))
    assert handle == "notebook-42"
    (create_request,) = backend.transport.calls
    assert create_request.full_url == "https://www.hpc-ai.com/api/instance/create"
    assert create_request.get_header("X-api-key") == "key-123"
    body = json.loads(create_request.data)
    assert body["name"].startswith("mainboard-")
    assert body["isSpotInstance"] is True
    assert (body["instanceTypeId"], body["imageId"], body["region"]) == ("t1", "i1", "r1")


def test_submit_sends_every_field_their_create_validator_calls_required() -> None:
    """A body missing `billing` or `nodePorts` is rejected outright rather than defaulted."""
    backend = authed_backend(_CREATED)
    backend.submit(hpc_ai_plan(_VARS), "echo hi", Resources(max_usd=1.0))
    (body,) = bodies(backend)
    assert body["billing"] == {"chargeMode": "perHour", "duration": 1}
    assert body["remoteStorages"] == []
    assert body["nodePorts"] == []
    configuration = body["instanceConfiguration"]
    assert configuration["enableCommonData"] is False
    assert configuration["enableDocker"] is False


def test_submit_wraps_the_command_in_an_init_script_with_exit_and_log_sentinels() -> None:
    backend = authed_backend(_CREATED)
    backend.submit(hpc_ai_plan(_VARS), "python train.py", Resources(max_usd=5.0))
    (body,) = bodies(backend)
    init_script = body["instanceConfiguration"]["initScript"]
    assert "python train.py" in init_script
    assert "mkdir -p /root/dataDisk" in init_script
    assert "/root/dataDisk/mainboard.exit" in init_script
    assert "/root/dataDisk/mainboard.log" in init_script


# --- state ---


def test_state_is_vanished_when_no_instance_matches_the_handle() -> None:
    backend = authed_backend(listing())
    assert backend.state("missing").verdict == "vanished"


def test_state_pages_the_listing_until_the_handle_turns_up() -> None:
    backend = authed_backend(
        listing(listed("other", "Running"), total=51),
        listing(listed("h1", "Running"), total=51),
    )
    assert backend.state("h1").verdict == "running"
    assert [body["pager"]["currentPage"] for body in bodies(backend)] == [1, 2]


def test_state_stops_paging_once_it_has_read_past_the_total() -> None:
    backend = authed_backend(listing(listed("other", "Running")))
    assert backend.state("h1").verdict == "vanished"
    assert len(backend.transport.calls) == 1


@pytest.mark.parametrize(
    ("status", "verdict"),
    [
        ("Initializing", "running"),
        ("PullingImage", "running"),
        ("Starting", "running"),
        ("Running", "running"),
        ("Stopped", "ok"),
        ("Archived", "ok"),
        ("Released", "vanished"),
        ("StartingFailed", "failed"),
        ("InitializationFailed", "failed"),
        ("Mystery", "unknown"),
    ],
)
def test_state_maps_their_camel_case_runtime_status_onto_a_verdict(
    status: str, *, verdict: str
) -> None:
    backend = authed_backend(listing(listed("h1", status)))
    state = backend.state("h1")
    assert state.state == status
    assert state.verdict == verdict


def test_state_asks_the_listing_with_the_pager_it_refuses_a_request_without() -> None:
    backend = authed_backend(listing(listed("h1", "Running")))
    backend.state("h1")
    (request,) = backend.transport.calls
    assert request.full_url == "https://www.hpc-ai.com/api/instance/list"
    assert json.loads(request.data) == {"pager": {"currentPage": 1, "pageSize": 50}}


# --- catalog ---


def test_catalog_flattens_the_console_feed_to_one_row_per_type_per_region() -> None:
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
                }
            ]
        }
    )
    rows = backend.catalog()
    assert [row["instance_type_id"] for row in rows] == ["t-us", "t-eu"]
    assert rows[0] == {
        "gpu": "RTX-4090",
        "gpus": 8,
        "usd_hr": pytest.approx(5.0),
        "region": "us-west-1",
        "region_id": "r-us",
        "instance_type_id": "t-us",
        "in_stock": True,
    }
    (request,) = backend.transport.calls
    assert request.full_url == "https://www.hpc-ai.com/api/resource/user/instance/list"


def test_catalog_of_a_type_quoting_no_hourly_rate_reads_as_unpriced_rather_than_crashing() -> None:
    backend = authed_backend(
        {
            "instanceInfos": [
                {
                    "regionInfos": [
                        {
                            "instanceTypeInfos": [
                                {"instanceTypeId": "t1", "price": [{"chargeMode": "perMonth"}]}
                            ]
                        }
                    ]
                }
            ]
        }
    )
    (row,) = backend.catalog()
    assert (row["usd_hr"], row["gpu"], row["gpus"], row["in_stock"]) == (0.0, "", 0, False)


def test_catalog_of_an_empty_feed_is_empty() -> None:
    assert authed_backend({}).catalog() == []


# --- logs / cancel / deliver ---


def test_the_declared_log_gap_names_the_sentinel_path_on_the_instance() -> None:
    advice = hpc_ai_backend(transport=FakeTransport()).refusal(LogSource, handle="h1")
    assert advice == (
        "hpc-ai backend has no server-side logs; read /root/dataDisk/mainboard.log on instance "
        "h1 over ssh instead"
    )


def test_cancel_stops_then_terminates_the_instance_by_id() -> None:
    backend = authed_backend({}, {})
    backend.cancel("notebook-42")
    urls = [request.full_url for request in backend.transport.calls]
    assert urls == [
        "https://www.hpc-ai.com/api/instance/stop",
        "https://www.hpc-ai.com/api/instance/terminate",
    ]
    assert bodies(backend) == [{"instanceId": "notebook-42"}] * 2


def test_standing_reads_the_account_balance_under_the_same_console_key() -> None:
    backend = authed_backend({"balance": 100, "availableVoucherAmount": 5})
    standing = backend.standing()
    assert standing.keyed is True
    assert standing.credit_usd == pytest.approx(100.0)
    assert standing.usd_hr is None
    (request,) = backend.transport.calls
    assert request.full_url == "https://www.hpc-ai.com/api/balance"
    assert request.headers["X-api-key"] == "key-123"


def test_standing_without_a_key_names_the_variable_and_never_calls_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HPCAI_API_KEY")
    transport = FakeTransport()
    standing = hpc_ai_backend(transport=transport).standing()
    assert standing.keyed is False
    assert "HPCAI_API_KEY" in standing.note
    assert transport.calls == []


def test_the_declared_delivery_gap_names_the_sentinel_path_and_the_path_asked_for() -> None:
    backend = hpc_ai_backend(transport=FakeTransport())
    advice = backend.refusal(Delivery, handle="h1", path="out/results.json")
    assert advice == (
        "hpc-ai backend cannot deliver 'out/results.json' yet; download "
        "/root/dataDisk/mainboard.* from instance h1 over ssh until that path lands"
    )


def test_default_transport_is_the_shared_seam() -> None:
    backend = HpcAiBackend()
    assert backend.transport is http_transport
