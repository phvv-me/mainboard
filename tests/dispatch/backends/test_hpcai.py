import json

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import HpcAiBackend, api_key, http_transport
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


_VARS = {"instance-type-id": "t1", "image-id": "i1", "region": "r1"}


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


def test_submit_posts_an_authed_create_request_and_returns_a_handle() -> None:
    backend = authed_backend({})
    backend.spot = True
    handle = backend.submit(hpc_ai_plan(_VARS), "python train.py", Resources(max_usd=5.0))
    assert len(handle) == 32
    (create_request,) = backend.transport.calls
    assert create_request.full_url == "https://www.hpc-ai.com/api/instance/create"
    assert create_request.get_header("X-api-key") == "key-123"
    body = json.loads(create_request.data)
    assert (body["name"], body["isSpotInstance"]) == (handle, True)
    assert (body["instanceTypeId"], body["imageId"], body["region"]) == ("t1", "i1", "r1")


def test_submit_wraps_the_command_in_an_init_script_with_exit_and_log_sentinels() -> None:
    backend = authed_backend({})
    handle = backend.submit(hpc_ai_plan(_VARS), "python train.py", Resources(max_usd=5.0))
    (create_request,) = backend.transport.calls
    body = json.loads(create_request.data)
    init_script = body["instanceConfiguration"]["initScript"]
    assert "python train.py" in init_script
    assert f"/mnt/vol/{handle}.exit" in init_script
    assert f"/mnt/vol/{handle}.log" in init_script


# --- state ---


def test_state_is_vanished_when_no_instance_matches_the_handle() -> None:
    backend = authed_backend({"instances": []})
    assert backend.state("missing").verdict == "vanished"


@pytest.mark.parametrize(
    ("status", "verdict"),
    [("running", "running"), ("stopped", "ok"), ("failed", "failed"), ("mystery", "unknown")],
)
def test_state_maps_instance_runtime_status_onto_a_verdict(status: str, *, verdict: str) -> None:
    backend = authed_backend(
        {"instances": [{"name": "h1", "instanceRuntimeInfo": {"status": status}}]}
    )
    state = backend.state("h1")
    assert state.state == status
    assert state.verdict == verdict


# --- logs / cancel / deliver ---


def test_logs_raises_naming_the_volume_tail_path() -> None:
    backend = hpc_ai_backend(transport=FakeTransport())
    with pytest.raises(MissionError, match=r"/mnt/vol/h1\.log"):
        backend.logs("h1")


def test_cancel_posts_stop_then_delete() -> None:
    backend = authed_backend({}, {})
    backend.cancel("h1")
    urls = [request.full_url for request in backend.transport.calls]
    assert urls == [
        "https://www.hpc-ai.com/api/instance/stop",
        "https://www.hpc-ai.com/api/instance/delete",
    ]


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


def test_deliver_raises_naming_the_volume_path() -> None:
    backend = hpc_ai_backend(transport=FakeTransport())
    with pytest.raises(MissionError, match=r"/mnt/vol/h1"):
        backend.deliver("h1", path="out/results.json")


def test_default_transport_is_the_shared_seam() -> None:
    backend = HpcAiBackend()
    assert backend.transport is http_transport
