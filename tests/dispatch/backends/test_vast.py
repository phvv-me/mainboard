import json
from time import sleep

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import VastBackend
from mainboard.dispatch.backends.base import http_transport
from mainboard.dispatch.backends.vast import api_key, exit_sentinel
from mainboard.dispatch.schedulers import Resources
from mainboard.manifest import Container, HostProfile

from .conftest import not_found, plan, vast_backend


def vast_plan(**overrides: Container | HostProfile):
    """An `ExecutionPlan` whose profile is `kind="vast"`, containerized only when asked."""
    fields: dict[str, Container | HostProfile] = {
        "profile": HostProfile(kind="vast", root="/repo", sync={"include": ["src"]})
    }
    fields.update(overrides)
    return plan(**fields)


def offer(identifier: int, *, dph: float, bid: float = 0.1, **extra: float | int | str):
    """One `/bundles` offer row, only the fields the backend and the catalog probe read."""
    row = {"id": identifier, "dph_total": dph, "min_bid": bid, "gpu_name": "RTX 4090"}
    row.update({"num_gpus": 1, "geolocation": "Texas, US", "rentable": True})
    row.update(extra)
    return row


def terminal_backend(status: str, *, log: str) -> VastBackend:
    """A backend whose instance is terminal and whose log tail is `log`, in reply order."""
    return vast_backend(
        {"instances": {"id": 7, "actual_status": status}},
        {"result_url": "https://s3.example/logs/7.log"},
        log,
    )


def bodies(backend: VastBackend) -> list[dict]:
    """The JSON body of every request the backend's fake transport recorded."""
    return [json.loads(call.data) for call in backend.transport.calls]


_OFFERS = {"offers": [offer(11, dph=0.5), offer(22, dph=0.2, bid=0.05)]}
_CREATED = {"success": True, "new_contract": 4242}


@pytest.fixture(autouse=True)
def _key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")


# --- api_key ---


def test_api_key_reads_the_env_and_refuses_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    assert api_key() == "key-123"
    monkeypatch.delenv("VAST_API_KEY")
    with pytest.raises(MissionError, match="VAST_API_KEY"):
        api_key()


# --- search ---


def test_search_posts_the_console_default_filters_and_an_authed_bearer_header() -> None:
    backend = vast_backend(_OFFERS)
    assert len(backend.search()) == 2
    (request,) = backend.transport.calls
    assert request.full_url == "https://console.vast.ai/api/v0/bundles/"
    assert request.get_header("Authorization") == "Bearer key-123"
    body = json.loads(request.data)
    assert body["verified"] == {"eq": True}
    assert body["rentable"] == {"eq": True}
    assert body["rented"] == {"eq": False}
    assert body["external"] == {"eq": False}
    assert body["type"] == "on-demand"
    assert body["order"] == [["dph_total", "asc"]]
    assert "gpu_name" not in body and "num_gpus" not in body and "dph_total" not in body


def test_search_narrows_by_gpu_count_and_price_reading_underscores_as_spaces() -> None:
    backend = vast_backend(_OFFERS)
    backend.search(gpu_name="RTX_4090", gpus=2, max_usd_hr=0.4, limit=5)
    (body,) = bodies(backend)
    assert body["gpu_name"] == {"eq": "RTX 4090"}
    assert body["num_gpus"] == {"eq": 2}
    assert body["dph_total"] == {"lte": 0.4}
    assert body["limit"] == 5


def test_search_asks_for_bid_pricing_and_re_ranks_by_the_bid_floor_when_renting_spot() -> None:
    backend = vast_backend(_OFFERS, spot=True)
    assert [row["id"] for row in backend.search()] == [22, 11]
    assert bodies(backend)[0]["type"] == "bid"


def test_search_reads_an_empty_market_as_no_offers() -> None:
    assert vast_backend({}).search() == []


# --- cheapest ---


def test_cheapest_picks_the_lowest_on_demand_rate() -> None:
    assert vast_backend(_OFFERS).cheapest(gpu_name="RTX 4090", gpus=1)["id"] == 22


def test_cheapest_ranks_by_the_bid_floor_when_renting_spot() -> None:
    offers = {"offers": [offer(11, dph=0.5, bid=0.01), offer(22, dph=0.2, bid=0.09)]}
    assert vast_backend(offers, spot=True).cheapest(gpu_name="", gpus=1)["id"] == 11


def test_cheapest_refuses_when_the_market_is_empty_naming_the_ceiling() -> None:
    backend = vast_backend({"offers": []})
    with pytest.raises(MissionError, match=r"1x H100 offer under \$2.00/hr"):
        backend.cheapest(gpu_name="H100", gpus=1, max_usd_hr=2.0)


# --- hourly_cap ---


@pytest.mark.parametrize(
    ("walltime", "cap"),
    [("00:30:00", 8.0), ("01:00:00", 4.0), (None, 0.0), ("00:00:00", 0.0)],
)
def test_hourly_cap_turns_a_budget_and_a_walltime_into_a_rate_ceiling(
    walltime: str | None, *, cap: float
) -> None:
    resources = Resources(max_usd=4.0, walltime=walltime)
    assert VastBackend.hourly_cap(resources) == pytest.approx(cap)


# --- submit ---


def test_submit_refuses_before_any_network_call_when_budget_is_unset() -> None:
    backend = vast_backend()
    with pytest.raises(MissionError, match="max-usd"):
        backend.submit(vast_plan(), "echo hi", Resources())
    assert backend.transport.calls == []


def test_submit_rents_the_cheapest_offer_and_returns_the_new_contract_id() -> None:
    backend = vast_backend(_OFFERS, _CREATED)
    handle = backend.submit(vast_plan(), "python train.py", Resources(max_usd=5.0, gpus=2))
    assert handle == "4242"
    search, create = backend.transport.calls
    assert json.loads(search.data)["num_gpus"] == {"eq": 2}
    assert create.full_url == "https://console.vast.ai/api/v0/asks/22/"
    assert create.get_method() == "PUT"
    body = json.loads(create.data)
    assert (body["client_id"], body["runtype"], body["onstart"]) == ("me", "args", "bash")
    assert body["image"] == "vastai/base-image:cuda-12.9.2-auto"
    assert body["label"] == "mainboard-provider-host"
    assert body["cancel_unavail"] is True
    assert body["disk"] == pytest.approx(16.0)
    assert "price" not in body


def test_submit_wraps_the_command_with_an_exit_sentinel_as_the_container_entrypoint() -> None:
    backend = vast_backend(_OFFERS, _CREATED)
    backend.submit(vast_plan(), "python train.py", Resources(max_usd=5.0))
    flag, script = bodies(backend)[1]["args"]
    assert flag == "-c"
    assert script.startswith("python train.py\n")
    assert "echo mainboard-exit:$status" in script
    assert script.endswith("exit $status\n")


def test_submit_rents_a_single_gpu_when_the_request_names_no_count() -> None:
    backend = vast_backend(_OFFERS, _CREATED)
    backend.submit(vast_plan(), "echo hi", Resources(max_usd=1.0))
    assert bodies(backend)[0]["num_gpus"] == {"eq": 1}


def test_submit_bids_the_offer_floor_when_renting_spot() -> None:
    backend = vast_backend(_OFFERS, _CREATED, spot=True)
    backend.submit(vast_plan(), "echo hi", Resources(max_usd=1.0))
    assert bodies(backend)[1]["price"] == pytest.approx(0.05)


def test_submit_caps_the_hourly_rate_when_the_job_declares_a_walltime() -> None:
    backend = vast_backend(_OFFERS, _CREATED)
    backend.submit(vast_plan(), "echo hi", Resources(max_usd=1.0, walltime="02:00:00"))
    assert bodies(backend)[0]["dph_total"] == {"lte": pytest.approx(0.5)}


def test_submit_rents_under_the_plan_container_image_when_the_plan_is_containerized() -> None:
    backend = vast_backend(_OFFERS, _CREATED)
    backend.submit(
        vast_plan(container=Container(image="pytorch/pytorch:latest")),
        "echo hi",
        Resources(max_usd=1.0),
    )
    assert bodies(backend)[1]["image"] == "pytorch/pytorch:latest"


# --- state ---


@pytest.mark.parametrize("status", ["created", "loading", "running", "stopping", "mystery"])
def test_state_of_a_container_that_has_not_finished_costs_no_log_fetch(status: str) -> None:
    """A live (or unrecognized) status is answered from the instance row alone."""
    backend = vast_backend({"instances": {"id": 7, "actual_status": status}})
    state = backend.state("7")
    assert state.state == status
    assert state.verdict == ("unknown" if status == "mystery" else "running")
    assert state.exit_code is None
    (request,) = backend.transport.calls
    assert request.full_url == "https://console.vast.ai/api/v0/instances/7/?owner=me"


@pytest.mark.parametrize(
    ("status", "code", "verdict"),
    [("exited", 0, "ok"), ("stopped", 3, "failed"), ("offline", 137, "failed")],
)
def test_state_of_a_terminal_container_reports_the_process_exit_code(
    status: str, *, code: int, verdict: str
) -> None:
    backend = terminal_backend(status, log=f"training done\nmainboard-exit:{code}\n")
    state = backend.state("7")
    assert (state.state, state.exit_code, state.verdict) == (status, code, verdict)
    instance, ask, fetch = backend.transport.calls
    assert instance.full_url == "https://console.vast.ai/api/v0/instances/7/?owner=me"
    assert ask.full_url == "https://console.vast.ai/api/v0/instances/request_logs/7/"
    assert fetch.full_url == "https://s3.example/logs/7.log"


def test_state_stays_unknown_when_the_log_carries_no_sentinel() -> None:
    """A container killed before the wrapper spoke never says how the command ended."""
    state = terminal_backend("exited", log="killed mid-epoch\n").state("7")
    assert (state.exit_code, state.verdict) == (None, "unknown")


def test_state_stays_unknown_when_vast_refuses_the_log() -> None:
    backend = vast_backend(
        {"instances": {"id": 7, "actual_status": "exited"}},
        {"success": False, "msg": "instance not running"},
    )
    state = backend.state("7")
    assert (state.exit_code, state.verdict) == (None, "unknown")


def test_state_stays_unknown_when_the_log_upload_is_already_gone() -> None:
    backend = vast_backend({"instances": {"id": 7, "actual_status": "exited"}}, not_found())
    assert backend.state("7").verdict == "unknown"


def test_state_is_vanished_when_the_row_is_null() -> None:
    assert vast_backend({"instances": None}).state("7").verdict == "vanished"


def test_state_is_vanished_when_the_instance_is_already_gone() -> None:
    assert vast_backend(not_found()).state("7").verdict == "vanished"


def test_state_re_raises_a_refusal_that_is_not_a_missing_instance() -> None:
    backend = vast_backend(not_found())
    backend.transport.responses = [
        type(not_found())("https://console.vast.ai/api/v0/instances/7/", 401, "no", {}, None)
    ]
    with pytest.raises(OSError, match="401"):
        backend.state("7")


# --- exit_sentinel ---


def test_exit_sentinel_reads_the_status_the_wrapper_echoed() -> None:
    assert exit_sentinel("epoch 3\nmainboard-exit:0\n") == 0


def test_exit_sentinel_takes_the_last_marker_a_restarted_container_left() -> None:
    assert exit_sentinel("mainboard-exit:0\nrestarted\nmainboard-exit:2\n") == 2


def test_exit_sentinel_skips_a_marker_carrying_no_number() -> None:
    assert exit_sentinel("mainboard-exit:1\nmainboard-exit:truncated") == 1


def test_exit_sentinel_of_a_log_without_any_marker_is_none() -> None:
    assert exit_sentinel("nothing to see here\n") is None


# --- logs ---


def test_logs_requests_an_upload_then_fetches_it_without_the_api_key() -> None:
    backend = vast_backend({"result_url": "https://s3.example/logs/7.log"}, "hello from vast\n")
    assert backend.logs("7") == "hello from vast\n"
    ask, fetch = backend.transport.calls
    assert ask.full_url == "https://console.vast.ai/api/v0/instances/request_logs/7/"
    assert ask.get_method() == "PUT"
    assert json.loads(ask.data)["tail"] == "2000"
    assert fetch.full_url == "https://s3.example/logs/7.log"
    assert fetch.get_header("Authorization") is None


def test_logs_waits_out_an_upload_still_in_flight() -> None:
    backend = vast_backend(
        {"result_url": "https://s3.example/logs/7.log"}, not_found(), "landed at last"
    )
    assert backend.logs("7") == "landed at last"


def test_logs_hands_the_url_over_when_the_upload_never_lands() -> None:
    backend = vast_backend({"result_url": "https://s3.example/logs/7.log"}, *[not_found()] * 20)
    with pytest.raises(MissionError, match=r"https://s3\.example/logs/7\.log"):
        backend.logs("7")


def test_logs_raises_when_vast_refuses_to_upload_at_all() -> None:
    backend = vast_backend({"success": False, "msg": "instance not running"})
    with pytest.raises(MissionError, match="instance not running"):
        backend.logs("7")


def test_logs_refuses_a_log_url_that_is_not_https() -> None:
    backend = vast_backend({"result_url": "http://s3.example/logs/7.log"})
    with pytest.raises(MissionError, match="non-https"):
        backend.logs("7")


# --- catalog ---


def test_catalog_turns_a_live_search_into_priced_offer_rows() -> None:
    rows = vast_backend(_OFFERS).catalog(gpu_name="RTX 4090")
    assert [row.rate_usd_hr for row in rows] == [pytest.approx(0.2), pytest.approx(0.5)]
    assert {row.provider for row in rows} == {"vast"}
    assert rows[0].gpu == "RTX 4090"
    assert rows[0].region == "Texas, US"
    assert rows[0].available is True
    assert rows[0].source == "probed:vast"
    assert not rows[0].spot


def test_catalog_prices_the_bid_floor_when_the_backend_rents_spot() -> None:
    rows = vast_backend(_OFFERS, spot=True).catalog()
    assert rows[0].spot is True
    assert [row.rate_usd_hr for row in rows] == [pytest.approx(0.05), pytest.approx(0.1)]


def test_catalog_of_an_offer_without_geolocation_leaves_the_region_empty() -> None:
    offers = {"offers": [offer(11, dph=0.5, geolocation="")]}
    assert not vast_backend(offers).catalog()[0].region


# --- cancel / deliver ---


def test_cancel_deletes_the_instance() -> None:
    backend = vast_backend({"success": True})
    backend.cancel("4242")
    (request,) = backend.transport.calls
    assert request.full_url == "https://console.vast.ai/api/v0/instances/4242/"
    assert request.get_method() == "DELETE"


def test_deliver_raises_naming_the_logs_verb_instead() -> None:
    with pytest.raises(MissionError, match=r"logs 4242"):
        vast_backend().deliver("4242", path="out/results.json")


def test_defaults_are_the_shared_transport_and_a_real_sleep() -> None:
    backend = VastBackend()
    assert backend.transport is http_transport
    assert backend.sleeper is sleep
