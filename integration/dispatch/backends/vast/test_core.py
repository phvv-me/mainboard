"""The Vast backend against its API's recorded answers, without a key or a bill."""

import json
from urllib.error import HTTPError

from mainboard.dispatch import arch
from mainboard.dispatch.backends import VastBackend
from mainboard.dispatch.backends.vast.market import Ask

from ..builders import refusal
from ..fakes import Recorded


def test_vast_asks_its_own_capability_field_for_an_architecture(monkeypatch) -> None:
    """The query also carries the data-center scope only when asked, and the download floor."""
    monkeypatch.setenv("VAST_API_KEY", "test-vast")
    wire = Recorded({("POST", "/bundles/"): {"offers": []}})
    market = VastBackend(transport=wire).market
    market.search(Ask(gpus=1, arch=arch("sm_120")))
    market.search(Ask(gpus=1, datacenter=True))
    query, scoped = wire.sent[0][2], wire.sent[1][2]
    assert query["compute_cap"] == {"gte": 1200, "lte": 1200} and "gpu_name" not in query
    assert "datacenter" not in query and scoped["datacenter"] == {"eq": True}
    assert query["inet_down"] == scoped["inet_down"] == {"gte": 500.0}


def test_a_rate_limited_vast_create_waits_and_asks_again(monkeypatch) -> None:
    monkeypatch.setenv("VAST_API_KEY", "test-vast")
    answers = iter([refusal(429, "{}"), refusal(429, "{}"), {"new_contract": 7}])

    def wire(request):
        answer = next(answers)
        if isinstance(answer, HTTPError):
            raise answer
        return json.dumps(answer).encode()

    slept: list[float] = []
    created = VastBackend(transport=wire, sleeper=slept.append).api.created({"id": 1}, {})
    assert created == {"new_contract": 7} and slept == [2.0, 5.0]
