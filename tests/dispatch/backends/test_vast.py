from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch.backends import Delivery, VastBackend
from mainboard.dispatch.backends import vast as vast_module
from mainboard.dispatch.backends.base import ProviderBackend, image_cuda
from mainboard.dispatch.backends.vast import (
    api_key,
    capability,
    cuda_max_good,
    download,
    exit_sentinel,
)
from mainboard.dispatch.evidence import framing, staging
from mainboard.dispatch.rentals import LANDING_SECONDS, waiting
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import Container, HostProfile

from ...strategies import WORDS
from ..support import created_request, keypair
from .support import Naps, Reply, not_found, plan, refused, vast_backend

# The v0 root their own CLI defaults to, which every request a test reads back hangs off.
_ROOT = "https://console.vast.ai/api/v0"
# The marker the onstart wrapper echoes after the command, carrying its real exit code.
_MARKER = "mainboard-exit:"

# The four constant filters their console applies to every search, plus this backend's own page
# size and storage figure. Every narrowing a caller asks for is added on top of exactly this.
_BASE_QUERY = {
    "verified": {"eq": True},
    "external": {"eq": False},
    "rentable": {"eq": True},
    "rented": {"eq": False},
    "type": "on-demand",
    "order": [["dph_total", "asc"]],
    "allocated_storage": 64.0,
    "limit": 32,
    "cuda_max_good": {"gte": VastBackend.CUDA_FLOOR},
    "compute_cap": {"gte": 750},
    "inet_down": {"gte": 500.0},
}

# `actual_status` values that mean the container has not run the command yet, so no marker can
# exist and the row alone answers.
_PENDING = ("created", "loading")
# `actual_status` values whose container has been up, so the log is asked for a marker first.
# Without one, a container still up keeps waiting, while anything else (a state Vast has not
# invented yet, a row carrying none, or a container that stopped before the wrapper spoke) is
# unknown rather than a crash or a clean run.
_STARTED = {
    "running": "running",
    "stopping": "running",
    "exited": "unknown",
    "mystery": "unknown",
    "": "unknown",
}

# The statuses their own docs call terminal, keyed with the exit status the wrapper echoed, since
# a container status says only that the container stopped and never how the command ended.
_TERMINAL = {
    ("exited", 0): "ok",
    ("stopped", 3): "failed",
    ("offline", 137): "failed",
    ("error", 1): "failed",
}


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
    row.update({"reliability2": 0.99, "cuda_max_good": 13.3, "compute_cap": 890})
    row.update(extra)
    return row


def terminal_backend(status: str, *, log: str) -> VastBackend:
    """A backend whose instance is terminal and whose log tail is `log`, in reply order."""
    return vast_backend(
        {"instances": {"id": 7, "actual_status": status}},
        {"result_url": "https://s3.example/logs/7.log"},
        log,
    )


_OFFERS = {"offers": [offer(11, dph=0.5), offer(22, dph=0.2, bid=0.05)]}
# Two offers whose on-demand and bid orderings disagree, so a spot search that forgot to re-rank
# would hand back the on-demand order and be caught.
_MIXED = {"offers": [offer(11, dph=0.5, bid=0.01), offer(22, dph=0.2, bid=0.09)]}
_CREATED = {"success": True, "new_contract": 4242}


@pytest.fixture(autouse=True)
def _key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly one Vast key spelling in the environment, whatever the machine already exports.

    `api_key` reads either name, so a workspace `.env` carrying `VASTAI_API_KEY` used to keep the
    no-key test finding a key it never set. The fallback spelling is cleared here so the suite
    reads the same on a keyed machine as on a bare one.
    """
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)


def test_api_key_reads_either_spelling_and_refuses_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpuhunt reads `VASTAI_API_KEY` while Vast's own CLI documents `VAST_API_KEY`."""
    assert api_key() == "key-123"
    monkeypatch.delenv("VAST_API_KEY")
    monkeypatch.setenv("VASTAI_API_KEY", "key-456")
    assert api_key() == "key-456"
    monkeypatch.delenv("VASTAI_API_KEY")
    with pytest.raises(MissionError, match="VAST_API_KEY"):
        api_key()


@pytest.mark.parametrize(
    ("spot", "narrowing", "extra", "ranked"),
    [
        pytest.param(False, {}, {}, [22, 11], id="the-whole-market-on-demand"),
        pytest.param(
            False,
            {"gpu_name": "RTX_4090", "gpus": 2, "max_usd_hr": 0.4, "limit": 5},
            {
                "gpu_name": {"eq": "RTX 4090"},
                "num_gpus": {"eq": 2},
                "dph_total": {"lte": 0.4},
                "limit": 5,
            },
            [22],
            id="one-card-a-count-and-a-ceiling-with-underscores-read-as-spaces",
        ),
        pytest.param(True, {}, {"type": "bid"}, [11, 22], id="the-whole-market-at-the-bid-floor"),
    ],
)
def test_search_posts_the_consoles_own_filters_and_ranks_by_what_a_rental_will_pay(
    spot: bool, narrowing: dict, extra: Mapping, ranked: list[int]
) -> None:
    """The offer search filters the market and ranks by what will actually be paid.

    The four constant filters keep unverified hosts, resold capacity and already-rented
    machines out. Vast ranks by the on-demand total whichever mode is asked for, so a spot
    search is re-ranked here by the bid floor it will actually pay.
    """
    backend = vast_backend(_MIXED, spot=spot)
    assert [row["id"] for row in backend.search(**narrowing)] == ranked
    (request,) = backend.transport.calls
    assert request.full_url == f"{_ROOT}/bundles/"
    assert request.get_header("Authorization") == "Bearer key-123"
    assert backend.transport.bodies == [_BASE_QUERY | extra]


def test_pick_never_prefers_reliability_over_the_returned_price_ceiling() -> None:
    rows = {"offers": [offer(11, dph=0.25, reliability2=1.0), offer(22, dph=0.17)]}
    assert vast_backend(rows).pick(gpu_name="RTX 4090", gpus=1, max_usd_hr=0.18)["id"] == 22
    with pytest.raises(MissionError, match="no rentable"):
        vast_backend(rows, rows).pick(gpu_name="RTX 4090", gpus=1, max_usd_hr=0.1)


def test_create_refusal_keeps_the_provider_reason_without_a_traceback() -> None:
    refused = HTTPError(
        f"{_ROOT}/asks/11/", 400, "Bad Request", Message(), BytesIO(b'{"msg":"ask expired"}')
    )
    backend = vast_backend(refused)
    with pytest.raises(MissionError, match="offer 11.*400.*ask expired"):
        backend.rented(
            offer(11, dph=0.17),
            plan=vast_plan(),
            launch={"runtype": "ssh"},
            allocation=created_request(),
            resources=Resources(max_usd=1.0, walltime="00:30:00"),
        )


@pytest.mark.parametrize(
    ("offers", "spot", "rented"),
    [
        pytest.param(
            [offer(11, dph=0.5, reliability2=0.999), offer(22, dph=0.2)],
            False,
            11,
            id="the-most-reliable-machine-rather-than-the-cheapest-one",
        ),
        pytest.param(
            [offer(11, dph=0.5), offer(22, dph=0.2)],
            False,
            22,
            id="a-reliability-tie-broken-toward-the-cheaper-machine",
        ),
        pytest.param(
            [offer(11, dph=0.5, bid=0.01), offer(22, dph=0.2, bid=0.09)],
            True,
            11,
            id="a-spot-tie-broken-by-the-bid-floor-it-will-really-pay",
        ),
    ],
)
def test_pick_rents_the_most_reliable_machine_the_budget_already_allows(
    offers: list[dict], spot: bool, rented: int
) -> None:
    """Price decides admission and nothing more.

    Renting the lowest-priced listing put earlier rentals at the bottom of the market, where
    the container is billed for and never starts.
    """
    backend = vast_backend({"offers": offers}, spot=spot)
    assert backend.pick(gpu_name="RTX 4090", gpus=1)["id"] == rented


@pytest.mark.parametrize(
    ("gpu_name", "max_usd_hr", "refusal"),
    [
        pytest.param(
            "H100",
            2.0,
            r"no rentable 1x H100 offer under \$2\.00/hr right now",
            id="a-ceiling-the-market-has-nothing-under",
        ),
        pytest.param(
            "", 0.0, "no rentable 1x any offer right now", id="a-market-with-nothing-in-it"
        ),
    ],
)
def test_pick_refuses_when_the_market_has_no_matching_offer(
    gpu_name: str, max_usd_hr: float, refusal: str
) -> None:
    with pytest.raises(MissionError, match=refusal):
        vast_backend({}, {}).pick(gpu_name=gpu_name, gpus=1, max_usd_hr=max_usd_hr)


def test_pick_refuses_a_market_whose_every_driver_is_below_the_cuda_floor() -> None:
    """The refusal names the floor, the best CUDA on offer, and the offer carrying it.

    The floored search comes back empty and the unfloored one does not, which is a card whose
    hosts have all aged out rather than a card nobody is renting. Reported as a skip, that is
    exactly the silence that turned five dispatches into contract ids and destroyed instances on
    2026-08-27, when every live Tesla T4 offer read `cuda_max_good = 12.6` under a 12.9 image.
    """
    stale = {
        "offers": [
            offer(11, dph=0.4, cuda_max_good=12.6),
            offer(22, dph=0.9, cuda_max_good=12.8, gpu_name="Tesla T4"),
        ]
    }
    backend = vast_backend({}, stale)
    with pytest.raises(MissionError) as refused_at:
        backend.pick(gpu_name="Tesla T4", gpus=1)
    refusal = str(refused_at.value)
    assert f"CUDA {VastBackend.CUDA_FLOOR}" in refusal, "the floor it failed is named"
    assert "12.8" in refusal, "the best CUDA actually on offer is named"
    assert "offer 22" in refusal, "the offending offer is named"
    assert "Texas, US" in refusal, "and where it is, so the reader can check it by hand"
    assert "destroys a container its driver cannot start" in refusal, "and what would happen"
    # The diagnosis costs one extra search, and only on the refusal path.
    floored, unfloored = backend.transport.bodies
    assert floored["cuda_max_good"] == {"gte": VastBackend.CUDA_FLOOR}
    assert floored["compute_cap"] == {"gte": 750}
    assert "cuda_max_good" not in unfloored
    assert "compute_cap" not in unfloored
    assert "inet_down" not in unfloored


def test_pick_refuses_a_market_whose_every_host_downloads_too_slowly() -> None:
    """The refusal names the download floor, the fastest host on offer, and where it is.

    A host that passes both CUDA floors can still spend the whole landing window pulling the
    image and end with no container, which is how three rentals went on 2026-09-12, so the
    floor rides on the search and its refusal is the last one asked, after the two CUDA floors.
    """
    slow = {
        "offers": [
            offer(11, dph=0.4, inet_down=80.0),
            offer(22, dph=0.9, inet_down=240.0, gpu_name="RTX 5090"),
        ]
    }
    backend = vast_backend({}, slow)
    with pytest.raises(MissionError) as refused_at:
        backend.pick(gpu_name="RTX 5090", gpus=1)
    refusal = str(refused_at.value)
    assert "500 Mbps" in refusal, "the floor it failed is named"
    assert "240 Mbps" in refusal, "the fastest host actually on offer is named"
    assert "offer 22" in refusal and "Texas, US" in refusal, "and which offer, and where"
    assert "no container" in refusal, "and what renting it would do"


def test_pick_refuses_an_architecture_this_cuda_no_longer_builds_for() -> None:
    """A driver new enough and a card too old is its own refusal, because it fails later.

    A live search on 2026-08-27 found Volta and Pascal machines reporting `cuda_max_good` of
    13.0, so the driver floor alone would have rented one. It would have booted, billed, and died
    at the first kernel launch with no cubin for its own card.
    """
    volta = {"offers": [offer(33, dph=0.1, compute_cap=700, gpu_name="Tesla V100")]}
    backend = vast_backend({}, volta)
    with pytest.raises(MissionError) as refused_at:
        backend.pick(gpu_name="Tesla V100", gpus=1)
    refusal = str(refused_at.value)
    assert "700" in refusal, "the capability the card really has"
    assert "sm_75" in refusal, "the floor it failed"
    assert "offer 33" in refusal, "the offending offer"
    assert "Maxwell, Pascal and Volta" in refusal, "and which families went with it"


@pytest.mark.parametrize(
    ("reader", "field", "absent", "present"),
    [
        (cuda_max_good, "cuda_max_good", 0.0, 13.3),
        (capability, "compute_cap", 0, 890),
        (download, "inet_down", 0.0, 900.0),
    ],
    ids=["driver-version", "compute-capability", "download-rate"],
)
def test_a_row_that_publishes_nothing_never_satisfies_a_floor(
    reader: Callable[[Mapping], float], field: str, absent: float, present: float
) -> None:
    """Silence reads as too old, since the unknown machine is the one that costs a rental."""
    blank = offer(11, dph=0.4)
    blank.pop(field, None)
    assert reader(blank) == absent
    assert reader(offer(11, dph=0.4, **{field: "not a number"})) == absent
    assert reader(offer(11, dph=0.4, **{field: present})) == present


@given(
    hours=st.integers(min_value=1, max_value=24),
    minutes=st.integers(min_value=0, max_value=59),
    budget=st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
)
def test_the_hourly_cap_spends_exactly_the_budget_over_the_walltime_a_job_declares(
    hours: int, minutes: int, budget: float
) -> None:
    """A spend cap needs a walltime before it can bound an hourly rental.

    A walltime-less request searches the whole market and leans on `max_usd` alone.
    """
    capped = Resources(max_usd=budget, walltime=f"{hours:02d}:{minutes:02d}:00")
    assert VastBackend.hourly_cap(capped) * (hours + minutes / 60) == pytest.approx(budget)
    assert VastBackend.hourly_cap(Resources(max_usd=budget)) == 0.0
    assert VastBackend.hourly_cap(Resources(max_usd=budget, walltime="00:00:00")) == 0.0


def test_submit_refuses_before_any_network_call_when_the_budget_is_unset() -> None:
    backend = vast_backend()
    with pytest.raises(MissionError, match="max-usd"):
        backend.submit(vast_plan(), "echo hi", Resources(), allocation=created_request())
    assert backend.transport.calls == []


def test_submit_rents_the_picked_offer_as_a_one_shot_container_and_returns_its_contract() -> None:
    """A rental runs the image one-shot and a failed rent never parks an instance.

    `args` launch mode runs the image as it is, with `onstart` as the entrypoint and `args` as
    its argv, which is how the official CLI spells a one-shot container. Vast reports container
    status and never a process exit code, so the wrapper echoes the real one into the log, and
    the rent fails outright rather than parking a stopped instance we would owe storage on.

    The receipts file is staged before the command and framed back after it, because vast cuts
    every log line at 500 characters and a printed receipt would arrive here in half.
    """
    backend = vast_backend(_OFFERS, _CREATED)
    handle = backend.submit(
        vast_plan(),
        "python train.py",
        Resources(max_usd=5.0, gpus=2),
        allocation=created_request(),
    )
    assert handle == "4242"
    assert backend.transport.urls == [f"{_ROOT}/bundles/", f"{_ROOT}/asks/22/"]
    assert backend.transport.calls[1].get_method() == "PUT"
    search, create = backend.transport.bodies
    assert search["num_gpus"] == {"eq": 2}
    assert create == {
        "client_id": "me",
        "image": "vastai/base-image:cuda-13.3.1-auto",
        "disk": 64.0,
        "label": "mainboard-test-creation",
        "runtype": "args",
        "onstart": "bash",
        "args": [
            "-c",
            f"{staging()}\npython train.py\nstatus=$?\n{framing()}\n"
            "echo mainboard-exit:$status\nexit $status\n",
        ],
        "cancel_unavail": True,
    }


def test_submit_narrows_the_search_and_the_rental_to_what_the_request_asks_for() -> None:
    """Every resource field lands in the search, the bid, or the image.

    A request naming no GPU count still rents one machine's worth, a walltime turns the spend
    cap into the hourly ceiling the search filters on, a spot rental bids the offer's own
    floor, and a containerized plan rents under its own image rather than Vast's base one.
    """
    backend = vast_backend(_OFFERS, _CREATED, spot=True)
    backend.submit(
        vast_plan(container=Container(image="pytorch/pytorch:latest")),
        "echo hi",
        Resources(max_usd=1.0, walltime="02:00:00"),
        allocation=created_request(),
    )
    search, create = backend.transport.bodies
    assert search["num_gpus"] == {"eq": 1}
    assert search["dph_total"] == {"lte": pytest.approx(0.5)}
    assert search["type"] == "bid"
    assert create["price"] == pytest.approx(0.05)
    assert create["image"] == "pytorch/pytorch:latest"


def test_state_of_a_container_that_has_not_started_the_command_costs_no_log_fetch() -> None:
    """A container still being created has no marker to read, so the instance row alone answers."""
    read = {}
    for status in _PENDING:
        backend = vast_backend({"instances": {"id": 7, "actual_status": status}})
        state = backend.state("7")
        read[status] = (state.state, state.verdict, state.exit_code)
        assert backend.transport.urls == [f"{_ROOT}/instances/7/?owner=me"]
    assert read == {status: (status, "running", None) for status in _PENDING}


def test_a_container_that_has_been_up_is_asked_for_a_marker_before_its_status_is_read() -> None:
    """The marker is the only thing that knows the command ended, so it is asked for first.

    Without one the container's own status decides, and it decides only between waiting longer
    and admitting the run is unreadable. A clean container stop is never read as a clean run.
    """
    read = {}
    for status in _STARTED:
        backend = terminal_backend(status, log="epoch 1\nepoch 2\n")
        state = backend.state("7")
        read[status] = (state.state, state.verdict, state.exit_code)
        assert backend.transport.urls == [
            f"{_ROOT}/instances/7/?owner=me",
            f"{_ROOT}/instances/request_logs/7/",
            "https://s3.example/logs/7.log",
        ]
    assert read == {status: (status, verdict, None) for status, verdict in _STARTED.items()}


def test_a_container_vast_restarted_settles_on_the_marker_the_finished_command_left() -> None:
    """The money leak: Vast restarts the container it exited, so a finished run reads `running`.

    An instance is held at its intended status, so the exited container comes back up and the
    command runs again. A sweep that believed that status never reached a terminal verdict, never
    cancelled, and left the meter running on work that was already done (eight instances still
    billing after a campaign had finished, $2.35 against $0.65 expected, 2026-08-26). The last
    marker is what says the command ended, however many times the container has come back.
    """
    backend = terminal_backend(
        "running", log=f"{_MARKER}0\nrestarted\ntraining done\n{_MARKER}0\n"
    )
    state = backend.state("7")
    assert (state.state, state.exit_code, state.verdict) == ("running", 0, "ok")


def test_state_of_a_terminal_container_reports_the_process_exit_code() -> None:
    """The exit verdict is the wrapper's own marker.

    The marker is echoed after the command, so it describes the command rather than the
    container that happened to stop cleanly around it.
    """
    read = {}
    for status, code in _TERMINAL:
        backend = terminal_backend(status, log=f"training done\n{_MARKER}{code}\n")
        state = backend.state("7")
        read[status, code] = (state.state, state.exit_code, state.verdict)
        assert backend.transport.urls == [
            f"{_ROOT}/instances/7/?owner=me",
            f"{_ROOT}/instances/request_logs/7/",
            "https://s3.example/logs/7.log",
        ]
    assert read == {key: (key[0], key[1], verdict) for key, verdict in _TERMINAL.items()}


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param(
            ({"result_url": "https://s3.example/logs/7.log"}, "killed mid-epoch\n"),
            id="a-container-killed-before-the-wrapper-spoke",
        ),
        pytest.param(
            ({"success": False, "msg": "instance not running"},),
            id="a-log-upload-vast-refused",
        ),
        pytest.param((not_found(),), id="a-log-upload-that-is-already-gone"),
    ],
)
def test_state_stays_unknown_when_the_log_cannot_say_how_the_command_ended(
    replies: tuple[Reply, ...],
) -> None:
    """An unknown verdict is honest where reading a clean container stop as a clean run is not."""
    backend = vast_backend({"instances": {"id": 7, "actual_status": "exited"}}, *replies)
    state = backend.state("7")
    assert (state.state, state.exit_code, state.verdict) == ("exited", None, "unknown")


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        pytest.param({"instances": None}, "vanished", id="an-instance-row-vast-nulled"),
        pytest.param(not_found(), "vanished", id="an-instance-vast-has-already-forgotten"),
        pytest.param(refused(401), None, id="a-refusal-that-is-not-a-missing-instance"),
    ],
)
def test_state_reads_a_gone_instance_as_vanished_and_re_raises_anything_else(
    reply: Reply, verdict: str | None
) -> None:
    """A destroyed instance reads empty however long ago it went.

    It answers either a null row or a 404 depending on the age, and a post-mortem reads both
    the same way.
    """
    backend = vast_backend(reply)
    if verdict is None:
        with pytest.raises(HTTPError, match="401"):
            backend.state("7")
    else:
        assert backend.state("7").verdict == verdict


@given(
    codes=st.lists(st.integers(min_value=-1, max_value=255), min_size=1, max_size=3),
    chatter=st.lists(WORDS, max_size=3),
)
def test_exit_sentinel_reads_the_last_status_the_wrapper_echoed(
    codes: Sequence[int], chatter: list[str]
) -> None:
    """The last marker wins.

    A container Vast restarted appends its own line below the first and the command ran again
    (thirteen restarts in five minutes, verified live).
    """
    lines = [*chatter, *(f"{_MARKER}{code}" for code in codes)]
    assert exit_sentinel("\n".join(lines) + "\n") == codes[-1]


@pytest.mark.parametrize(
    ("log", "status"),
    [
        pytest.param(
            f"{_MARKER}1\n{_MARKER}truncated", 1, id="a-marker-carrying-no-number-at-all"
        ),
        pytest.param("nothing to see here\n", None, id="a-log-without-any-marker"),
    ],
)
def test_exit_sentinel_skips_what_it_cannot_read_as_a_status(log: str, status: int | None) -> None:
    assert exit_sentinel(log) == status


def test_logs_requests_an_upload_then_polls_for_it_without_the_api_key() -> None:
    """A log fetch retries until storage has the file.

    `request_logs` answers before the log reaches storage, so the first fetches come back 404
    until it lands, and the url is storage's own signed link rather than ours.
    """
    naps = Naps()
    backend = vast_backend(
        {"result_url": "https://s3.example/logs/7.log"}, not_found(), "landed at last", naps=naps
    )
    assert backend.logs("7") == "landed at last"
    ask, first, second = backend.transport.calls
    assert ask.full_url == f"{_ROOT}/instances/request_logs/7/"
    assert ask.get_method() == "PUT"
    assert backend.transport.bodies == [{"tail": "2000"}]
    assert first.full_url == second.full_url == "https://s3.example/logs/7.log"
    assert second.get_header("Authorization") is None
    assert naps.waited == [1.0]


@pytest.mark.parametrize(
    ("replies", "refusal", "polls"),
    [
        pytest.param(
            ({"success": False, "msg": "instance not running"},),
            "instance not running",
            0,
            id="an-upload-vast-refused-outright",
        ),
        pytest.param(
            ({"result_url": "http://s3.example/logs/7.log"},),
            "non-https",
            0,
            id="a-log-url-that-is-not-https",
        ),
        pytest.param(
            ({"result_url": "https://s3.example/logs/7.log"}, *[not_found()] * 20),
            r"fetch https://s3\.example/logs/7\.log directly",
            20,
            id="an-upload-that-never-lands",
        ),
    ],
)
def test_logs_hands_the_url_over_when_it_cannot_bring_the_log_back_itself(
    replies: tuple[Reply, ...], refusal: str, polls: int
) -> None:
    """The poll is bounded, so a log in flight costs a fixed wait and never a wedged sweep."""
    naps = Naps()
    backend = vast_backend(*replies, naps=naps)
    with pytest.raises(MissionError, match=refusal):
        backend.logs("7")
    assert naps.waited == [1.0] * polls


@pytest.mark.parametrize(
    ("spot", "limit", "asked", "rates"),
    [
        pytest.param(False, 0, 32, [0.2, 0.5], id="this-backends-own-page-size-on-demand"),
        pytest.param(
            True, 5, 5, [0.05, 0.1], id="a-page-size-the-caller-asked-for-at-the-bid-floor"
        ),
    ],
)
def test_catalog_turns_a_live_search_into_priced_offer_rows(
    spot: bool, limit: int, asked: int, rates: Sequence[float]
) -> None:
    """The authed refresh of the imported price feed, priced by the mode that rents the machine."""
    backend = vast_backend(_OFFERS, spot=spot)
    rows = backend.catalog(gpu_name="RTX 4090", limit=limit)
    assert [row.rate_usd_hr for row in rows] == [pytest.approx(rate) for rate in rates]
    assert {row.provider for row in rows} == {"vast"}
    assert {row.spot for row in rows} == {spot}
    assert (rows[0].gpu, rows[0].region, rows[0].source) == (
        "RTX 4090",
        "Texas, US",
        "probed:vast",
    )
    assert rows[0].available is True
    assert backend.transport.bodies[0]["limit"] == asked


@pytest.mark.parametrize(
    ("account", "offers", "credit", "usd_hr", "note"),
    [
        pytest.param(
            {"credit": 42.5},
            [offer(11, dph=0.31)],
            42.5,
            0.31,
            "1x RTX 4090 Texas, US",
            id="a-credited-account-and-a-live-sample-offer",
        ),
        pytest.param(
            {"credit": 3.0},
            [],
            3.0,
            None,
            "no 1x RTX 4090 offer right now",
            id="a-sample-card-nobody-is-renting",
        ),
        pytest.param(
            {},
            [offer(11, dph=0.4, geolocation="")],
            None,
            0.4,
            "1x RTX 4090",
            id="an-account-that-reports-no-credit-on-a-machine-with-no-location",
        ),
        pytest.param(
            {},
            [offer(11, dph=0.4, geolocation=", US")],
            None,
            0.4,
            "1x RTX 4090 US",
            id="a-machine-whose-city-is-unset",
        ),
    ],
)
def test_standing_reads_the_credit_and_prices_one_sample_card(
    account: dict, offers: list[dict], credit: float | None, usd_hr: float | None, note: str
) -> None:
    """Standing reads the prepaid credit and prints a tidy location.

    `credit` is the spendable figure on a prepaid account, where its sibling `balance` is the
    invoicing one and sits at zero, and a machine whose city is unset carries its country as
    `, US`, so the separator is trimmed rather than printed as a stray comma.
    """
    backend = vast_backend(account, {"offers": offers})
    standing = backend.standing()
    assert standing.keyed is True
    assert standing.credit_usd == (None if credit is None else pytest.approx(credit))
    assert standing.usd_hr == (None if usd_hr is None else pytest.approx(usd_hr))
    assert standing.note == note
    assert backend.transport.urls == [f"{_ROOT}/users/current/", f"{_ROOT}/bundles/"]
    assert backend.transport.calls[0].get_method() == "GET"
    assert backend.transport.bodies[1]["gpu_name"] == {"eq": "RTX 4090"}


def test_standing_without_a_key_names_the_variable_and_never_calls_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured Vast row costs nothing but the environment lookup."""
    monkeypatch.delenv("VAST_API_KEY")
    backend = vast_backend()
    standing = backend.standing()
    assert standing.keyed is False
    assert "VAST_API_KEY" in standing.note
    assert backend.transport.calls == []


@pytest.mark.parametrize(
    ("reply", "refuses"),
    [
        pytest.param({"success": True}, False, id="a-rental-whose-meter-is-still-running"),
        pytest.param(
            not_found(f"{_ROOT}/instances/4242/"),
            False,
            id="an-instance-vast-has-already-forgotten",
        ),
        pytest.param(refused(401), True, id="a-refusal-that-is-not-a-gone-instance"),
    ],
)
def test_cancel_destroys_the_rental_and_treats_one_vast_already_forgot_as_ended(
    reply: Reply, refuses: bool
) -> None:
    """Cancel is the call that stops the meter, and it tolerates repeats.

    A finished command leaves the rental up, and cancel is asked more than once by design, by
    a sweep that settles the same run twice and by anyone who already destroyed the instance
    in the console.
    """
    backend = vast_backend(reply)
    with pytest.raises(HTTPError) if refuses else nullcontext():
        backend.cancel("4242")
    (request,) = backend.transport.calls
    assert request.full_url == f"{_ROOT}/instances/4242/"
    assert request.get_method() == "DELETE"


@pytest.mark.parametrize("reply", [{}, {"success": False}, {"success": "true"}])
def test_destroy_requires_explicit_provider_confirmation(reply: dict) -> None:
    """HTTP success alone must not make a billable rental disappear from monitoring."""
    with pytest.raises(MissionError, match="release remains pending"):
        vast_backend(reply).cancel("4242")


def test_the_declared_delivery_gap_points_at_the_logs_verb_instead() -> None:
    """A rented machine's disk dies with the instance, so there is nothing to deliver from here."""
    advice = vast_backend().refusal(Delivery, handle="4242", path="out/results.json")
    assert advice == (
        "vast backend cannot deliver 'out/results.json' yet; a rented machine's disk dies with "
        "the instance, so have the command upload its own results and read `logs 4242` "
        "until that path lands"
    )


def rental_backend(*responses: Reply, naps: Naps | None = None) -> VastBackend:
    """A backend queued with a search, a create, a key attach and then instance rows."""
    return vast_backend(_OFFERS, _CREATED, {"success": True}, *responses, naps=naps)


def running(**extra: str | int) -> dict:
    """One instance row for a machine that is up and publishes its proxy ssh endpoint."""
    row = {"id": 4242, "actual_status": "running", "ssh_host": "ssh5.vast.ai", "ssh_port": 41022}
    row.update(extra)
    return {"instances": row}


def test_a_rental_is_created_waiting_for_a_landing_rather_than_running_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare image has no workspace, no tool and no environment, so the job cannot start yet.

    The entrypoint holds the machine still until the dispatch has put all three on it, and the
    exit marker stays exactly where it was, since the log is still the only thing that leaves a
    rental. The minutes that landing costs are inside the hourly ceiling the search filters on,
    which is why the same budget buys a cheaper machine here than a prebuilt container would.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    key = keypair(tmp_path)
    monkeypatch.setattr(vast_module, "reachable", lambda endpoint, *, sleeper: endpoint)
    backend = rental_backend(running())
    allocation = created_request()
    transport = backend.transport

    def observed(request: Request):
        if "/asks/" in request.full_url:
            assert allocation.cache.run(allocation.label).verdict == "submitting"
        if "/instances/4242" in request.full_url:
            assert allocation.cache.run("4242").creation == allocation.label
        return transport(request)

    backend = VastBackend(transport=observed, sleeper=Naps())
    rental = backend.rent(
        vast_plan(),
        Resources(max_usd=1.0, walltime="00:30:00", gpus=1),
        allocation=allocation,
    )
    attached = transport.urls.index(f"{_ROOT}/instances/4242/ssh/")
    search, create = transport.bodies[:2]
    attach = transport.bodies[attached]
    assert search["dph_total"] == {"lte": pytest.approx(1.0 * 3600 / (1800 + LANDING_SECONDS))}
    assert create["runtype"] == "ssh" and "args" not in create
    assert create["label"] == allocation.label
    assert create["image"] == "vastai/base-image:cuda-13.3.1-auto"
    assert waiting() in create["onstart"]
    assert create["onstart"].endswith(f"echo {_MARKER}$status\nexit $status\n")
    # The entrypoint seeds the landing's own key before it waits, so a host whose key injection
    # never lands or lands with the wrong modes still lets the landing in.
    seeding, _, rest = create["onstart"].partition(waiting())
    assert "'ssh-ed25519 AAAA me@here' >> /root/.ssh/authorized_keys" in seeding
    assert "chmod 600 /root/.ssh/authorized_keys" in seeding
    assert rest, "the seeding comes first and the wait follows it"
    assert attach == {"ssh_key": "ssh-ed25519 AAAA me@here"}
    assert rental.handle == "4242"
    assert rental.endpoint.destination == "root@ssh5.vast.ai"
    assert (rental.endpoint.port, rental.endpoint.identity) == (41022, str(key))


def test_a_create_refused_as_no_such_ask_re_picks_the_next_offer_on_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The market moves between the search and the create, and the next best offer is asked.

    Vast answers a create for an offer someone else just took with `no_such_ask` (RTX 5090
    offer 26371154, 2026-09-12); that is a definitive nothing-was-created, so the reservation
    reopens and the next offer on the same page is rented without a second search.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    keypair(tmp_path)
    monkeypatch.setattr(vast_module, "reachable", lambda endpoint, *, sleeper: endpoint)
    taken = HTTPError(
        f"{_ROOT}/asks/11/",
        400,
        "Bad Request",
        Message(),
        BytesIO(b'{"msg":"error 404/3603: no_such_ask  Instance type by id 11 is not available"}'),
    )
    page = {"offers": [offer(11, dph=0.5, reliability2=0.999), offer(22, dph=0.2)]}
    backend = vast_backend(page, taken, _CREATED, {"success": True}, running(), naps=Naps())
    resources = Resources(max_usd=1.0, walltime="00:30:00", gpus=1)
    rental = backend.rent(vast_plan(), resources, allocation=created_request())
    asked = [url for url in backend.transport.urls if "/asks/" in url]
    assert asked == [f"{_ROOT}/asks/11/", f"{_ROOT}/asks/22/"], "the taken offer, then the next"
    assert backend.transport.urls.count(f"{_ROOT}/bundles/") == 1, "one page serves both picks"
    assert rental.handle == "4242"


def test_a_definitive_create_refusal_closes_the_reservation_it_opened() -> None:
    """A 4xx on the create is the provider declining before any instance exists.

    The reservation crossed the API boundary, so it sat `submitting`; a refusal that proves
    nothing was created puts it back to prepared, so the same rent can try another offer and
    the dispatcher's exit can close it, instead of blocking every later request for the same
    script until someone reconciles a label that never reached the provider.
    """
    refused = HTTPError(
        f"{_ROOT}/asks/11/", 400, "Bad Request", Message(), BytesIO(b'{"msg":"ask expired"}')
    )
    backend = vast_backend(refused)
    allocation = created_request()
    with pytest.raises(MissionError, match="offer 11"):
        backend.rented(
            offer(11, dph=0.17),
            plan=vast_plan(),
            launch={"runtype": "ssh"},
            allocation=allocation,
            resources=Resources(max_usd=1.0, walltime="00:30:00"),
        )
    reopened = allocation.cache.creation(allocation.label, allocation.record.target)
    assert reopened.verdict == "prepared", "back where it stood before the boundary"
    allocation.interrupted()
    closed = allocation.cache.creation(allocation.label, allocation.record.target)
    assert closed.verdict == "failed", "and the dispatcher's exit closes it"


def test_a_server_fault_on_the_create_keeps_the_reservation_for_a_reconciliation() -> None:
    """A 5xx proves nothing either way, so the label stays `submitting` until it is reconciled.

    Its body is a gateway's page rather than the API's JSON, and the refusal still names the
    offer and the status instead of failing on the body it could not read.
    """
    fault = HTTPError(
        f"{_ROOT}/asks/11/", 502, "Bad Gateway", Message(), BytesIO(b"<html>gateway</html>")
    )
    backend = vast_backend(fault)
    allocation = created_request()
    with pytest.raises(MissionError, match=r"offer 11 \(HTTP 502\): $"):
        backend.rented(
            offer(11, dph=0.17),
            plan=vast_plan(),
            launch={"runtype": "ssh"},
            allocation=allocation,
            resources=Resources(max_usd=1.0, walltime="00:30:00"),
        )
    held = allocation.cache.creation(allocation.label, allocation.record.target)
    assert held.verdict == "submitting"


def taken(identifier: int) -> HTTPError:
    """The refusal Vast answers a create with once someone else rented `identifier` first."""
    return HTTPError(
        f"{_ROOT}/asks/{identifier}/",
        400,
        "Bad Request",
        Message(),
        BytesIO(b'{"msg":"error 404/3603: no_such_ask"}'),
    )


@pytest.mark.parametrize(
    ("offers", "refusal"),
    [
        pytest.param([], "no rentable 1x any offer", id="a-market-with-nothing-in-it"),
        pytest.param([11], "no rentable 1x any offer", id="a-page-whose-every-offer-was-taken"),
        pytest.param([11, 22, 33, 44], "took 3 offers", id="a-market-moving-faster-than-a-rent"),
    ],
)
def test_a_rental_the_market_cannot_place_is_refused_rather_than_searched_again(
    offers: list[int], refusal: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty page, a page emptied by takers, or three takers in a row end the rent.

    The refusal's own unfloored search is the only other search, so a moving market cannot turn
    one rent into a loop over fresh pages, and every create it declined leaves the reservation
    back at prepared.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    keypair(tmp_path)
    page = {"offers": [offer(identifier, dph=0.5) for identifier in offers]}
    backend = vast_backend(page, *(taken(identifier) for identifier in offers[:3]), {})
    allocation = created_request()
    with pytest.raises(MissionError, match=refusal):
        backend.rent(
            vast_plan(), Resources(max_usd=1.0, walltime="00:30:00"), allocation=allocation
        )
    assert backend.transport.urls.count(f"{_ROOT}/bundles/") <= 2
    assert allocation.cache.creation(allocation.label, allocation.record.target).verdict == (
        "prepared"
    )


def test_a_rental_that_never_comes_up_is_destroyed_rather_than_left_billing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup is attempted while the durable registry retains the returned handle."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    keypair(tmp_path)
    naps = Naps()
    loading = [{"instances": {"id": 4242, "actual_status": "loading"}}] * 90
    backend = vast_backend(_OFFERS, _CREATED, *loading, {"success": True}, naps=naps)
    with pytest.raises(MissionError, match="never came up with an ssh address"):
        backend.rent(
            vast_plan(), Resources(max_usd=1.0, walltime="00:30:00"), allocation=created_request()
        )
    assert backend.transport.calls[-1].get_method() == "DELETE"
    assert backend.transport.urls[-1] == f"{_ROOT}/instances/4242/"
    assert naps.waited == [10.0] * 90


def test_a_provider_that_will_not_take_the_key_refuses_before_anything_is_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An instance nobody can log into is a rental that bills for a landing that cannot happen."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    keypair(tmp_path)
    backend = vast_backend(_OFFERS, _CREATED, running(), refused(403), {"success": True})
    with pytest.raises(MissionError, match="cloud.vast.ai/manage-keys"):
        backend.rent(
            vast_plan(), Resources(max_usd=1.0, walltime="00:30:00"), allocation=created_request()
        )
    assert backend.transport.calls[-1].get_method() == "DELETE"


def test_a_workspace_holding_no_key_pair_never_reaches_the_market(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is free here and costs a whole rental one call later."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    backend = rental_backend(running())
    with pytest.raises(MissionError, match="ssh-keygen"):
        backend.rent(
            vast_plan(), Resources(max_usd=1.0, walltime="00:30:00"), allocation=created_request()
        )
    assert backend.transport.calls == []


def test_the_driver_floor_is_read_off_the_oldest_image_a_rental_can_load() -> None:
    """The filter and the image were once written twice and drifted, 13.0 against 13.3.1.

    Three rentals on 2026-09-21 landed on hosts whose driver topped out at CUDA 13.0 under a
    13.3.1 image, never started a container, and were destroyed at the end of the address wait.
    """
    oldest = min(image_cuda(image) or 0 for image in vast_module._BASE_IMAGES)
    assert oldest == VastBackend.CUDA_FLOOR
    assert VastBackend.CUDA_FLOOR >= ProviderBackend.CUDA_FLOOR


@pytest.mark.parametrize(
    ("driver", "image"),
    [
        (13.0, "vastai/base-image:cuda-13.0.3-auto"),
        (13.2, "vastai/base-image:cuda-13.2.1-auto"),
        (13.4, "vastai/base-image:cuda-13.3.1-auto"),
    ],
)
def test_a_rental_takes_the_newest_base_image_its_driver_loads(driver: float, image: str) -> None:
    """One pinned 13.3.1 image shut every L40S, A100 and H100 host out of the market (2026-09-25).

    Those hosts run 13.0 to 13.2 drivers, which load an image no newer than their own CUDA, so
    each offer is matched to the newest base image it can start rather than filtered by one.
    """
    assert vast_module.base_image(offer(1, dph=0.5, cuda_max_good=driver)) == image


def test_rentals_list_every_instance_on_the_account_as_vast_reports_it() -> None:
    """The provider's own listing, whoever rented each machine, with its rate where it has one."""
    backend = vast_backend(
        {
            "instances": [
                {
                    "id": 51,
                    "label": "mainboard-abc",
                    "num_gpus": 2,
                    "gpu_name": "RTX 5090",
                    "actual_status": "running",
                    "dph_total": 0.81,
                },
                {"id": 52},
            ]
        }
    )
    first, second = backend.rentals()
    assert (first.handle, first.label, first.gpu, first.status, first.usd_hr) == (
        "51",
        "mainboard-abc",
        "2x RTX 5090",
        "running",
        0.81,
    )
    assert (second.gpu, second.usd_hr) == ("1x unknown card", None)
    assert backend.transport.urls == [f"{_ROOT}/instances/?owner=me"]
