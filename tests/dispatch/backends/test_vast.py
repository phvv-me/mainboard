from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from urllib.error import HTTPError

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch.backends import Delivery, VastBackend
from mainboard.dispatch.backends.vast import api_key, exit_sentinel
from mainboard.dispatch.evidence import framing, staging
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import Container, HostProfile

from ...strategies import WORDS
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
    "allocated_storage": 16.0,
    "limit": 32,
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
    row.update({"reliability2": 0.99})
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
            [22, 11],
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
        vast_backend({}).pick(gpu_name=gpu_name, gpus=1, max_usd_hr=max_usd_hr)


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
        backend.submit(vast_plan(), "echo hi", Resources())
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
    handle = backend.submit(vast_plan(), "python train.py", Resources(max_usd=5.0, gpus=2))
    assert handle == "4242"
    assert backend.transport.urls == [f"{_ROOT}/bundles/", f"{_ROOT}/asks/22/"]
    assert backend.transport.calls[1].get_method() == "PUT"
    search, create = backend.transport.bodies
    assert search["num_gpus"] == {"eq": 2}
    assert create == {
        "client_id": "me",
        "image": "vastai/base-image:cuda-12.9.2-auto",
        "disk": 16.0,
        "label": "mainboard-provider-host",
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


def test_the_declared_delivery_gap_points_at_the_logs_verb_instead() -> None:
    """A rented machine's disk dies with the instance, so there is nothing to deliver from here."""
    advice = vast_backend().refusal(Delivery, handle="4242", path="out/results.json")
    assert advice == (
        "vast backend cannot deliver 'out/results.json' yet; a rented machine's disk dies with "
        "the instance, so have the command upload its own results and read `logs 4242` "
        "until that path lands"
    )
