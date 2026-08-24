import os
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from threading import Event, Thread
from time import sleep
from types import SimpleNamespace
from urllib.request import Request

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch.backends import (
    Account,
    Capability,
    Credentials,
    Delivery,
    HpcAiBackend,
    LogSource,
    Market,
    ModalBackend,
    ProviderBackend,
    Standing,
    VastBackend,
    base,
    http_transport,
    require_budget,
    route,
)
from mainboard.dispatch.backends import api_key as hpc_ai_key
from mainboard.dispatch.backends.modal import declared_credit
from mainboard.dispatch.backends.vast import api_key as vast_key
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import HostProfile

from .support import BareBackend, FakeTransport, hpc_ai_backend, vast_backend

# Money as a provider really quotes it, from a free rental up to a balance nobody has.
_MONEY = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)


class Blocking:
    """A workspace locator that holds the merge open until a test lets it finish.

    Standing in for `Project` is what makes the critical section observable, since a thread can
    only be caught inside `load` while `load` is still busy.
    """

    def __init__(self, root: Path, reading: Event, may_finish: Event) -> None:
        self.root = root
        self.reading = reading
        self.may_finish = may_finish

    def find_root(self, start: Path) -> Path:
        del start
        self.reading.set()
        self.may_finish.wait(timeout=5)
        return self.root


def test_the_workspace_env_defines_only_what_the_environment_lacks_and_is_read_once(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file read as data, with comments, blanks and junk skipped and quotes taken off.

    The one rule that matters beyond parsing is that the environment wins, so a key someone
    exported deliberately is never replaced by a line in a file they may have forgotten. The
    second load proves the merge happens once however many backends ask, which is what keeps a
    rewritten file from overwriting a value a provider is already holding.
    """
    monkeypatch.chdir(workspace)
    (workspace / ".env").write_text(
        """# provider keys

export VAST_API_KEY=from-the-file
HPCAI_API_KEY='already exported'
MODAL_CREDIT_USD = 30
a line that declares nothing
=headless
"""
    )
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_CREDIT_USD", raising=False)
    monkeypatch.setenv("HPCAI_API_KEY", "kept")
    assert Credentials().load() == ("VAST_API_KEY", "MODAL_CREDIT_USD")
    assert os.environ["VAST_API_KEY"] == "from-the-file"
    assert os.environ["MODAL_CREDIT_USD"] == "30"
    assert os.environ["HPCAI_API_KEY"] == "kept"
    (workspace / ".env").write_text("VAST_API_KEY=second\n")
    assert Credentials().load() == ()
    assert os.environ["VAST_API_KEY"] == "from-the-file"


def test_neither_a_workspace_without_an_env_file_nor_a_machine_outside_one_defines_anything(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are one plain empty answer rather than the refusal `find_root` raises."""
    monkeypatch.chdir(workspace)
    assert Credentials().load() == ()
    Credentials().loaded = False
    monkeypatch.chdir(workspace.parent)
    assert Credentials().load() == ()


def test_a_second_backend_asking_at_once_waits_for_the_whole_merge(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compute survey probes every provider at once, from a pool this loader does not own.

    A flag flipped before the file was read would let the second provider look its key up while
    the first is still merging, and report a paid account as unkeyed for no reason but timing.
    """
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    (workspace / ".env").write_text("VAST_API_KEY=one\n")
    credentials = Credentials()
    reading, may_finish = Event(), Event()
    monkeypatch.setattr(credentials, "project", Blocking(workspace, reading, may_finish))
    seen: list[str | None] = []

    def second() -> None:
        credentials.load()
        seen.append(os.environ.get("VAST_API_KEY"))

    first, other = Thread(target=credentials.load), Thread(target=second)
    first.start()
    reading.wait(timeout=5)
    other.start()
    other.join(timeout=0.03)
    assert other.is_alive()  # waiting on the merge rather than reporting an empty environment
    may_finish.set()
    first.join(timeout=5)
    other.join(timeout=5)
    assert seen == ["one"]


@pytest.mark.parametrize(
    ("reader", "variable", "declared", "expected"),
    [
        pytest.param(vast_key, "VAST_API_KEY", "from-the-env", "from-the-env", id="vast"),
        pytest.param(hpc_ai_key, "HPCAI_API_KEY", "from-the-env", "from-the-env", id="hpc-ai"),
        pytest.param(declared_credit, "MODAL_CREDIT_USD", "30", 30.0, id="modal"),
    ],
)
def test_a_providers_own_reader_finds_what_only_the_workspace_env_declares(
    reader: Callable[[], str | float],
    variable: str,
    declared: str,
    expected: str | float,
    unsealed: None,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each refusal tells someone to write that file, so reading it is ours and not their chore."""
    monkeypatch.chdir(workspace)
    monkeypatch.delenv(variable, raising=False)
    (workspace / ".env").write_text(f"{variable}={declared}\n")
    assert reader() == expected


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        pytest.param("ssh", "ssh-family", id="ssh"),
        pytest.param("pbs", "ssh-family", id="pbs"),
        pytest.param("slurm", "ssh-family", id="slurm"),
        pytest.param("local", "ssh-family", id="local"),
        pytest.param(HostProfile().kind, "ssh-family", id="an-unprobed-host-left-on-auto"),
        pytest.param("modal", ModalBackend, id="modal"),
        pytest.param("hpc-ai", HpcAiBackend, id="hpc-ai"),
        pytest.param("vast", VastBackend, id="vast"),
    ],
)
def test_route_sends_each_kind_down_the_path_that_can_run_it(
    kind: str, path: str | type[ProviderBackend]
) -> None:
    assert route(kind) == path


def test_route_raises_a_mission_error_naming_known_kinds_for_an_unregistered_kind() -> None:
    with pytest.raises(MissionError) as excinfo:
        route("ec2")
    assert "ec2" in str(excinfo.value)
    assert "modal" in str(excinfo.value)
    assert "hpc-ai" in str(excinfo.value)


def test_every_backend_carries_the_job_lifecycle_and_nothing_it_cannot_honor() -> None:
    """The capability map, asserted as a table so a new backend's shape is one line to read.

    Every contract is an optional half a backend opts into, so the lifecycle root carries none of
    them and a caller finds the real ones by `isinstance` instead of calling and being refused.
    """
    contracts = (Account, Delivery, LogSource, Market)
    assert all(issubclass(contract, Capability) for contract in contracts)
    assert not any(issubclass(ProviderBackend, contract) for contract in contracts)
    carried = {
        "modal": ModalBackend(),
        "hpc-ai": hpc_ai_backend(transport=FakeTransport()),
        "vast": vast_backend(),
        "bare": BareBackend(),
    }
    assert {
        name: sorted(contract.__name__ for contract in contracts if isinstance(backend, contract))
        for name, backend in carried.items()
    } == {
        "modal": ["Account", "LogSource"],
        "hpc-ai": ["Account"],
        "vast": ["Account", "LogSource", "Market"],
        "bare": [],
    }


@pytest.mark.parametrize(
    ("capability", "line"),
    [
        pytest.param(
            LogSource,
            "bare backend keeps no logs; read bare-1.log on the box instead",
            id="a-gap-the-backend-wrote-its-own-advice-for",
        ),
        pytest.param(
            Delivery,
            "the bare backend does not implement Delivery",
            id="a-gap-it-never-described",
        ),
    ],
)
def test_a_refusal_carries_the_backends_own_advice_or_a_plain_statement_of_the_gap(
    capability: type[Capability], line: str
) -> None:
    assert BareBackend().refusal(capability, handle="bare-1") == line


def test_the_audited_url_open_is_what_every_rest_backend_takes_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One seam reaches urllib, under the package's own deadline since urllib itself has none."""
    assert VastBackend().transport is http_transport
    assert HpcAiBackend().transport is http_transport
    assert VastBackend().sleeper is sleep
    seen: list[tuple[Request, float]] = []

    def fake_urlopen(request: Request, *, timeout: float) -> SimpleNamespace:
        seen.append((request, timeout))
        return SimpleNamespace(status=200, read=lambda: b"{}")

    monkeypatch.setattr(base, "urlopen", fake_urlopen)
    request = Request("https://example.test/probe")
    assert http_transport(request).read() == b"{}"
    seen_request, deadline = seen[0]
    assert seen_request is request
    assert 0 < deadline < 60


# Ten rather than the profile's thirty, because the whole rule is one `round` call and building
# a validated model is the most expensive thing this file does under coverage. The two figures
# below are the live ones the rule was written for, so they run whatever the sample draws.
@settings(max_examples=10)
@given(figure=_MONEY)
@example(figure=99.99680725539)
@example(figure=0.285925925926)
def test_standing_quotes_money_at_the_precision_money_has(figure: float) -> None:
    """Money keeps the precision money has.

    Four places rather than two, so two real hourly rates never read as the same price, and a
    figure the provider publishes none of stays absent rather than becoming a zero.
    """
    priced = Standing(keyed=True, credit_usd=figure, usd_hr=None)
    assert priced.credit_usd == pytest.approx(figure, abs=1e-4)
    assert priced.credit_usd == round(priced.credit_usd, 4)
    assert priced.usd_hr is None


@given(cap=_MONEY)
@example(cap=0.0)
@example(cap=5.0)
def test_require_budget_refuses_a_submission_nobody_capped(cap: float) -> None:
    """Every provider bills someone, so an uncapped submit is refused before any network call."""
    refuses = pytest.raises(MissionError, match="max-usd") if not cap else nullcontext()
    with refuses:
        require_budget(Resources(max_usd=cap))
