import os
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from urllib.request import Request

import pytest

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
from mainboard.dispatch.schedulers import Resources
from mainboard.manifest import HostProfile

from .conftest import BareBackend, FakeTransport, hpc_ai_backend, vast_backend

# --- credentials ---


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


def test_the_workspace_env_defines_only_what_the_environment_lacks(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file read as data, with comments, blanks and junk skipped and quotes taken off.

    The one rule that matters beyond parsing is that the environment wins, so a key someone
    exported deliberately is never replaced by a line in a file they may have forgotten.
    """
    monkeypatch.chdir(workspace)
    (workspace / ".env").write_text(
        "# provider keys\n"
        "\n"
        "export VAST_API_KEY=from-the-file\n"
        "HPCAI_API_KEY='already exported'\n"
        "MODAL_CREDIT_USD = 30\n"
        "a line that declares nothing\n"
        "=headless\n"
    )
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_CREDIT_USD", raising=False)
    monkeypatch.setenv("HPCAI_API_KEY", "kept")
    assert Credentials().load() == ("VAST_API_KEY", "MODAL_CREDIT_USD")
    assert os.environ["VAST_API_KEY"] == "from-the-file"
    assert os.environ["MODAL_CREDIT_USD"] == "30"
    assert os.environ["HPCAI_API_KEY"] == "kept"


def test_the_workspace_env_is_read_once_however_many_backends_ask(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    (workspace / ".env").write_text("VAST_API_KEY=first\n")
    assert Credentials().load() == ("VAST_API_KEY",)
    (workspace / ".env").write_text("VAST_API_KEY=second\n")
    assert Credentials().load() == ()
    assert os.environ["VAST_API_KEY"] == "first"


def test_a_workspace_with_no_env_file_defines_nothing(
    unsealed: None, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
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
    other.join(timeout=0.1)
    assert other.is_alive()  # waiting on the merge rather than reporting an empty environment
    may_finish.set()
    first.join(timeout=5)
    other.join(timeout=5)
    assert seen == ["one"]


def test_a_machine_standing_outside_a_workspace_defines_nothing(
    unsealed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No manifest above the cwd is a plain empty answer, not the refusal `find_root` raises."""
    monkeypatch.chdir(tmp_path)
    assert Credentials().load() == ()


# --- route ---


@pytest.mark.parametrize("kind", ["ssh", "pbs", "slurm", "local"])
def test_route_keeps_ssh_family_kinds_on_the_scheduler_path(kind: str) -> None:
    assert route(kind) == "ssh-family"


def test_route_resolves_a_registered_provider_backend_by_kind() -> None:
    assert route("modal") is ModalBackend
    assert route("hpc-ai") is HpcAiBackend
    assert route("vast") is VastBackend


def test_route_raises_a_mission_error_naming_known_kinds_for_an_unregistered_kind() -> None:
    with pytest.raises(MissionError) as excinfo:
        route("ec2")
    assert "ec2" in str(excinfo.value)
    assert "modal" in str(excinfo.value)
    assert "hpc-ai" in str(excinfo.value)


# --- the core contract and its capabilities ---


def test_provider_backend_root_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ProviderBackend()  # type: ignore[abstract]  reason=proving the abstract contract is enforced since=2026-08-17


def test_every_backend_carries_the_job_lifecycle_and_nothing_it_cannot_honor() -> None:
    """The capability map, asserted as a table so a new backend's shape is one line to read."""
    carried = {
        "modal": ModalBackend(),
        "hpc-ai": hpc_ai_backend(transport=FakeTransport()),
        "vast": vast_backend(),
        "bare": BareBackend(),
    }
    assert {
        name: sorted(
            contract.__name__
            for contract in (Account, Delivery, LogSource, Market)
            if isinstance(backend, contract)
        )
        for name, backend in carried.items()
    } == {
        "modal": ["Account", "LogSource"],
        "hpc-ai": ["Account"],
        "vast": ["Account", "LogSource", "Market"],
        "bare": [],
    }


@pytest.mark.parametrize("contract", [Account, Delivery, LogSource, Market])
def test_every_capability_is_one_of_the_optional_halves_of_the_contract(
    contract: type[Capability],
) -> None:
    assert issubclass(contract, Capability)
    assert not issubclass(ProviderBackend, contract)


# --- refusal ---


def test_a_declared_gap_refuses_with_the_backends_own_advice() -> None:
    assert BareBackend().refusal(LogSource, handle="bare-1") == (
        "bare backend keeps no logs; read bare-1.log on the box instead"
    )


def test_an_undeclared_gap_refuses_by_naming_the_contract_it_never_implemented() -> None:
    assert BareBackend().refusal(Delivery) == ("the bare backend does not implement Delivery")


# --- require_budget ---


def test_http_transport_hands_the_prepared_request_to_urllib_under_a_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


# --- Standing ---


def test_standing_rounds_money_and_leaves_an_absent_figure_absent() -> None:
    priced = Standing(keyed=True, credit_usd=99.99680725539, usd_hr=0.285925925926)
    assert priced.credit_usd == pytest.approx(99.9968)
    assert priced.usd_hr == pytest.approx(0.2859)
    assert Standing().credit_usd is None
    assert Standing().usd_hr is None


# --- require_budget ---


def test_require_budget_raises_when_max_usd_is_unset() -> None:
    with pytest.raises(MissionError, match="max-usd"):
        require_budget(Resources())


def test_require_budget_passes_when_max_usd_is_set() -> None:
    require_budget(Resources(max_usd=1.0))


def test_route_keeps_auto_on_the_ssh_family() -> None:
    assert route(HostProfile().kind) == "ssh-family"
