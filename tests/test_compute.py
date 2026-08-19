from inspect import signature
from typing import TYPE_CHECKING

import pytest

from mainboard import Board, HostFacts, Survey
from mainboard.compute import Access, reachable, summary
from mainboard.dispatch import HostSetup
from mainboard.dispatch.backends import ProviderBackend, VastBackend
from mainboard.dispatch.transport import HostUnreachable, SshTransport
from mainboard.probe.snapshot import GpuFact

from .dispatch.backends.conftest import BareBackend, not_found, vast_backend

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mainboard.compute import ComputePath

_GOLD = "gold"
_MIYABI_G = "miyabi-g"
_OFFER = {
    "id": 11,
    "dph_total": 0.31,
    "min_bid": 0.09,
    "gpu_name": "RTX 4090",
    "num_gpus": 1,
    "geolocation": "Texas, US",
}


def facts(*gpus: str, memory_gb: int = 64) -> HostFacts:
    """A hardware snapshot naming `gpus` and `memory_gb`, the shape a survey summarizes."""
    return HostFacts(
        hostname="box",
        memory_total_bytes=memory_gb * 10**9,
        gpus=tuple(GpuFact(name=name, memory_total_bytes=24 * 10**9) for name in gpus),
    )


@pytest.fixture
def board(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Board:
    """A board over the fixture workspace, its dispatch state kept inside that directory."""
    monkeypatch.chdir(workspace)
    return Board(workspace)


def survey(
    board: Board,
    *,
    reach: Callable[[str], str] = lambda host: "",
    providers: tuple[ProviderBackend, ...] = (),
) -> Survey:
    """A survey whose every network touch is injected, this machine's facts included."""
    return Survey(
        board, facts=lambda: facts("NVIDIA GeForce RTX 4090"), reach=reach, providers=providers
    )


def named(paths: list[ComputePath]) -> dict[str, ComputePath]:
    """The surveyed rows keyed by name, for a test asserting about one of them."""
    return {path.name: path for path in paths}


# --- summary ---


def test_summary_counts_identical_gpus_and_always_names_the_memory() -> None:
    assert summary(facts("H100", "H100", memory_gb=512)) == "2x H100, 512 GB RAM"
    assert summary(facts(memory_gb=16)) == "16 GB RAM"


# --- reachable ---


def test_reachable_is_empty_when_one_bounded_round_trip_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SshTransport, "warm", lambda self, host: None)
    assert not reachable(_GOLD)


def test_reachable_answers_with_the_refusal_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(self: SshTransport, host: str) -> None:
        raise HostUnreachable(f"ssh connect to {host!r} timed out")

    monkeypatch.setattr(SshTransport, "warm", refuse)
    assert reachable(_GOLD) == "ssh connect to 'gold' timed out"


def test_reachable_probes_under_a_bounded_default_policy() -> None:
    policy = signature(reachable).parameters["ssh"].default
    assert policy.deadline < 30


# --- the local row ---


def test_the_first_row_is_this_machine_with_its_own_hardware(board: Board) -> None:
    first = survey(board).paths()[0]
    assert first.name == "local"
    assert first.kind == "local"
    assert first.access is Access.HERE
    assert first.detail == "1x NVIDIA GeForce RTX 4090, 64 GB RAM"


# --- host rows ---


def test_a_host_that_will_not_answer_is_a_row_state_carrying_why(board: Board) -> None:
    paths = named(survey(board, reach=lambda host: f"{host} is down").paths())
    assert paths[_MIYABI_G].access is Access.UNREACHABLE
    assert paths[_MIYABI_G].kind == "pbs"
    assert paths[_MIYABI_G].detail == "miyabi-g is down"


def test_a_host_that_answers_but_was_never_set_up_is_reachable_not_ready(board: Board) -> None:
    paths = named(survey(board).paths())
    assert paths[_GOLD].access is Access.REACHABLE
    assert paths[_GOLD].kind == "ssh"
    assert paths[_GOLD].detail == "never set up"


def test_an_onboarded_host_reports_the_hardware_onboarding_recorded(board: Board) -> None:
    board.dispatcher.cache.save_host(
        HostSetup(host=_GOLD, root="/repo", hardware=facts("NVIDIA GB10", memory_gb=119))
    )
    paths = named(survey(board).paths())
    assert paths[_GOLD].access is Access.READY
    assert paths[_GOLD].detail == "1x NVIDIA GB10, 119 GB RAM"


def test_an_onboarded_host_with_no_recorded_hardware_says_so(board: Board) -> None:
    board.dispatcher.cache.save_host(HostSetup(host=_GOLD, root="/repo", env="serving"))
    assert named(survey(board).paths())[_GOLD].detail == "serving, hardware unrecorded"


def test_a_host_whose_kind_routes_to_a_provider_is_left_to_that_provider_s_row(
    board: Board,
) -> None:
    """A rented machine is listed once, by its provider, never probed as if it were an ssh box."""
    hosts = {
        **board.manifest.hosts,
        "rented": board.manifest.profile(_GOLD).model_copy(update={"kind": "vast"}),
    }
    board.shared["manifest"] = board.manifest.model_copy(update={"hosts": hosts})
    assert "rented" not in named(survey(board).paths())


# --- provider rows ---


def test_a_provider_with_a_key_carries_its_credit_and_a_live_rate(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    backend = vast_backend({"credit": 42.5}, {"offers": [_OFFER]})
    row = named(survey(board, providers=(backend,)).paths()).get("vast")
    assert row is not None
    assert row.kind == "provider"
    assert row.access is Access.KEYED
    assert row.credit_usd == pytest.approx(42.5)
    assert row.usd_hr == pytest.approx(0.31)
    assert row.detail == "1x RTX 4090 Texas, US"


def test_a_provider_with_no_key_is_a_row_naming_the_variable_to_set(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    row = named(survey(board, providers=(VastBackend(),)).paths())["vast"]
    assert row.access is Access.UNKEYED
    assert row.credit_usd is None
    assert "VAST_API_KEY" in row.detail


def test_a_provider_answering_for_no_account_is_listed_with_what_it_lacks(board: Board) -> None:
    """A survey stays a listing, so a backend with no account notion is a row, not a raise."""
    row = named(survey(board, providers=(BareBackend(),)).paths())["bare"]
    assert row.access is Access.UNKEYED
    assert row.detail == "the bare backend does not implement Account"


def test_a_provider_that_will_not_answer_is_a_row_state_not_a_failure(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    row = named(survey(board, providers=(vast_backend(not_found()),)).paths())["vast"]
    assert row.access is Access.UNREACHABLE
    assert "404" in row.detail


# --- the whole survey ---


def test_every_declared_host_and_provider_is_surveyed_this_machine_first(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    backend = vast_backend({"credit": 1.0}, {"offers": []})
    paths = survey(board, providers=(backend,)).paths()
    assert [path.name for path in paths] == ["local", _GOLD, _MIYABI_G, "vast"]


def test_the_default_roster_is_every_registered_provider_backend(board: Board) -> None:
    names = {backend.name for backend in Survey(board).providers}
    assert {"modal", "vast", "hpc-ai"} <= names
    assert names == set(ProviderBackend.names())
