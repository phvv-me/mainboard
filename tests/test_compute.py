from inspect import signature
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import Board, HostFacts, Survey
from mainboard.compute import Access, reachable, summary
from mainboard.dispatch import HostSetup, HostUnreachable, SshTransport
from mainboard.dispatch.backends import ProviderBackend, VastBackend
from mainboard.manifest import HostProfile
from mainboard.probe import GpuFact

from .dispatch.backends.conftest import BareBackend, not_found, vast_backend
from .strategies import PATHS, WORDS

if TYPE_CHECKING:
    from collections.abc import Callable

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

# An onboarding record as the survey reads one, either carrying the hardware the probe found or
# carrying nothing but the environment it installed.
_SETUPS = st.none() | st.builds(
    HostSetup,
    host=WORDS,
    root=PATHS,
    env=WORDS,
    hardware=st.none() | st.builds(HostFacts, memory_total_bytes=st.integers(0, 10**12)),
)


def facts(*gpus: str, memory_gb: int = 64) -> HostFacts:
    """A hardware snapshot naming `gpus` and `memory_gb`, the shape a survey summarizes."""
    return HostFacts(
        hostname="box",
        memory_total_bytes=memory_gb * 10**9,
        gpus=tuple(GpuFact(name=name, memory_total_bytes=24 * 10**9) for name in gpus),
    )


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


@pytest.mark.parametrize(
    ("gpus", "memory_gb", "line"),
    [
        (("H100", "H100"), 512, "2x H100, 512 GB RAM"),
        ((), 16, "16 GB RAM"),
    ],
    ids=["identical gpus counted once", "a machine with no gpu at all"],
)
def test_summary_counts_identical_gpus_and_always_names_the_memory(
    gpus: tuple[str, ...], memory_gb: int, line: str
) -> None:
    assert summary(facts(*gpus, memory_gb=memory_gb)) == line


@pytest.mark.parametrize(
    "refusal", ["", "ssh connect to 'gold' timed out"], ids=["one round trip lands", "it refuses"]
)
def test_reachable_answers_with_the_refusal_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    """The probe is bounded and its failure is an answer, since a survey stays a listing."""

    def warm(self: SshTransport, host: str) -> None:
        if refusal:
            raise HostUnreachable(refusal)

    monkeypatch.setattr(SshTransport, "warm", warm)
    assert reachable(_GOLD) == refusal
    assert signature(reachable).parameters["ssh"].default.deadline < 30


def test_the_first_row_is_this_machine_with_its_own_hardware(board: Board) -> None:
    first = survey(board).paths()[0]
    assert first.name == "local"
    assert first.kind == "local"
    assert first.access is Access.HERE
    assert first.detail == "1x NVIDIA GeForce RTX 4090, 64 GB RAM"


@given(
    alias=WORDS,
    profile=st.builds(HostProfile, kind=WORDS),
    setup=_SETUPS,
    refusal=st.sampled_from(["", "gold is down"]),
)
def test_a_host_row_says_only_what_the_probe_and_the_onboarding_record_support(
    board: Board, alias: str, profile: HostProfile, setup: HostSetup | None, refusal: str
) -> None:
    """Three states and one rule each: a host that will not answer is unreachable whatever was
    recorded of it, one that answers without a record is reachable rather than ready, and a ready
    row describes real hardware without a second round trip.
    """
    row = survey(board, reach=lambda host: refusal).machine(alias, profile, setup)
    assert (row.name, row.kind) == (alias, profile.kind)
    if refusal:
        assert row.access is Access.UNREACHABLE and row.detail == refusal
    elif setup is None:
        assert row.access is Access.REACHABLE and row.detail == "never set up"
    else:
        assert row.access is Access.READY
        assert row.detail.endswith("GB RAM") if setup.hardware else setup.env in row.detail


def test_a_provider_with_a_key_carries_its_credit_and_a_live_rate(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    backend = vast_backend({"credit": 42.5}, {"offers": [_OFFER]})
    row = named(survey(board, providers=(backend,)).paths())["vast"]
    assert row.kind == "provider"
    assert row.access is Access.KEYED
    assert row.credit_usd == pytest.approx(42.5)
    assert row.usd_hr == pytest.approx(0.31)
    assert row.detail == "1x RTX 4090 Texas, US"


@pytest.mark.parametrize(
    ("keyed", "backend", "name", "access", "fragment"),
    [
        (False, VastBackend, "vast", Access.UNKEYED, "VAST_API_KEY"),
        (False, BareBackend, "bare", Access.UNKEYED, "does not implement Account"),
        (True, None, "vast", Access.UNREACHABLE, "404"),
    ],
    ids=[
        "a provider with no key names the variable to set",
        "a provider answering for no account is listed with what it lacks",
        "a provider that will not answer is a row state, not a failure",
    ],
)
def test_a_provider_row_without_a_price_still_says_what_stands_in_the_way(
    board: Board,
    monkeypatch: pytest.MonkeyPatch,
    keyed: bool,
    backend: type[ProviderBackend] | None,
    name: str,
    access: Access,
    fragment: str,
) -> None:
    """A survey stays a listing, so nothing a provider refuses ever costs the rest of the rows."""
    if keyed:
        monkeypatch.setenv("VAST_API_KEY", "key-123")
    else:
        monkeypatch.delenv("VAST_API_KEY", raising=False)
        monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    asked = backend() if backend is not None else vast_backend(not_found())
    row = named(survey(board, providers=(asked,)).paths())[name]
    assert row.access is access
    assert row.credit_usd is None
    assert fragment in row.detail


def test_every_declared_host_and_provider_is_surveyed_this_machine_first(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rented machine is listed once, by its provider, never probed as if it were an ssh box."""
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    hosts = {
        **board.manifest.hosts,
        "rented": board.manifest.profile(_GOLD).model_copy(update={"kind": "vast"}),
    }
    board.shared["manifest"] = board.manifest.model_copy(update={"hosts": hosts})
    backend = vast_backend({"credit": 1.0}, {"offers": []})
    paths = survey(board, providers=(backend,)).paths()
    assert [path.name for path in paths] == ["local", _GOLD, _MIYABI_G, "vast"]


def test_the_default_roster_is_every_registered_provider_backend(board: Board) -> None:
    names = {backend.name for backend in Survey(board).providers}
    assert {"modal", "vast", "hpc-ai"} <= names
    assert names == set(ProviderBackend.names())
