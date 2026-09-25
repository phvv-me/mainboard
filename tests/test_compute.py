from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from inspect import signature
from threading import get_ident
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import Board, HostFacts, Survey
from mainboard.compute import Access, reachable, summary
from mainboard.dispatch import HostSetup, HostUnreachable, SshTransport
from mainboard.dispatch.backends import Credentials, ProviderBackend, VastBackend
from mainboard.manifest import HostProfile
from mainboard.manifest.held import Held
from mainboard.probe import GpuFact
from mainboard.probe.system import System

from .dispatch.backends.support import BareBackend, not_found, vast_backend
from .strategies import PATHS, WORDS

if TYPE_CHECKING:
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


def named(paths: Sequence[ComputePath]) -> dict[str, ComputePath]:
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

    def run(self: SshTransport, command: tuple[str, ...], host: str, *, operation: str) -> str:
        assert command[-2:] == ("echo", "mainboard-reachable")
        assert operation == "survey"
        if refusal:
            raise HostUnreachable(refusal)
        return "mainboard-reachable\r\n"

    monkeypatch.setattr(SshTransport, "run", run)
    assert reachable(_GOLD) == refusal
    assert signature(reachable).parameters["ssh"].default.deadline < 30


@pytest.mark.parametrize("reply", ["", "unexpected shell output"])
def test_a_successful_ssh_exit_without_the_marker_is_not_reachability(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    monkeypatch.setattr(SshTransport, "run", lambda self, command, host, operation: reply)
    assert "expected survey marker" in reachable("homelab")


def test_the_first_row_is_this_machine_with_its_own_hardware(board: Board) -> None:
    first = survey(board).paths()[0]
    assert first.name == "local"
    assert first.kind == "local"
    assert first.access is Access.HERE
    assert first.detail.startswith("1x NVIDIA GeForce RTX 4090, 64 GB RAM")
    assert "GPU availability not checked" in first.detail
    assert datetime.fromisoformat(first.observed_at).tzinfo is UTC
    assert first.cached_at == ""


def test_credentials_load_before_any_concurrent_host_probe(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = get_ident()
    loaded: list[int] = []

    def load(self: Credentials) -> tuple[str, ...]:
        loaded.append(get_ident())
        return ()

    def reach(host: str) -> str:
        assert loaded == [owner]
        assert get_ident() != owner
        return ""

    monkeypatch.setattr(Credentials, "load", load)
    survey(board, reach=reach).paths()
    assert loaded == [owner]


@given(
    alias=WORDS,
    profile=st.builds(HostProfile, kind=WORDS),
    setup=_SETUPS,
    refusal=st.sampled_from(["", "gold is down"]),
)
def test_a_host_row_says_only_what_the_probe_and_the_onboarding_record_support(
    board: Board, alias: str, profile: HostProfile, setup: HostSetup | None, refusal: str
) -> None:
    """A live reply and a cached setup never imply current job readiness."""
    row = survey(board, reach=lambda host: refusal).machine(alias, profile, setup)
    assert (row.name, row.kind) == (alias, profile.kind)
    if refusal:
        assert row.access is Access.UNREACHABLE and row.detail.startswith(refusal)
    elif setup is None:
        assert row.access is Access.REACHABLE
        assert f"mainboard setup {alias}" in row.detail
        assert "no cached setup or hardware" in row.detail
    else:
        assert row.access is Access.PROVISIONED
        assert "GB RAM" in row.detail if setup.hardware else setup.env in row.detail
        assert "job readiness and GPU availability not checked" in row.detail
    assert row.cached_at == (setup.onboarded_at if setup else "")


def test_cached_hardware_keeps_its_original_observation_after_a_later_sync(board: Board) -> None:
    setup = HostSetup(
        host="homelab",
        root="C:/projects",
        hardware=facts("RTX 5080"),
        onboarded_at="2026-09-01T01:00:00+00:00",
        synced_at="2026-09-09T23:00:00+00:00",
    )
    row = survey(board).machine("homelab", HostProfile(kind="ssh"), setup)
    assert row.access is Access.PROVISIONED
    assert row.cached_at == setup.onboarded_at
    assert setup.synced_at not in row.detail
    assert "cached" in row.detail and "RTX 5080" in row.detail


@pytest.mark.parametrize("recorded", [False, True])
def test_manifest_status_note_names_a_supported_route_without_claiming_readiness(
    board: Board, *, recorded: bool
) -> None:
    note = "Native managed SSH route only; generic submit is unsupported"
    profile = HostProfile(kind="ssh", vars={"status-note": note})
    setup = HostSetup(host="homelab", root="C:/managed") if recorded else None
    row = survey(board).machine("homelab", profile, setup)
    assert row.access is (Access.PROVISIONED if recorded else Access.REACHABLE)
    assert note in row.detail
    assert "mainboard setup homelab" not in row.detail


@pytest.mark.parametrize("kind", ["pbs", "slurm"])
def test_scheduler_login_probe_does_not_claim_compute_node_availability(
    board: Board, kind: str
) -> None:
    setup = HostSetup(host=_MIYABI_G, root="/work/projects", hardware=facts())
    row = survey(board).machine(_MIYABI_G, HostProfile(kind=kind), setup)
    assert row.access is Access.PROVISIONED
    assert "login endpoint only" in row.detail
    assert "GPU availability not checked" in row.detail
    assert "mainboard jobs" in row.detail


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


def test_every_machine_a_keyed_provider_rents_is_listed_named_by_its_hold(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider's own listing, so a rental from another checkout or a lost record shows."""
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    instances = {
        "instances": [
            {"id": 7, "gpu_name": "RTX 5090", "actual_status": "running", "dph_total": 0.6},
            {"id": 8, "label": "mainboard-lost", "actual_status": "running"},
        ]
    }
    backend = vast_backend({"credit": 1.0}, {"offers": [_OFFER]}, instances)
    deadline = datetime(2026, 9, 25, 18, tzinfo=UTC)
    held = Held(alias="box", provider="vast", handle="7", deadline=deadline, profile=HostProfile())
    rows = named(survey(board).offered(backend, {"box": held}))
    assert list(rows) == ["vast", "box", "vast:8"]
    assert (rows["box"].kind, rows["box"].access, rows["box"].usd_hr) == (
        "rental",
        Access.RENTED,
        0.6,
    )
    assert "1x RTX 5090, running; held until 2026-09-25T18:00:00+00:00" in rows["box"].detail
    assert rows["vast:8"].detail.endswith("not held here, label mainboard-lost")


def test_a_provider_whose_listing_fails_says_so_beside_its_own_row(
    board: Board, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAST_API_KEY", "key-123")
    backend = vast_backend({"credit": 1.0}, {"offers": [_OFFER]}, not_found())
    standing, refused = survey(board).offered(backend, {})
    assert standing.access is Access.KEYED
    assert (refused.kind, refused.access) == ("rental", Access.UNREACHABLE)
    assert "404" in refused.detail


def test_a_machine_row_carries_every_finding_that_is_not_a_pass(board: Board) -> None:
    """The survey judges each census the way `facts` does, and says what is wrong in one cell.

    A host onboarded before censuses were recorded has nothing to say rather than a warning,
    and this machine is judged from its own live facts.
    """
    census = System(system="Windows", arch="AMD64", free_bytes=10**12, root="C:/")
    windows = HostSetup(host="homelab", root="C:/p", hardware=HostFacts(system=census))
    judged = survey(board).machine("homelab", HostProfile(kind="ssh"), windows)
    assert judged.issues.startswith("platform: win-64 is not among the declared platforms")
    older = HostSetup(host="gold", root="/p", hardware=facts("RTX 4090"))
    assert survey(board).machine(_GOLD, HostProfile(kind="ssh"), older).issues == ""
    here = Survey(
        board,
        facts=lambda: HostFacts(memory_total_bytes=10**9, system=census),
        reach=lambda host: "",
        providers=(),
    ).here()
    assert "platform:" in here.issues
