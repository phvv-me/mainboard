from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import Board, MissionError
from mainboard.costs.catalog import Offer
from mainboard.dispatch import HostSetup, vocabulary
from mainboard.dispatch.aliases import SshAliases
from mainboard.dispatch.lease import Lease
from mainboard.dispatch.rentals import Rental
from mainboard.dispatch.transport import Endpoint
from mainboard.holds import Holds, duration_seconds
from mainboard.manifest.held import Holdings
from mainboard.manifest.loading import load
from mainboard.verdicts import Verdicts

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mainboard.context.plan import ExecutionPlan
    from mainboard.dispatch.allocation import Allocation
    from mainboard.dispatch.vocabulary import Resources

# A provider host the fixture manifest does not declare, with the defaults a rental spends.
_PROVIDER = """
[hosts.rent]
kind = "rent"
sync = { include = ["packages/lab-core"] }

[hosts.rent.defaults]
gpu-name = "RTX 5090"
max-usd = 4.0
walltime = "00:30:00"
"""

# The quote a lease-keeping provider accepted before it created the machine.
_OFFER = Offer(provider="rent", gpu="RTX 5090", rate_usd_hr=0.62)


class Renter:
    """A provider that rents one machine answering ssh, keeping a lease the way vast does.

    leases: whether the create records the accepted quote, as vast does and hpc-ai does not.
    refuses: whether renting fails before any machine exists.
    """

    rented: list[Resources] = []
    cancelled: list[str] = []

    def __init__(self, *, leases: bool = True, refuses: bool = False) -> None:
        self.leases = leases
        self.refuses = refuses

    def __call__(self) -> Renter:
        return self

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        del plan
        if self.refuses:
            raise MissionError("no rentable offer right now")
        Renter.rented.append(resources)
        release = datetime.now(UTC) + timedelta(hours=1)
        allocation.begin(lease=Lease(offer=_OFFER, release_by=release) if self.leases else None)
        handle = allocation.created("77")
        endpoint = Endpoint(address="1.2.3.4", port=2222, user="root", identity="/keys/rent")
        return Rental(handle=handle, endpoint=endpoint)

    def cancel(self, handle: str) -> None:
        Renter.cancelled.append(handle)


class Remote:
    """The far side of the one connection a hold opens to park its machine."""

    handed: list[str] = []

    def __init__(self, retcode: int) -> None:
        self.retcode = retcode

    def __getitem__(self, argv: str | tuple[str, ...]) -> Remote:
        return self

    def __lshift__(self, script: str) -> Remote:
        Remote.handed.append(script)
        return self

    def run(self, *, retcode: None) -> tuple[int, str, str]:
        return self.retcode, "", "no such file"


@pytest.fixture
def rental(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Holds]:
    """Holds over a workspace declaring a `rent` provider, every machine behind it a double.

    The provider rents through `Renter`, onboarding answers with a root, parking hands its
    script to `Remote`, and the ssh config is a file under the workspace.
    """
    manifest = workspace / "mainboard.toml"
    manifest.write_text(manifest.read_text() + _PROVIDER)
    Renter.rented, Renter.cancelled, Remote.handed = [], [], []
    monkeypatch.setattr("mainboard.holds.route", lambda kind: Renter())
    monkeypatch.setattr("mainboard.holds.Provisioner", Unvouched)
    monkeypatch.setattr(Board, "install", onboarded)
    monkeypatch.setattr("mainboard.holds.connection", parked(0))
    monkeypatch.setattr(Verdicts, "cancel", settled)
    yield Holds(Board(workspace), aliases=SshAliases(workspace / "ssh_config"))


class Unvouched:
    """A provisioner whose lock always vouches for the manifest."""

    def __init__(self, root: Path, manifest: object) -> None:
        del root, manifest

    def compiler_for(self, env: str) -> Unvouched:
        del env
        return self

    def vouch(self) -> None:
        return


def onboarded(self: Board, env: str = "", **options: object) -> HostSetup:
    """Onboarding answering with where it put the workspace, refusing a host it cannot resolve."""
    del options
    assert self.plan().profile.kind == "ssh"
    return HostSetup(host=self.host, root="/root/projects", env=env or "default")


def parked(retcode: int):
    """The connection a hold parks its machine over, answering `retcode`."""

    @contextmanager
    def connection(host: str, ssh: object) -> Iterator[Remote]:
        del host, ssh
        yield Remote(retcode)

    return connection


def settled(self: Verdicts, handle: str, *, host: str = "") -> None:
    """Cancel as the settle path does: the record ends, the rental with it."""
    cache = self.board.dispatcher.cache
    cache.resolve(cache.run(handle, host), vocabulary.CANCELLED, None, vocabulary.CANCELLED)
    Renter.cancelled.append(handle)


def test_a_hold_rents_names_sets_up_and_parks_a_machine_that_answers_as_a_host(
    rental: Holds,
) -> None:
    """Every verb reaches the alias the way it reaches `gold`, and the sweep knows its deadline."""
    before = datetime.now(UTC)
    held = rental.hold("rent", duration="3h")
    assert (held.alias, held.handle, held.gpu, held.usd_hr) == (
        "rent-rtx-5090",
        "77",
        "RTX 5090",
        0.62,
    )
    assert held.profile.root == "/root/projects"
    assert before + timedelta(hours=3) <= held.deadline <= datetime.now(UTC) + timedelta(hours=3)
    [resources] = Renter.rented
    assert (resources.walltime, resources.gpu_name, resources.max_usd) == (
        "03:00:00",
        "RTX 5090",
        4.0,
    )
    assert Remote.handed == ["exec sleep infinity\n"]
    profile = load(rental.board.root / "mainboard.toml").profile(held.alias)
    assert (profile.kind, profile.platform, profile.root) == ("ssh", "linux-64", "/root/projects")
    assert profile.sync.include == ["packages/lab-core"]
    assert profile.defaults.walltime == "03:00:00"
    config = rental.aliases.path.read_text()
    assert "Host rent-rtx-5090\n  HostName 1.2.3.4\n  Port 2222\n  User root" in config
    record = rental.board.dispatcher.cache.run("77", "rent")
    assert record.lease is not None and record.lease.release_by == held.deadline
    assert (record.name, record.evidence) == ("hold-rent-rtx-5090", "pending")


def test_a_provider_that_keeps_no_quote_still_leaves_the_sweep_a_deadline(
    rental: Holds, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mainboard.holds.route", lambda kind: Renter(leases=False))
    held = rental.hold("rent", duration="90m", alias="box", gpu_name="L40S", max_usd=9.0)
    assert (held.alias, held.gpu, held.usd_hr) == ("box", "", None)
    lease = rental.board.dispatcher.cache.run("77", "rent").lease
    assert lease is not None and (lease.offer.rate_usd_hr, lease.release_by) == (
        0.0,
        held.deadline,
    )


@pytest.mark.parametrize(
    "fault",
    [RuntimeError("onboarding died"), None],
    ids=["onboarding fails", "the parked script is refused"],
)
def test_a_hold_that_fails_after_renting_ends_the_rental_and_forgets_the_alias(
    rental: Holds, monkeypatch: pytest.MonkeyPatch, fault: Exception | None
) -> None:
    if fault is not None:

        def broken(self: Board, env: str = "", **options: object) -> HostSetup:
            raise fault

        monkeypatch.setattr(Board, "install", broken)
    else:
        monkeypatch.setattr("mainboard.holds.connection", parked(1))
    with pytest.raises(
        RuntimeError if fault else MissionError, match="onboarding died|could not park"
    ):
        rental.hold("rent", duration="1h")
    assert Renter.cancelled == ["77"]
    assert Holdings(rental.board.root).read() == {}
    assert "rent-rtx-5090" not in rental.aliases.path.read_text()


def test_a_rental_that_never_happened_leaves_nothing_to_release(
    rental: Holds, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("mainboard.holds.route", lambda kind: Renter(refuses=True))
    with pytest.raises(MissionError, match="no rentable offer"):
        rental.hold("rent", duration="1h")
    [record] = rental.board.dispatcher.cache.recent()
    assert record.verdict == vocabulary.FAILED
    assert Renter.cancelled == []


@pytest.mark.parametrize(
    ("provider", "alias", "route", "refusal"),
    [
        ("rent", "gold", None, "already names a host"),
        ("gold", "", "ssh-family", "only a provider that rents an ssh machine"),
        ("rent", "", object, "only a provider that rents an ssh machine"),
    ],
    ids=["a declared host's name", "an owned host", "a provider that hands out no machine"],
)
def test_a_hold_refuses_what_it_cannot_rent_or_name(
    rental: Holds,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    alias: str,
    route: object,
    refusal: str,
) -> None:
    if route is not None:
        monkeypatch.setattr("mainboard.holds.route", lambda kind: route)
    with pytest.raises(MissionError, match=refusal):
        rental.hold(provider, duration="1h", alias=alias)
    assert Renter.rented == []


def test_release_ends_the_rental_and_forgets_everything_the_hold_wrote(rental: Holds) -> None:
    held = rental.hold("rent", duration="2h")
    assert rental.release(held.alias) == held
    assert Renter.cancelled == ["77"]
    assert Holdings(rental.board.root).read() == {}
    assert held.alias not in rental.aliases.path.read_text()
    assert held.alias not in load(rental.board.root / "mainboard.toml").hosts
    with pytest.raises(MissionError, match=r"'ghost' is not held; held machines are \[\]"):
        rental.release("ghost")


def test_expiry_releases_the_due_the_already_ended_and_the_unrecorded_and_keeps_the_rest(
    rental: Holds,
) -> None:
    live = rental.hold("rent", duration="2h", alias="live")
    holdings = Holdings(rental.board.root)
    past = datetime.now(UTC) - timedelta(minutes=1)
    cache = rental.board.dispatcher.cache
    recorded = cache.run("77", "rent")
    for alias, handle, verdict in (("due", "79", None), ("ended", "78", vocabulary.OK)):
        cache.record(recorded.model_copy(update={"handle": handle, "verdict": verdict}))
        holdings.save(live.model_copy(update={"alias": alias, "handle": handle}))
    holdings.save(live.model_copy(update={"alias": "due", "handle": "79", "deadline": past}))
    holdings.save(live.model_copy(update={"alias": "gone", "handle": "nowhere"}))
    released = sorted(held.alias for held in rental.expire())
    assert released == ["due", "ended", "gone"]
    assert list(holdings.read()) == ["live"]
    assert sorted(Renter.cancelled) == ["78", "79", "nowhere"]


@given(st.integers(0, 99), st.integers(0, 59))
def test_a_duration_reads_the_way_a_person_writes_it(hours: int, minutes: int) -> None:
    seconds = hours * 3600 + minutes * 60
    assert duration_seconds(f"{hours:02d}:{minutes:02d}:00") == seconds
    if hours and minutes:
        assert duration_seconds(f"{hours}h{minutes}m") == seconds
    if hours:
        assert duration_seconds(f"{hours}h") == hours * 3600
    if minutes:
        assert duration_seconds(f" {minutes}m ") == minutes * 60


@pytest.mark.parametrize("spelled", ["", "h", "3 hours", "1:2", "a:b:c"])
def test_a_duration_nobody_writes_is_refused(spelled: str) -> None:
    with pytest.raises(MissionError, match="a hold lasts"):
        duration_seconds(spelled)


def test_a_release_the_provider_refuses_stays_held_for_the_next_pass(
    rental: Holds, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    held = rental.hold("rent", duration="1h")
    past = datetime.now(UTC) - timedelta(minutes=1)
    rental.holdings.save(held.model_copy(update={"deadline": past}))

    def refused(self: Verdicts, handle: str, *, host: str = "") -> None:
        raise MissionError("vast did not confirm destruction")

    monkeypatch.setattr(Verdicts, "cancel", refused)
    assert rental.expire() == []
    assert list(rental.holdings.read()) == [held.alias]
    assert "could not release rent-rtx-5090 yet" in caplog.text
