from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch import Dispatcher, GitignoreFilter
from mainboard.dispatch import landing as landing_module
from mainboard.dispatch.landing import Landing, renter
from mainboard.dispatch.rentals import LAUNCH, Rental
from mainboard.dispatch.shared import state_dir
from mainboard.dispatch.snapshots import SOURCES
from mainboard.dispatch.transport import Endpoint
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import Container, HostProfile

from .backends.support import BareBackend
from .support import RecordingMachine, cache, machine_with, plan

# The machine a rental hands over, and the resources every landing below runs under.
_ENDPOINT = Endpoint(address="ssh5.vast.ai", port=41022, user="root", identity="/keys/id")
_ASKED = Resources(max_usd=1.0, walltime="00:30:00", gpus=1)

# A provider host's plan: no root declared, since a rented box is asked where its own workspace
# goes rather than told, and no container, since that is what makes it a landing at all.
_RENTED = {
    "host": "vast",
    "profile": HostProfile(kind="vast", sync={"include": ["src"]}),
}


class FakeRenter:
    """A `Renter` double: hands one rental over, and remembers whether it was ever ended."""

    def __init__(self, fault: Exception | None = None) -> None:
        """fault: what renting raises instead of answering, for a market that turned us away."""
        self.fault = fault
        self.asked: list[tuple[str, str]] = []
        self.cancelled: list[str] = []

    def cancel(self, handle: str) -> None:
        self.cancelled.append(handle)

    def rent(self, execution, resources: Resources) -> Rental:
        self.asked.append((execution.host, resources.walltime or ""))
        if self.fault is not None:
            raise self.fault
        return Rental(handle="4242", endpoint=_ENDPOINT)


def dispatcher_for(workdir: Path) -> Dispatcher:
    """A dispatcher whose mirror only records what it was asked to ship, and where."""
    instance = Dispatcher(cache=cache(), sync=GitignoreFilter(workdir))
    instance.mirrored: list[tuple[str, tuple[str, ...], str]] = []

    def mirror(execution, root: str, **kwargs: object) -> list[str]:
        policy = kwargs.get("ssh")
        extra = kwargs.get("extra", ())
        instance.mirrored.append(
            (root, tuple(extra), policy.destination(execution.host) if policy else execution.host)
        )
        return ["src"]

    instance.rsync_up = mirror
    return instance


def landing(
    workdir: Path, host: RecordingMachine, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> tuple[Landing, FakeRenter, Dispatcher]:
    """A `Landing` onto `host`, its ssh connection stubbed and its mirror recorded."""
    monkeypatch.setattr(landing_module, "connection", lambda where, ssh=None: host)
    backend = FakeRenter(**overrides)
    dispatcher = dispatcher_for(workdir)
    return (
        Landing(dispatcher, backend, plan(**_RENTED), resources=_ASKED),
        backend,
        dispatcher,
    )


def test_a_rental_gets_the_workspace_the_tool_and_the_environment_before_the_job(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order is the failure this exists to end: a bare box cannot run the command yet.

    So the mirror, the tool install and the environment come first, the tree the job runs from is
    pinned next, and only then does the waiting entrypoint get the line that starts the job. The
    mirror goes to the rented machine rather than to the alias, since `vast` is a profile name
    and never an address.
    """
    host = machine_with("/root/projects\n")
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    assert landed.land("python train.py") == "4242"
    assert backend.asked == [("vast", "00:30:00")]
    root, extra, where = dispatcher.mirrored[0]
    assert (root, where) == ("/root/projects", "root@ssh5.vast.ai")
    assert extra[0].startswith(f"{state_dir()}/jobs/job-")
    ordered = [
        next(at for at, line in enumerate(host.lines) if marker in line)
        for marker in ("uv tool install", "mainboard install default", "mb_snap", LAUNCH)
    ]
    assert ordered == sorted(ordered)
    assert host.ran("--profile vast")


def test_the_waiting_entrypoint_is_handed_the_same_staged_line_an_ssh_host_would_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rented box has no queue, and must not: its entrypoint owns the log and the meter.

    What it is handed is the staging every other host gets around the ordinary bash job script,
    so a rented run's receipts, its walltime cap and its source stamp are the ones gold produces.
    """
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    landed.land("python train.py")
    (written,) = host.inputs
    (_, (script,), _) = dispatcher.mirrored[0]
    assert written.startswith(f"cd /root/projects/{SOURCES}/")
    assert written.endswith(f"bash {script}\n")
    assert "export PATH=" in written
    body = (dispatcher.root / script).read_text(encoding="utf-8")
    assert "timeout --kill-after=30s 1800" in body
    assert "bash -c 'python train.py'" in body
    assert "MAINBOARD_RECEIPTS" in body and "mainboard-receipts-begin" in body


def test_a_landing_that_fails_anywhere_ends_the_rental_it_was_holding(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between the create and the launch this process is the only thing holding the handle."""
    host = machine_with("/root/projects\n", rules=[("mb_snap", 1, "no space left on device")])
    landed, backend, _ = landing(workdir, host, monkeypatch)
    with pytest.raises(SystemExit, match="could not pin the source tree"):
        landed.land("python train.py")
    assert backend.cancelled == ["4242"]
    assert LAUNCH not in " ".join(host.lines)


def test_a_market_that_never_rented_anything_leaves_nothing_to_cancel(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal before the create holds no handle, so there is no rental to end."""
    host = machine_with("/root/projects\n")
    landed, backend, _ = landing(workdir, host, monkeypatch, fault=MissionError("no rentable"))
    with pytest.raises(MissionError, match="no rentable"):
        landed.land("python train.py")
    assert backend.cancelled == [] and host.calls == []


def test_a_declared_root_is_honoured_and_a_bare_machine_is_asked_where_to_put_the_workspace(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile that names a root pins the location; one that does not lets the box answer."""
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    landed.plan = plan(host="vast", profile=HostProfile(kind="vast", root="/workspace"))
    landed.land("python train.py")
    assert dispatcher.mirrored[0][0] == "/workspace"
    assert not host.ran("ls -d /work/")


@pytest.mark.parametrize(
    ("container", "landed"),
    [
        pytest.param(None, True, id="a-plan-that-brings-no-image-of-its-own"),
        pytest.param(Container(image="nvcr.io/nvidia/pytorch:25.06-py3"), False, id="prebuilt"),
    ],
)
def test_only_a_plan_without_an_image_of_its_own_is_worth_landing(
    container: Container | None, landed: bool
) -> None:
    """A prebuilt image already holds everything its command needs, so nothing is installed."""
    backend = FakeRenter()
    execution = plan(**_RENTED, container=container)
    assert (renter(backend, execution) is backend) is landed
    assert renter(BareBackend(), plan(**_RENTED)) is None
