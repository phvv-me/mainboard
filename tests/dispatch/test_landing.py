from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch import Dispatcher, GitignoreFilter, Shipment
from mainboard.dispatch import landing as landing_module
from mainboard.dispatch.agent import AgentRefused
from mainboard.dispatch.allocation import Allocation
from mainboard.dispatch.landing import Landing, renter
from mainboard.dispatch.provenance import Source
from mainboard.dispatch.rentals import LAUNCH, Rental, handoff
from mainboard.dispatch.shared import state_dir
from mainboard.dispatch.snapshots import SOURCES, Snapshots
from mainboard.dispatch.state import Cache
from mainboard.dispatch.transport import Endpoint
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import Container, HostProfile

from .backends.support import BareBackend
from .support import RecordingAgent, RecordingMachine, machine_with, plan, recorded

# The machine a rental hands over, and the resources every landing below runs under.
_ENDPOINT = Endpoint(address="ssh5.vast.ai", port=41022, user="root", identity="/keys/id")
_ASKED = Resources(max_usd=1.0, walltime="00:30:00", gpus=1)


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_a_rental_budget_must_be_finite_and_nonnegative(amount: float) -> None:
    with pytest.raises(ValueError):
        Resources(max_usd=amount)


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

    def rent(self, execution, resources: Resources, *, allocation: Allocation) -> Rental:
        self.asked.append((execution.host, resources.walltime or ""))
        if self.fault is not None:
            raise self.fault
        allocation.begin()
        allocation.created("4242")
        return Rental(handle="4242", endpoint=_ENDPOINT)


def shipped(dispatcher: Dispatcher, command: str) -> Shipment:
    """`command` as a board ships it to a landing: the mirror, under the tree's provenance."""
    return Shipment.of_command(command, source=dispatcher.source(command), imports=())


class PinningAgent(RecordingAgent):
    """A rental's agent double that also marks, in the machine's own log, when it was asked."""

    def __init__(self, machine: RecordingMachine) -> None:
        super().__init__()
        self.machine = machine

    def ask(self, request, *, payload=None):
        self.machine.calls.append(["agent", "mainboard-agent-pin"])
        return super().ask(request, payload=payload)


def dispatcher_for(workdir: Path) -> Dispatcher:
    """A dispatcher whose mirror only records what it was asked to ship, and where."""
    instance = Dispatcher(cache=Cache(workdir / "dispatch.sqlite"), sync=GitignoreFilter(workdir))
    instance.mirrored: list[tuple[str, tuple[str, ...], str]] = []

    def mirror(execution, root: str, **kwargs: object) -> list[str]:
        policy = kwargs.get("ssh")
        extra = kwargs.get("extra", ())
        instance.mirrored.append(
            (root, tuple(extra), policy.destination(execution.host) if policy else execution.host)
        )
        return ["src"]

    instance.mirror = mirror
    instance.reached: list[str] = []
    return instance


def landing(
    workdir: Path, host: RecordingMachine, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> tuple[Landing, FakeRenter, Dispatcher]:
    """A `Landing` onto `host`, its ssh connection stubbed and its mirror recorded."""
    monkeypatch.setattr(landing_module, "connection", lambda where, ssh=None: host)
    backend = FakeRenter(**overrides)
    dispatcher = dispatcher_for(workdir)
    dispatcher.pins = PinningAgent(host)

    def agent(execution, ssh=None) -> PinningAgent:
        dispatcher.reached.append(ssh.destination(execution.host))
        return dispatcher.pins

    dispatcher.agent = agent
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
    assert landed.land(shipped(dispatcher, "python train.py")).id == "4242"
    assert backend.asked == [("vast", "00:30:00")]
    root, extra, where = dispatcher.mirrored[0]
    assert (root, where) == ("/root/projects", "root@ssh5.vast.ai")
    assert dispatcher.reached == ["root@ssh5.vast.ai"]
    assert extra[0].startswith(f"{state_dir()}/jobs/job-")
    ordered = [
        next(at for at, line in enumerate(host.lines) if marker in line)
        for marker in (
            "uv tool install",
            "mainboard install default",
            "mainboard-agent-pin",
            LAUNCH,
        )
    ]
    assert ordered == sorted(ordered)
    assert host.ran("--profile vast")


def test_a_machine_that_ships_no_python_is_given_one_before_the_mirror_is_attempted(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror's far end is the machine's own Python, which a bare image may lack.

    Both lines run on the bare connection, since the workspace they would otherwise `cd` into is
    what the mirror underneath them is about to create.
    """
    host = machine_with("/root/projects\n", rules=[("python3 -c pass", 1, "")])
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    landed.land(shipped(dispatcher, "python train.py"))
    assert "python3 -c pass" in host.lines
    assert host.ran("apt-get install -y -qq python3")
    equipped = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, equipped, monkeypatch)
    landed.land(shipped(dispatcher, "python train.py"))
    assert not equipped.ran("apt-get")


def test_the_waiting_entrypoint_is_handed_the_same_staged_line_an_ssh_host_would_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rented box has no queue, and must not: its entrypoint owns the log and the meter.

    What it is handed is the staging every other host gets around the ordinary job script, so a
    rented run's receipts, its walltime cap and its source stamp are the ones gold produces.

    The script is named by the absolute path the mirror carried it to, and the assertion below
    is that those are the same path: a launch naming anything the transfer did not deliver is the
    `No such file or directory` a landed rental answered with once already.
    """
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    landed.land(shipped(dispatcher, "python train.py"))
    (written,) = host.inputs
    (root, (script,), _) = dispatcher.mirrored[0]
    assert written.startswith(f"cd {root}/{SOURCES}/")
    snapshot = written.removeprefix("cd ").split(" && ", maxsplit=1)[0]
    assert written.endswith(f"sh {snapshot}/.mainboard-jobs/{Path(script).name}\n")
    assert "export PATH=" in written
    job = recorded((dispatcher.root / script).read_text(encoding="utf-8"))
    assert (job.command, job.walltime, job.logs) == ("python train.py", "00:30:00", "")


def test_a_rented_job_carries_the_complete_source_seal(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    source = Source(identity="abc1234", key="abc1234-5678", commit="a" * 40, digest="b" * 64)
    shipment = Shipment.of_command("python train.py", source=source, imports=())
    script = landed.script(shipment, root="/root/projects", listing="")
    job = recorded((dispatcher.root / script).read_text(encoding="utf-8"))
    assert job.variables["MAINBOARD_SOURCE_COMMIT"] == source.commit
    assert job.variables["MAINBOARD_SOURCE_DIGEST"] == source.digest


def test_rental_results_link_to_the_live_root_that_fetch_reads(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sealed snapshot must not strand results on a disk that release destroys."""
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    shipment = shipped(dispatcher, "python train.py").model_copy(
        update={"fetch": "research/project/datasets/node"}
    )
    landed.land(shipment)
    [request] = dispatcher.pins.requests
    assert request["pin"]["results"] == "research/project/datasets/node"


def test_the_job_is_pointed_at_the_tree_the_pin_actually_created(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dirty tree's snapshot key digests its own delta, and a landing takes tens of minutes.

    Reading that key twice across one landing can answer twice differently, and a job rendered
    against the first answer while its tree is pinned under the second is a job standing in a
    directory nobody created: it activates from a path that does not exist and says it found no
    environment there (vast 49867368, 2026-09-04). The tree is read once, so the launch and the
    script it runs name the same snapshot however much the workspace moves underneath.
    """
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    landed.land(shipped(dispatcher, "python train.py"))
    (written,) = host.inputs
    (_, (script,), _) = dispatcher.mirrored[0]
    snapshot = written.removeprefix("cd ").split(" && ", maxsplit=1)[0]
    assert "/sources/sha256-" in snapshot
    assert snapshot in (dispatcher.root / script).read_text(encoding="utf-8")
    [request] = dispatcher.pins.requests
    assert Snapshots(request["pin"]["root"]).path(request["pin"]["key"]) == snapshot


def test_a_pinned_tree_the_job_could_not_activate_from_ends_the_rental(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last cheap moment: the machine is ours and the entrypoint has been handed nothing.

    A tree the job cannot activate from is caught here rather than paid for in full and answered
    with the job's own activation refusal.
    """
    host = machine_with(
        "/root/projects\n", rules=[("fi && true", 1, "found no default environment")]
    )
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    with pytest.raises(MissionError, match="pinned tree on the rental cannot run a command"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert backend.cancelled == ["4242"]
    assert host.inputs == []


def test_a_landing_that_fails_anywhere_ends_the_rental_it_was_holding(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed rental remains tracked independently of this process's cleanup attempt."""
    host = machine_with("/root/projects\n")
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    dispatcher.pins.answer = AgentRefused("no space left on device")
    with pytest.raises(SystemExit, match="could not pin the source tree"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert backend.cancelled == ["4242"]
    record = dispatcher.cache.run("4242", "vast")
    assert record.script == "python train.py"
    assert record.evidence == "not_started" and record.verdict == "failed"
    assert LAUNCH not in " ".join(host.lines)


def test_a_machine_that_cannot_be_given_python_is_ended_before_any_mirror(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without Python nothing can reach the box, so the rental is released rather than billed."""
    host = machine_with(
        "/root/projects\n",
        rules=[("python3 -c pass", 1, ""), ("apt-get", 100, "E: Unable to locate package")],
    )
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    with pytest.raises(MissionError, match="no python and could not install one: E: Unable"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert dispatcher.mirrored == []
    assert backend.cancelled == ["4242"]


def test_a_rental_the_registry_never_recorded_is_still_ended_when_its_landing_fails(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With no row to settle, ending the rental is all that stands between it and a bill."""
    host = machine_with("/root/projects\n")
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    dispatcher.pins.answer = AgentRefused("disk full")
    monkeypatch.setattr(
        backend,
        "rent",
        lambda execution, resources, *, allocation: Rental(handle="4242", endpoint=_ENDPOINT),
    )
    with (
        caplog.at_level("WARNING", logger="mainboard.dispatch"),
        pytest.raises(SystemExit, match="could not pin the source tree"),
    ):
        landed.land(shipped(dispatcher, "python train.py"))
    assert backend.cancelled == ["4242"]
    assert any("failed before provisioning" in message for message in caplog.messages)


def test_a_launch_the_entrypoint_refused_names_why_and_keeps_the_rental_tracked(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line may have reached the box before the refusal, so the monitor settles it."""
    host = machine_with("/root/projects\n", rules=[(handoff(), 1, "disk quota exceeded")])
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    with pytest.raises(MissionError, match="could not start the job on the rental: disk quota"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert backend.cancelled == []
    assert dispatcher.cache.run("4242", "vast") in dispatcher.cache.tracked()


def test_a_rental_is_durably_registered_before_provisioning(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    shipment = shipped(dispatcher, "python train.py").model_copy(update={"fetch": "out/run"})

    def inspect(rental: Rental, *, shipment: Shipment) -> None:
        record = dispatcher.cache.run(rental.handle, "vast")
        assert record.name == "early" and record.node == "carry"
        assert record.fetch_path == shipment.fetch
        assert record.commit == shipment.source.commit
        assert record.evidence == "not_started"
        assert record in dispatcher.cache.tracked()

    monkeypatch.setattr(landed, "equip", inspect)
    handle = landed.land(shipment, name="early", node="carry")
    assert handle.id == "4242"
    assert dispatcher.cache.total() == 1


def test_a_lost_launch_reply_retains_the_rental_and_pending_evidence(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = machine_with("/root/projects\n")
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)

    def interrupted(remote: RecordingMachine, *, pinned: str, script: str) -> None:
        assert dispatcher.cache.run("4242", "vast").evidence == "pending"
        raise MissionError("launch reply lost")

    monkeypatch.setattr(landed, "start", interrupted)
    with pytest.raises(MissionError, match="launch reply lost"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert backend.cancelled == []
    assert dispatcher.cache.run("4242", "vast").verdict == "queued"
    assert dispatcher.cache.total() == len(dispatcher.cache.tracked()) == 1


def test_a_cancelled_provisioning_job_cannot_start_after_cancellation(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)

    def cancelled(remote: RecordingMachine, pinned: str) -> None:
        record = dispatcher.cache.run("4242", "vast")
        stopped = dispatcher.cache.resolve(record, "cancelled", None, "cancelled")
        dispatcher.cache.report(stopped, "cancelled")

    monkeypatch.setattr(landed, "verify", cancelled)
    monkeypatch.setattr(landed, "start", lambda *args, **kwargs: pytest.fail("late launch"))
    with pytest.raises(MissionError, match="ended during setup"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert dispatcher.cache.run("4242", "vast").verdict == "cancelled"
    assert dispatcher.cache.tracked() == []


def test_failed_setup_cleanup_keeps_its_handle_for_a_later_monitor(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = machine_with("/root/projects\n")
    landed, backend, dispatcher = landing(workdir, host, monkeypatch)
    dispatcher.pins.answer = AgentRefused("disk full")

    def unavailable(handle: str) -> None:
        raise MissionError("provider cleanup unavailable")

    monkeypatch.setattr(backend, "cancel", unavailable)
    with pytest.raises(MissionError, match="provider cleanup unavailable"):
        landed.land(shipped(dispatcher, "python train.py"))
    record = dispatcher.cache.run("4242", "vast")
    assert record.evidence == "not_started" and record.verdict == "failed"
    assert record.reported is None and record in dispatcher.cache.tracked()


def test_a_market_that_never_rented_anything_leaves_nothing_to_cancel(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal before the create holds no handle, so there is no rental to end."""
    host = machine_with("/root/projects\n")
    landed, backend, dispatcher = landing(
        workdir, host, monkeypatch, fault=MissionError("no rentable")
    )
    with pytest.raises(MissionError, match="no rentable"):
        landed.land(shipped(dispatcher, "python train.py"))
    assert backend.cancelled == [] and host.calls == []


def test_a_declared_root_is_honoured_and_a_bare_machine_is_asked_where_to_put_the_workspace(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile that names a root pins the location; one that does not lets the box answer."""
    host = machine_with("/root/projects\n")
    landed, _, dispatcher = landing(workdir, host, monkeypatch)
    landed.plan = plan(host="vast", profile=HostProfile(kind="vast", root="/workspace"))
    landed.land(shipped(dispatcher, "python train.py"))
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
