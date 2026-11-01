"""A settled run's frames: spooled once, with the identities settling knows and nothing forged."""

from pathlib import Path

from mainboard.constants import FRAMES_MARK
from mainboard.core.events import EventSpool, JobFrame, JobReading, Key
from mainboard.settling.frames import FrameRelay
from mainboard.state import TelemetryLake

from .builders import forged, frames_dir, handle_on, ran, settle_with
from .constants import HANDLE, HOST
from .fakes import Connections


def _spooled(lake: TelemetryLake) -> list[str]:
    """The names of the events the spool holds."""
    files = sorted(lake.spool.glob("*.jsonl"))
    return [event.name for path in files for event in EventSpool.read(path).events]


def test_a_settled_runs_frames_are_spooled_under_the_handle_its_runner_could_not_know(
    tmp_path: Path, relay: FrameRelay, lake: TelemetryLake, connections: Connections
) -> None:
    sent = relay.pull(handle_on(tmp_path), ran(tmp_path))
    (path,) = lake.spool.glob("jobs-*.jsonl")
    events = EventSpool.read(path).events
    assert (sent, [event.name for event in events]) == (2, [JobFrame.started, JobFrame.settled])
    assert {event.identity.job for event in events} == {f"{HOST}/{HANDLE}"}
    assert (connections.asked, path.stat().st_mode & 0o777) == ([HOST], 0o600)


def test_pulling_a_run_twice_spools_nothing_the_second_time(
    tmp_path: Path, relay: FrameRelay, lake: TelemetryLake, connections: Connections
) -> None:
    output = ran(tmp_path)
    first = relay.pull(handle_on(tmp_path), output)
    second = relay.pull(handle_on(tmp_path), output)
    assert (first, second, _spooled(lake)) == (2, 0, [JobFrame.started, JobFrame.settled])


def test_a_file_the_job_wrote_itself_keeps_only_its_frames_with_numeric_readings_and_its_identity(
    tmp_path: Path, relay: FrameRelay, lake: TelemetryLake, connections: Connections
) -> None:
    ask = settle_with({}).model_copy(update={"name": "mb.policy.ask"})
    lying = settle_with(
        {
            Key.job: "elsewhere/1",
            JobReading.exit_code: "0",
            JobReading.stop: "rm -rf",
            "other": "x",
        }
    )
    relay.pull(handle_on(tmp_path), forged(tmp_path, [ask, lying]))
    (path,) = lake.spool.glob("jobs-*.jsonl")
    (event,) = EventSpool.read(path).events
    assert (event.attributes, event.body) == ({Key.host: HOST, Key.job: f"{HOST}/{HANDLE}"}, "")


def test_nothing_is_read_from_a_host_before_the_sink_has_made_the_lake(
    workspace: Path, tmp_path: Path, connections: Connections
) -> None:
    lake = TelemetryLake(root=workspace)
    sent = FrameRelay(lake).pull(handle_on(tmp_path), ran(tmp_path))
    assert (sent, connections.asked, lake.home.exists()) == (0, [], False)


def test_a_name_in_the_output_that_is_no_spool_file_is_never_read(
    tmp_path: Path, relay: FrameRelay, connections: Connections
) -> None:
    output = f"{FRAMES_MARK}../../../etc/passwd\n{FRAMES_MARK}{'a' * 32}.jsonl.bak\n"
    sent = relay.pull(handle_on(tmp_path), output)
    assert (sent, connections.asked, frames_dir(tmp_path).exists()) == (0, [], False)


def test_a_host_that_does_not_answer_costs_the_run_nothing(
    tmp_path: Path, relay: FrameRelay, lake: TelemetryLake, connections: Connections
) -> None:
    connections.answering = False
    sent = relay.pull(handle_on(tmp_path), ran(tmp_path))
    assert (sent, connections.asked, lake.spool.exists()) == (0, [HOST], False)
