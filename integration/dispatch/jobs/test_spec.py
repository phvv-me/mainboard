"""The record a dispatch hands a host: what the host's runner is told besides the command."""

from mainboard import ExecutionPlan
from mainboard.core.events import FrameSpec
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.shared import state_dir
from mainboard.manifest import HostProfile
from mainboard.runtime import Job


def test_a_job_record_tells_the_runner_which_host_it_frames_itself_as_and_where_to_spool() -> None:
    plan = ExecutionPlan(host="gold", profile=HostProfile(kind="ssh"), env="default")
    spec = JobSpec(cmd="true", plan=plan, root="/work/pinned")
    record = Job.handed(spec.render(pbs=False))
    assert record.frames == FrameSpec(directory=f"/work/pinned/{state_dir()}/frames", host="gold")
