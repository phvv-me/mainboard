import os
import subprocess
import sys
from pathlib import Path

from mainboard.core.events import FrameSpec
from mainboard.runtime import Job
from mainboard.runtime.job import WorkspaceActivation

from .constants import CANARY, HOST

# A runner of the job in a process of its own, beating every tenth of a second.
_RUNNER = (
    "import sys; from mainboard.runtime import Job; "
    "from mainboard.verbs.work.runner import Runner; "
    "runner = Runner(Job.model_validate_json(sys.argv[1])); runner.frames.interval = 0.1; "
    "runner.run()"
)


def job_in(
    root: Path,
    command: str,
    *,
    directory: Path | None = None,
    walltime: str = "",
    prefix: str = sys.prefix,
) -> Job:
    """A job running `command` that frames itself into `directory`, `root/frames` when unset.

    prefix: the environment it enters, which refuses to start the job when it holds no `bin/`.
    """
    return Job(
        command=command,
        root=str(root),
        activation=WorkspaceActivation(script="", prefix=prefix, refusal="missing prefix"),
        walltime=walltime,
        frames=FrameSpec(directory=str(directory or root / "frames"), host=HOST),
    )


def environment() -> dict[str, str]:
    """The environment a runner starts in: a PATH to run commands with, and a secret."""
    return {"PATH": os.environ["PATH"], "MB_CANARY": CANARY}


def spawned(job: Job) -> subprocess.Popen[bytes]:
    """The job run by a runner in a process of its own, which beats every tenth of a second."""
    return subprocess.Popen(
        [sys.executable, "-c", _RUNNER, job.model_dump_json()],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
