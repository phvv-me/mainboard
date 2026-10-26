# The library facade: `from mainboard import Board` and the rest of the flat namespace.

from importlib.metadata import PackageNotFoundError, version

from .context import ExecutionPlan, Resolver
from .core import MissionError, Project
from .core.execution.shell import script, sh
from .experiments.data import HfDataset, HfModel, Needs, RepoFile
from .fleet.board import Board, Job
from .fleet.compute import ComputePath, Survey
from .log import logger
from .manifest import Manifest, load
from .probe.machine import Machine
from .probe.readings.gating import gpu_busy, wait_for_idle
from .probe.readings.snapshot import HostFacts
from .profile.health.meter import Meter
from .profile.profiler import Collection, Profiler
from .profile.result import Profile
from .profile.spans import span
from .profile.studies.study import Study as ProfileStudy
from .results import Results

__all__ = [
    "Board",
    "Job",
    "ComputePath",
    "Survey",
    "ExecutionPlan",
    "Resolver",
    "MissionError",
    "Project",
    "script",
    "sh",
    "HfDataset",
    "HfModel",
    "Needs",
    "RepoFile",
    "Manifest",
    "load",
    "gpu_busy",
    "wait_for_idle",
    "Machine",
    "HostFacts",
    "Meter",
    "Collection",
    "Profiler",
    "Profile",
    "span",
    "ProfileStudy",
    "Results",
    "logger",
]


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
