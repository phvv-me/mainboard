from .board import Board, Job
from .context import ExecutionPlan, Resolver
from .core import MissionError, Project
from .core.shell import script, sh
from .experiments.data import HfDataset, HfModel, Needs, RepoFile
from .experiments.fleet import Fleet
from .experiments.study import Study as ExperimentStudy
from .manifest import Manifest, load
from .monitor import Monitor
from .probe.gating import gpu_busy, wait_for_idle
from .probe.machine import Machine
from .probe.snapshot import HostFacts
from .profile.meter import Meter
from .profile.profiler import Collection, Profiler, Reach
from .profile.result import Profile
from .profile.spans import span
from .profile.study import Study as ProfileStudy

__all__ = [
    "Board",
    "Collection",
    "ExecutionPlan",
    "ExperimentStudy",
    "Fleet",
    "HfDataset",
    "HfModel",
    "HostFacts",
    "Job",
    "Machine",
    "Manifest",
    "Meter",
    "MissionError",
    "Monitor",
    "Needs",
    "Profile",
    "ProfileStudy",
    "Profiler",
    "Project",
    "Reach",
    "RepoFile",
    "Resolver",
    "gpu_busy",
    "load",
    "script",
    "sh",
    "span",
    "wait_for_idle",
]
# PEP 810 forward-compatible declaration, inert on 3.14 and letting 3.15
# defer these subpackage imports for CLI startup.
__lazy_modules__ = [
    "mainboard.board",
    "mainboard.context",
    "mainboard.dispatch",
    "mainboard.engines",
    "mainboard.experiments",
    "mainboard.manifest",
    "mainboard.monitor",
    "mainboard.probe",
    "mainboard.profile",
]
__version__ = "0.1.0"
