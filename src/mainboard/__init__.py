# The library facade, resolved on first touch rather than at import.
#
# Every command run from a terminal executes this file before its own verb, and importing each
# subsystem here charged every verb for all of them (the profiler alone was 15 ms of a 250 ms
# `doctor` start). PEP 562 keeps the flat `from mainboard import Board` and charges for a name only
# when something reads it; a PEP 810 `__lazy_modules__` declaration is inert on this interpreter.

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .board import Board, Job
    from .compute import ComputePath, Survey
    from .context import ExecutionPlan, Resolver
    from .core import MissionError, Project
    from .core.shell import script, sh
    from .experiments.data import HfDataset, HfModel, Needs, RepoFile
    from .experiments.fleet import Fleet
    from .experiments.study import Study as ExperimentStudy
    from .manifest import Manifest, load
    from .probe.gating import gpu_busy, wait_for_idle
    from .probe.machine import Machine
    from .probe.snapshot import HostFacts
    from .profile.meter import Meter
    from .profile.profiler import Collection, Profiler
    from .profile.result import Profile
    from .profile.spans import span
    from .profile.study import Study as ProfileStudy
    from .results import Results

# Where each exported name lives and what it is called there, which is the whole facade; the two
# differ only for the experiment's and the profile's `Study`, which the flat namespace tells apart.
_HOMES: dict[str, tuple[str, str]] = {
    "Board": (".board", "Board"),
    "Collection": (".profile.profiler", "Collection"),
    "ComputePath": (".compute", "ComputePath"),
    "ExecutionPlan": (".context", "ExecutionPlan"),
    "ExperimentStudy": (".experiments.study", "Study"),
    "Fleet": (".experiments.fleet", "Fleet"),
    "HfDataset": (".experiments.data", "HfDataset"),
    "HfModel": (".experiments.data", "HfModel"),
    "HostFacts": (".probe.snapshot", "HostFacts"),
    "Job": (".board", "Job"),
    "Machine": (".probe.machine", "Machine"),
    "Manifest": (".manifest", "Manifest"),
    "Meter": (".profile.meter", "Meter"),
    "MissionError": (".core", "MissionError"),
    "Needs": (".experiments.data", "Needs"),
    "Profile": (".profile.result", "Profile"),
    "ProfileStudy": (".profile.study", "Study"),
    "Profiler": (".profile.profiler", "Profiler"),
    "Project": (".core", "Project"),
    "RepoFile": (".experiments.data", "RepoFile"),
    "Resolver": (".context", "Resolver"),
    "Results": (".results", "Results"),
    "Survey": (".compute", "Survey"),
    "gpu_busy": (".probe.gating", "gpu_busy"),
    "load": (".manifest", "load"),
    "script": (".core.shell", "script"),
    "sh": (".core.shell", "sh"),
    "span": (".profile.spans", "span"),
    "wait_for_idle": (".probe.gating", "wait_for_idle"),
}

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
    "Fleet",
    "ExperimentStudy",
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
]


def __getattr__(name: str) -> object:
    """One exported name, imported on first ask and bound here so later reads are plain lookups.

    Anything this facade never exported raises the usual missing-attribute `AttributeError`.
    """
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = home
    found = getattr(import_module(module, __name__), attribute)
    globals()[name] = found
    return found


__version__ = "0.1.0"
