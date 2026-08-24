# The library facade, resolved on first touch rather than at import.
#
# `mainboard.cli` is a console entry point, so every command run from a terminal executes this
# file before its own verb, and naming a subsystem here used to mean importing it whether or not
# that verb had any use for it. The profiler alone is 15 ms of a 250 ms start for a `doctor` that
# profiles nothing. PEP 562 keeps the flat spelling every caller already writes, `from mainboard
# import Board`, and charges for a name only when something actually reads it. This replaces the
# `__lazy_modules__` declaration that sat here, which was a forward-compatible note to a PEP 810
# interpreter and inert on the one this package runs on.

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
    from .profile.profiler import Collection, Profiler, Reach
    from .profile.result import Profile
    from .profile.spans import span
    from .profile.study import Study as ProfileStudy

# Where each exported name lives and what it is called there, which is the whole facade. The
# second half of each pair is only ever different for the two `Study` classes, an experiment's
# and a profile's, which the flat namespace has to tell apart.
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
    "Reach": (".profile.profiler", "Reach"),
    "RepoFile": (".experiments.data", "RepoFile"),
    "Resolver": (".context", "Resolver"),
    "Survey": (".compute", "Survey"),
    "gpu_busy": (".probe.gating", "gpu_busy"),
    "load": (".manifest", "load"),
    "script": (".core.shell", "script"),
    "sh": (".core.shell", "sh"),
    "span": (".profile.spans", "span"),
    "wait_for_idle": (".probe.gating", "wait_for_idle"),
}

__all__ = [
    *_HOMES,
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
    "Reach",
    "Profile",
    "span",
    "ProfileStudy",
]


def __getattr__(name: str) -> object:
    """One exported name, importing the module that defines it on first ask.

    Bound onto this module afterwards, so a name costs its import once and is a plain attribute
    lookup from then on. Anything this facade never exported raises the same `AttributeError` a
    missing module attribute always did.

    name: the exported name being read.
    """
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = home
    found = getattr(import_module(module, __name__), attribute)
    globals()[name] = found
    return found


__version__ = "0.1.0"
