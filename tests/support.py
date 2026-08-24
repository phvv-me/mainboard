from collections.abc import Iterator
from types import SimpleNamespace

from mainboard import Board, ComputePath, HostFacts
from mainboard.compute import Survey
from mainboard.deps import Change, Dependencies
from mainboard.dispatch import HostSetup
from mainboard.dispatch.state import MonitorReport
from mainboard.doctor import Doctor, Section
from mainboard.monitor import Monitor
from mainboard.scaffold import Scaffold, Scaffolded

# What a stand-in is handed, what it hands back, and what one recorded call looks like. The
# verbs pass names and commands positionally and everything else by keyword, so the option
# values are exactly the scalar kinds a flag parses into.
type Owner = Board | Dependencies | Doctor | Monitor | Scaffold | Survey
type Option = str | int | float | bool | dict[str, str] | None
type Answer = (
    int
    | None
    | HostFacts
    | HostSetup
    | MonitorReport
    | Scaffolded
    | SimpleNamespace
    | list[Change]
    | list[ComputePath]
    | list[Section]
    | Iterator[MonitorReport]
)
type Relayed = tuple[str, str, tuple[str, ...], dict[str, Option]]
