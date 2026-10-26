# Which settler this machine offers, and the one verb that installs or removes the pass.

from typing import TYPE_CHECKING

from patos import Strategy

from .base import Every, Settler, Unsupported
from .launchd import LaunchdAgent
from .systemd import SystemdUser

if TYPE_CHECKING:
    from pathlib import Path

    from .base import Settling

# Walked in order, the first implementation this machine can run winning and the null answer
# catching the rest. Registered by class, since which one wins never depends on the workspace.
_SETTLERS: Strategy[type[Settler]] = Strategy("settler")
_SETTLERS.register("systemd", SystemdUser)
_SETTLERS.register("launchd", LaunchdAgent)
_SETTLERS.register("none", Unsupported)


def settler(root: Path) -> Settler:
    """The periodic runner this machine offers `root`, the refusing one where it offers none."""
    return _SETTLERS.first_available()(root)


def schedule(root: Path, every: str) -> Settling:
    """Install the durable settling pass for `root` at `every` (`20m`), or remove it at `0`."""
    period = Every.parse(every)
    machine = settler(root)
    return machine.install(period) if period.seconds else machine.remove()
