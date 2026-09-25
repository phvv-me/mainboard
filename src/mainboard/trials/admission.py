"""Admit a card to an acquisition under the standard the workspace declares for it.

The policy lives in the manifest's `[admission.<card name>]` table, so a display-attached desktop
or a shared box is declared once and every acquisition on that card reads the same answer.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.project import Project
from ..manifest.loading import load
from ..manifest.schema.admission import Admission
from ..probe.gating import wait_for_idle

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..probe.units.gpu import GPU


class Admitted(FrozenModel):
    """What the card looked like when the acquisition was admitted, kept beside its rows.

    threshold_pct: the utilization the card had to stay below.
    memory_pct: its memory-controller utilization at admission.
    holders: the other compute processes on it, empty on an exclusive card.
    """

    card: str
    threshold_pct: int
    utilization_pct: int
    memory_pct: int
    holders: tuple[int, ...] = ()


def policy(card: str, root: Path | None = None) -> Admission:
    """The admission standard the workspace declares for `card`, the default when it names none.

    root: the workspace root; found from the working directory when omitted, which inside a
        dispatched job is the pinned tree the manifest shipped with.
    """
    manifest = load((root or Project().find_root(Path.cwd())) / "mainboard.toml")
    return manifest.admission.get(card, Admission())


def admit(
    device: GPU,
    *,
    timeout: float = 60.0,
    root: Path | None = None,
    idle: Callable[..., bool] = wait_for_idle,
) -> Admitted:
    """Wait for `device` to meet its declared standard, then say what it looked like.

    Refuses naming the standard when the card stays busy past `timeout` seconds, and naming the
    holders when other compute processes hold a card whose policy refuses them.

    root: the workspace root the policy is read from, as in `policy`.
    idle: the idle wait, injectable by a test.
    """
    standard = policy(device.label, root)
    if not idle(timeout=timeout, util_threshold=standard.utilization_pct):
        raise RuntimeError(
            f"{device.label} stayed above {standard.utilization_pct}% utilization for "
            f"{timeout:g} s; the acquisition needs an idle card"
        )
    holders = tuple(
        process.pid for process in device.snapshot().processes if process.pid != os.getpid()
    )
    if holders and standard.holders == "refuse":
        raise RuntimeError(f"{device.label} has other compute processes: {list(holders)}")
    reading = device.utilization
    return Admitted(
        card=device.label,
        threshold_pct=standard.utilization_pct,
        utilization_pct=reading.gpu_pct,
        memory_pct=reading.memory_pct,
        holders=holders,
    )
