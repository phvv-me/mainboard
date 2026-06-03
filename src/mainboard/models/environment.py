from __future__ import annotations

import getpass
import grp
import os
import shutil

from ..enums import Scheduler
from .base import FrozenModel


def _user() -> str:
    """Login name of the current user, or "" if it cannot be determined."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return ""


def _primary_group() -> str:
    """Primary group name of the current user, or "" if it cannot be resolved."""
    try:
        return grp.getgrgid(os.getgid()).gr_name
    except (KeyError, OSError):
        return ""


def _all_groups() -> tuple[str, ...]:
    """Every group the current user belongs to."""
    try:
        return tuple(grp.getgrgid(gid).gr_name for gid in os.getgroups())
    except (KeyError, OSError):
        return ()


def _scheduler() -> Scheduler:
    """Job scheduler on PATH; cluster schedulers take priority over pueue."""
    if shutil.which("sbatch"):
        return Scheduler.SLURM
    if shutil.which("qsub"):
        return Scheduler.PBS
    if shutil.which("pueue"):
        return Scheduler.PUEUE
    return Scheduler.NONE


class Environment(FrozenModel):
    """The host's execution environment: who is running and what scheduler is available.

    Probed from the OS and PATH so a tool can route work without re-detecting the
    user, group, or job scheduler on its own.

    user: login name of the current user.
    group: primary group name of the current user.
    groups: every group the current user belongs to.
    scheduler: the job scheduler found on PATH.
    """

    user: str = ""
    group: str = ""
    groups: tuple[str, ...] = ()
    scheduler: Scheduler = Scheduler.NONE

    @classmethod
    def probe(cls) -> Environment:
        """Detect the current user, group(s), and job scheduler."""
        return cls(
            user=_user(),
            group=_primary_group(),
            groups=_all_groups(),
            scheduler=_scheduler(),
        )
