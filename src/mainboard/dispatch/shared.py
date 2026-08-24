# The leaf every dispatch submodule imports instead of the package root, so nothing inside
# dispatch depends on `dispatch/__init__.py` and its re-exports.

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator

from ..core.project import Project


def now() -> str:
    """The current instant as an ISO-8601 string, the timestamp format every record shares."""
    return datetime.now(UTC).isoformat()


def as_handle(value: str | int) -> str:
    """A scheduler job handle as text, whatever shape its scheduler hands it out in.

    pueue numbers its tasks, so a handle read back from JSON (or typed at a CLI) arrives as an
    int often enough that every model holding one accepts both and stores text.
    """
    return str(value)


# The type every model uses for a scheduler handle, so the normalization is stated once.
type HandleId = Annotated[str, BeforeValidator(as_handle)]


def state_dir() -> str:
    """The subdirectory every dispatch artifact (sqlite state, job scripts, logs) lives under.

    One subdirectory of the workspace's generated tree, so `.mainboard/` never mixes dispatch
    state with the manifest compiler's own output. Workspace-relative on purpose: the same
    string names the directory here and on a host, which is what lets a job script write its
    log where a later poll already knows to look.
    """
    return f"{Project().out_dir}/dispatch"


# The subsystem's public path convention, computed once for a caller reading
# `mainboard.dispatch.STATE_DIR`; every internal submodule calls `state_dir()` instead.
STATE_DIR = state_dir()


def workspace(start: Path | None = None) -> Path:
    """The workspace `start` belongs to, found upward by its manifest the way `Board` finds it.

    Dispatch state belongs to the workspace, not to whichever directory a command was typed in.
    Rooting it here is what keeps one database under the workspace root instead of an empty
    second one per subdirectory, which is the difference between a cron sweep that settles every
    job and one that finds none. A directory under no workspace at all keeps its own state, so a
    scratch tree stays self-contained rather than raising.

    start: the directory the search begins in, the current one when None.
    """
    here = start or Path.cwd()
    try:
        return Project().find_root(here)
    except FileNotFoundError:
        return here


def state_path(root: Path | None = None) -> Path:
    """The dispatch state directory as a real path, under `root` or the discovered workspace."""
    return (root or workspace()) / state_dir()


def db_file(root: Path | None = None) -> Path:
    """The shared dispatch SQLite file, holding both the run registry and command history."""
    return state_path(root) / "db.sqlite"


# One logger for the whole subsystem; every module imports this instead of calling
# `logging.getLogger` itself, so a caller configuring `mainboard.dispatch` reaches every module.
logger = logging.getLogger("mainboard.dispatch")
