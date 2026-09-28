# The leaf every dispatch submodule imports instead of the package root, so nothing inside
# dispatch depends on `dispatch/__init__.py` and its re-exports.

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator

from ..core.project import Project
from ..log import logger


def now() -> str:
    """The current instant as an ISO-8601 string, the timestamp format every record shares."""
    return datetime.now(UTC).isoformat()


def since(stamp: str) -> str:
    """How long ago `stamp` was, as a compact `3h12m`, empty when it names no instant.

    A naive stamp is read as UTC, since every line this workspace writes is aware and a local
    reading would invent a timezone's worth of waiting.
    """
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return ""
    aware = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    days, rest = divmod(max(0, int((datetime.now(UTC) - aware).total_seconds())), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, seconds = divmod(rest, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m{seconds}s" if minutes else f"{seconds}s"


# The variables a run reads its provenance from, here because the dispatch exporting them and the
# receipt reading them sit at opposite ends of the package. Each is exported under every name the
# tool answers to, since the job may import an older release than the one that dispatched it.
SOURCE_VAR = Project().variable("SOURCE")
COMMIT_VAR = Project().variable("SOURCE_COMMIT")
DIGEST_VAR = Project().variable("SOURCE_DIGEST")
CLOSURE_VAR = Project().variable("CLOSURE")
FIRST_PARTY_VAR = Project().variable("FIRST_PARTY")
DEFERRED_VAR = Project().variable("DEFERRED")


# A scheduler job handle, always stored as text: pueue numbers its tasks, so a handle read back
# from JSON or typed at a CLI often arrives as an int.
type HandleId = Annotated[str, BeforeValidator(str)]


def state_dir(root: Path | None = None) -> str:
    """The subdirectory of the generated tree every dispatch artifact (sqlite state, job scripts,
    logs) lives under, apart from the manifest compiler's output, in workspace `root` (the cwd's
    workspace). Workspace-relative so the same string names it here and on a host, where a job
    script writes its log for a later poll."""
    return f"{Project().out_dir(root)}/dispatch"


def workspace(start: Path | None = None) -> Path:
    """The workspace `start` (the current directory when None) belongs to, found upward by its
    manifest as `Board` finds it.

    Dispatch state belongs to the workspace, keeping one database rather than an empty one per
    subdirectory: the difference between a cron sweep that settles every job and one that finds
    none.
    """
    return Project().workspace(start)


def state_path(root: Path | None = None) -> Path:
    """The dispatch state directory as a real path, under `root` or the discovered workspace."""
    here = root or workspace()
    return here / state_dir(here)


type Watcher = Callable[[str], None]
"""Announces the stage a long operation has reached, so a long run never stands silent.

In the leaf because its users (onboarding, a rental's landing, a dispatch priming an
environment) sit at three different levels of this package.
"""


def announce(stage: str) -> None:
    """The default `Watcher`, logging each stage for a caller that renders no progress."""
    logger.info(stage)
