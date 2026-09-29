# The leaf every dispatch submodule imports instead of the package root, so nothing inside
# dispatch depends on `dispatch/__init__.py` and its re-exports.

import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=fixed local git invocation off PATH, not untrusted input since=2026-08-18
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
# Where a job keeps what it resumes from, the same directory on every attempt of one run name,
# and which attempt this is (1 first).
CHECKPOINT_VAR = Project().variable("CHECKPOINT")
ATTEMPT_VAR = Project().variable("ATTEMPT")


def git(*args: str, exact: bool = False) -> str:
    """Stripped stdout of a local `git` command, the provenance of whatever is being recorded.

    Public: experiments in cutok and reproducibility import it to stamp their registrations.

    On `/dev/null` for the same reason every ssh this tool runs is: a dispatch is routinely
    called from inside a shell loop reading handles, and a child left on the caller's stdin can
    eat the rest of that loop's input. Nothing asked for here reads any.

    Here in the leaf rather than beside the one dispatch that first needed it, because a trial
    receipt asks git the same two questions a submit does and neither should drag the other's
    module in to do it.

    exact: keep the output byte for byte. A porcelain status line starts with the space that
        means `unstaged`, and stripping it turns ` M src/x.py` into `M src/x.py`, a staged
        change to a file called `rc/x.py`.
    """
    argv = ["git", *args]  # fixed local invocation off PATH, not untrusted input
    read = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=fixed local invocation off PATH, not untrusted input since=2026-08-16
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return read.stdout if exact else read.stdout.strip()


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
