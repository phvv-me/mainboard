# The leaf every dispatch submodule imports instead of the package root, so nothing inside
# dispatch depends on `dispatch/__init__.py` and its re-exports.

import logging
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=fixed local invocation off PATH, not untrusted input since=2026-08-18
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator

from ..core.project import Project


def now() -> str:
    """The current instant as an ISO-8601 string, the timestamp format every record shares."""
    return datetime.now(UTC).isoformat()


def since(stamp: str) -> str:
    """How long ago `stamp` was, as a compact `3h12m`, empty when it names no instant.

    A stamp with no offset is read as UTC rather than as this machine's local clock, since every
    line this workspace writes is aware and reading one locally would invent a whole timezone's
    worth of waiting.
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


# The variables a run reads its provenance from, set by whatever dispatched it. Here in the leaf
# because the dispatch that exports them and the receipt that reads them sit at two ends of the
# package, and neither should drag the other's module in to agree on a name.
SOURCE_VAR = "MAINBOARD_SOURCE"
COMMIT_VAR = "MAINBOARD_SOURCE_COMMIT"
DIGEST_VAR = "MAINBOARD_SOURCE_DIGEST"
CLOSURE_VAR = "MAINBOARD_CLOSURE"
FIRST_PARTY_VAR = "MAINBOARD_FIRST_PARTY"


def git(*args: str, exact: bool = False) -> str:
    """Stripped stdout of a local `git` command, the provenance of whatever is being recorded.

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
        argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False
    )
    return read.stdout if exact else read.stdout.strip()


def _as_handle(value: str | int) -> str:
    """A scheduler job handle as text, whatever shape its scheduler hands it out in.

    pueue numbers its tasks, so a handle read back from JSON (or typed at a CLI) arrives as an
    int often enough that every model holding one accepts both and stores text.
    """
    return str(value)


# The type every model uses for a scheduler handle, so the normalization is stated once.
type HandleId = Annotated[str, BeforeValidator(_as_handle)]


def state_dir() -> str:
    """The subdirectory every dispatch artifact (sqlite state, job scripts, logs) lives under.

    One subdirectory of the workspace's generated tree, so `.mainboard/` never mixes dispatch
    state with the manifest compiler's own output. Workspace-relative on purpose: the same
    string names the directory here and on a host, which is what lets a job script write its
    log where a later poll already knows to look.
    """
    return f"{Project().out_dir}/dispatch"


# The subsystem's path convention computed once, for a reader that wants the value without
# calling; every internal submodule calls `state_dir()` instead.
STATE_DIR = state_dir()


def workspace(start: Path | None = None) -> Path:
    """The workspace `start` belongs to, found upward by its manifest the way `Board` finds it.

    Dispatch state belongs to the workspace, not to whichever directory a command was typed in.
    Rooting it here is what keeps one database under the workspace root instead of an empty
    second one per subdirectory, which is the difference between a cron sweep that settles every
    job and one that finds none.

    start: the directory the search begins in, the current one when None.
    """
    return Project().workspace(start)


def state_path(root: Path | None = None) -> Path:
    """The dispatch state directory as a real path, under `root` or the discovered workspace."""
    return (root or workspace()) / state_dir()


def db_file(root: Path | None = None) -> Path:
    """The shared dispatch SQLite file, holding both the run registry and command history."""
    return state_path(root) / "db.sqlite"


# One logger for the whole subsystem; every module imports this instead of calling
# `logging.getLogger` itself, so a caller configuring `mainboard.dispatch` reaches every module.
logger = logging.getLogger("mainboard.dispatch")


type Watcher = Callable[[str], None]
"""Announces the stage a long operation has reached, so a long run never stands silent.

Here in the leaf rather than beside the first flow that needed one, because the flows that
announce their stages, an onboarding, a rental's landing, a dispatch priming an environment,
sit at three different levels of this package and a type they all name cannot live inside any
one of them.
"""


def announce(stage: str) -> None:
    """The default `Watcher`, logging each stage for a caller that renders no progress."""
    logger.info("%s", stage)
