# The installed snapshot's own freshness, kept current without anyone asking. The CLI on PATH is
# a uv tool snapshot of the package source, so an edit to that source silently changes nothing
# until someone reinstalls. The trap earned a standing check, and the check earned a nag that
# printed on every invocation until somebody ran the reinstall by hand: 139 copies of it in seven
# sessions, 186 filters written to hide it, and eleven `--json` parses that had to work around it.
# A line nobody acts on is noise, so the snapshot now does the reinstall itself and re-executes
# the command it was asked for on the new code, and nothing about any of it touches stdout.
#
# The uv receipt beside the installed environment says exactly which directory the snapshot was
# built from and with which extras, so the check needs no configuration: digest the source tree,
# remember what it looked like when this snapshot first ran, and refresh the moment the tree moves
# past it.
#
# The digest is over source and pyproject names, sizes and mtimes rather than contents, which
# keeps the whole check in the low milliseconds a CLI startup can afford. Package metadata is
# part of the snapshot because changing a runtime dependency changes what the installed command
# can do even when no Python module moved. The digest is recorded on the snapshot's first run
# rather than at install because uv owns the install and offers no hook. The one blind spot that
# buys is an edit landing between the install and the first run, which the next edit clears.
#
# Windows cannot replace an interpreter that is running, and this process is running the one the
# reinstall rebuilds. There the update is handed to a worker that waits for this process to exit,
# and the command at hand answers from the snapshot it started on, saying so on stderr once.

import hashlib
import json
import os
import platform
import sys
import tomllib
from contextlib import suppress
from functools import partial
from pathlib import Path
from time import time

from filelock import FileLock, Timeout
from patos import FrozenModel
from plumbum import CommandNotFound
from plumbum.commands.processes import ProcessTimedOut

from .core.errors import MissionError
from .core.project import Project
from .engines.compile.backend.engine import PixiEngine
from .engines.compile.backend.process import Process

# The file uv writes beside every tool it installs, naming the source of the snapshot.
_RECEIPT = "uv-receipt.toml"

# Where this check remembers the source tree the running snapshot answered for.
_STATE = "source-state.json"

# The lock two processes finding the same snapshot stale take turns under, beside the state.
_LOCK = "self-update.lock"

# The extra a plain reinstall silently drops, so the named command always carries it.
_EXTRA = "wandb"

# The exact package Pixi supplies to the otherwise isolated self-update process. uv is never a
# host-level prerequisite or a binary Mainboard searches for: Pixi resolves and runs this package
# inside its own cached exec environment.
_UV = "uv=0.12.7"

# The helper has to outlive the launcher it replaces. Its Pixi exec environment is independent
# of the uv tool directory, so these packages remain available while uv removes and rebuilds it.
_DEFERRED_SPECS = ("python=3.14", "psutil=7.2.2", "cyclopts=4.23")

# Set in the environment of the process a refresh re-executes, so an update that did not take
# answers from the snapshot it has rather than reinstalling in a loop.
REFRESHED = "MAINBOARD_REFRESHED"

# How long one process waits for another's reinstall of the same snapshot, which is a wheel
# build from a local tree, before answering from the snapshot it already has.
_LOCK_SECONDS = 300.0

# How long the reinstall itself may take before it is abandoned.
_INSTALL_SECONDS = 600.0

# How long a scheduled Windows update is trusted to still be on its way. Every command run while
# the launcher is waiting to be released finds the same stale snapshot, and scheduling a worker
# for each of them races several uv installs over one tool directory.
_PENDING_SECONDS = 600.0


class Snapshot(FrozenModel):
    """What the running snapshot knows about its own source.

    installed: whether this process runs from a uv tool snapshot at all; a checkout running
        its own source has nothing to be stale against.
    stale: whether the source tree has moved past what this snapshot was recorded against.
    detail: the one line behind the answer.
    uv: the reinstall as a bare uv argv, empty when nothing needs one. Carried whole rather than
        sliced out of the Pixi command: the deferred Windows worker runs exactly this once the
        launcher it replaces has exited.
    source: the package directory the snapshot was installed from, absolute, and the workspace
        the deferred worker writes its log into.
    tool: the uv tool directory holding the snapshot, where its receipt and state live.
    marker: the identity of the install this answer was read from, which is how a process that
        waited on another's reinstall knows the snapshot already moved.
    """

    installed: bool
    stale: bool = False
    detail: str = ""
    uv: tuple[str, ...] = ()
    source: Path | None = None
    tool: Path | None = None
    marker: str = ""

    @property
    def fix(self) -> tuple[str, ...]:
        """The reinstall as Pixi runs it, empty when nothing needs one."""
        return ("exec", "--spec", _UV, *self.uv) if self.uv else ()


class Refresh:
    """Brings a stale snapshot up to its source, then re-executes the command on the new one.

    Two processes finding the same snapshot stale take turns under one lock beside it, and the
    second one in finds the install already replaced and only re-executes. A job wave starting
    nine commands at once on a freshly synced host therefore reinstalls once.
    """

    def __init__(self, found: Snapshot) -> None:
        """found: a stale snapshot from `check()`, carrying its tool directory and source."""
        self.found = found
        self.tool = found.tool or Path(sys.prefix)

    def run(self) -> None:
        """Update, then replace this process with the same command on the updated snapshot.

        Returns only when the command has to answer from the snapshot it started on: a Windows
        update deferred until this process exits, or an update that failed, both said on stderr.
        """
        if platform.system() == "Windows":
            self.defer()
            return
        try:
            with FileLock(self.tool / _LOCK, timeout=_LOCK_SECONDS):
                failure = "" if self.replaced() else self.reinstall()
        except Timeout:
            failure = f"another update held its lock for {_LOCK_SECONDS:g}s"
        if failure:
            say(f"{self.found.detail} and could not update itself ({failure})")
            return
        say(f"updated from {self.found.source}")
        os.environ[REFRESHED] = "1"
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])

    def replaced(self) -> bool:
        """Whether another process reinstalled the snapshot while this one waited its turn."""
        return _marker(self.tool / _RECEIPT) != self.found.marker

    def reinstall(self) -> str:
        """Run the Pixi-owned reinstall with its output held back, answering why it failed.

        Held back rather than streamed because stdout belongs to the verb this process is about
        to run, and a machine-readable document must be the only thing on it. A failure brings
        the tail of what the installer said.
        """
        try:
            result = PixiEngine().within_cwd(
                partial(Process.capture, timeout=_INSTALL_SECONDS), *self.found.fix
            )
        except (CommandNotFound, MissionError, OSError, ProcessTimedOut) as fault:
            return " ".join(str(fault).split())[:200] or type(fault).__name__
        if result.succeeded:
            return ""
        said = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
        return (said or [f"exit {result.returncode}"])[-1]

    def defer(self) -> None:
        """Hand the Windows update to a worker that runs once this launcher has exited.

        One worker per stale snapshot: a marker beside the worker's log says one is already on
        its way, and the worker removes it when it is done, so the commands run meanwhile answer
        from the snapshot they have without scheduling another install over the same directory.
        """
        log = _refresh_log(self.found.source)
        pending = log.with_suffix(".pending")
        log.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            if time() - pending.stat().st_mtime < _PENDING_SECONDS:
                return
        pending.write_text(str(os.getpid()), encoding="utf-8")
        worker = Path(__file__).with_name("_refresh.py")
        specs = tuple(token for spec in _DEFERRED_SPECS for token in ("--spec", spec))
        PixiEngine().defer(
            "exec",
            "--spec",
            _UV,
            *specs,
            "python",
            str(worker),
            str(os.getpid()),
            str(log),
            "--",
            *self.found.uv,
        )
        say(f"{self.found.detail}; it updates itself once this command exits")


def current() -> None:
    """Keep this process on its source's newest code, the first thing every invocation does.

    A fresh snapshot, and a checkout running its own source, return at once. A stale one is
    refreshed and the command re-executed on it, so the caller never learns anything happened
    beyond one line on stderr. A process that is itself the re-execution never refreshes again:
    if its snapshot is still stale the update did not take, and it says so instead of looping.
    """
    again = os.environ.pop(REFRESHED, None) is not None
    found = check()
    if not found.stale:
        return
    if again:
        say(f"{found.detail} and the update did not take; answering from it anyway")
        return
    Refresh(found).run()


def say(line: str) -> None:
    """Write one diagnostic line, named for the tool, where no document is ever printed."""
    sys.stderr.write(f"{Project().name}: {line}\n")
    sys.stderr.flush()


def _refresh_log(source: Path | None) -> Path:
    """Durable deferred-update log under `source`'s workspace, beside this one when there is none.

    source: the package directory the snapshot was installed from, None when the receipt named
        no source at all.
    """
    return (source or Path.cwd()) / Project().out_dir / "self-update.log"


def check(package: Path | None = None) -> Snapshot:
    """Compare the running snapshot against its recorded source tree, recording on first run.

    package: the installed package directory, this module's own when None.
    """
    home = package or Path(__file__).resolve().parent
    root = tool_root(home)
    if root is None:
        return Snapshot(installed=False, detail="running from source")
    receipt = root / _RECEIPT
    try:
        declared = tomllib.loads(receipt.read_text(encoding="utf-8"))
        requirement = next(
            entry
            for entry in declared["tool"]["requirements"]
            if entry.get("name") == Project().name and "directory" in entry
        )
    except OSError, tomllib.TOMLDecodeError, KeyError, StopIteration:
        return Snapshot(installed=True, detail="the uv receipt names no source directory")
    package = root.joinpath(requirement["directory"]).resolve()
    source = package / "src"
    if not source.is_dir():
        return Snapshot(installed=True, detail=f"no source tree at {source}")
    extras = ",".join(requirement.get("extras") or [_EXTRA])
    interpreter = _durable_interpreter(declared["tool"].get("python"))
    # Absolute, because the deferred Windows worker runs it from wherever pixi's exec
    # environment happens to stand rather than from the directory the receipt was written
    # relative to.
    uv = (
        "uv",
        "tool",
        "install",
        "--reinstall-package",
        Project().name,
        *(("--python", str(interpreter)) if interpreter else ()),
        "--from",
        f"{package}[{extras}]",
        Project().name,
        "--force",
    )
    current = digest(source)
    marker = _marker(receipt)
    recorded = _recorded(root / _STATE, marker=marker, current=current)
    if recorded == current:
        return Snapshot(installed=True, detail="snapshot matches the source tree")
    return Snapshot(
        installed=True,
        stale=True,
        detail=f"the source at {package} is newer than this installed snapshot",
        uv=uv,
        source=package,
        tool=root,
        marker=marker,
    )


def _durable_interpreter(declared: str | None) -> Path | None:
    """An existing uv-tool interpreter that does not belong to generated project state.

    Reusing uv's exact managed-Python path keeps updates deterministic and avoids another
    interpreter download. A Pixi or Mainboard environment is different: it is replaceable
    workspace output, so retaining it in the tool receipt makes the public launcher depend on a
    shard that may move or be rebuilt. Missing interpreters are likewise left for uv to replace.
    """
    if not declared:
        return None
    interpreter = Path(declared)
    generated = {".mainboard", ".pixi"}
    if generated & {part.casefold() for part in interpreter.parts}:
        return None
    return interpreter if interpreter.is_file() else None


def digest(source: Path) -> str:
    """One cheap digest of runtime source and package metadata, with contents unread.

    source: the tree to fingerprint.
    """
    fingerprint = hashlib.sha256()
    package = source.parent
    files = [
        *sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ),
        *(metadata for metadata in (package / "pyproject.toml",) if metadata.is_file()),
    ]
    for path in files:
        stat = path.stat()
        line = f"{path.relative_to(package)}:{stat.st_size}:{stat.st_mtime_ns}\n"
        fingerprint.update(line.encode())
    return fingerprint.hexdigest()


def tool_root(package: Path) -> Path | None:
    """The uv tool directory holding `package`'s snapshot, None when it runs from a checkout.

    package: the imported package's own directory.
    """
    for parent in package.parents:
        if (parent / _RECEIPT).is_file():
            return parent
    return None


def _marker(receipt: Path) -> str:
    """The identity of one install, so a reinstall invalidates what was recorded for the last."""
    return f"{Project().name}:{receipt.stat().st_mtime_ns}"


def _recorded(state: Path, *, marker: str, current: str) -> str:
    """The digest recorded for this install, `current` recorded fresh on a new install.

    A state file that is missing, torn or from another install is replaced with `current`, so
    the first run after an install is the baseline every later run compares against. A tool
    directory that cannot be written leaves the check answering fresh rather than failing the
    command that asked.
    """
    with suppress(OSError, json.JSONDecodeError):
        held = json.loads(state.read_text(encoding="utf-8"))
        if held.get("marker") == marker:
            return str(held.get("digest", ""))
    with suppress(OSError):
        state.write_text(json.dumps({"marker": marker, "digest": current}), encoding="utf-8")
    return current
