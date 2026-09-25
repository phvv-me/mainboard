# The installed snapshot keeps itself current. The CLI on PATH is a uv tool snapshot of the
# package source, so a source edit changes nothing until a reinstall. A nag asking for that
# reinstall printed on every invocation (139 copies in seven sessions, 186 filters written to hide
# it, eleven `--json` parses working around it), so the snapshot now reinstalls itself and
# re-executes the command on the new code, never touching stdout.
#
# The uv receipt beside the snapshot names its source directory and extras, so the check needs no
# configuration. It digests source and pyproject names, sizes and mtimes (never contents, to stay
# in the low milliseconds a CLI startup affords; pyproject because a runtime dependency changes
# what the command can do), records the digest on the snapshot's first run because uv owns the
# install and offers no hook, and refreshes once the tree moves past it. The blind spot is an edit
# landing between install and first run, which the next edit clears.
#
# Windows cannot replace the running interpreter, so there the update goes to a worker that waits
# for this process to exit, and the command answers from its snapshot, saying so once on stderr.

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

_RECEIPT = "uv-receipt.toml"
_STATE = "source-state.json"
_LOCK = "self-update.lock"

# The extra a plain reinstall silently drops, so the named command always carries it.
_EXTRA = "wandb"

# uv is never a host prerequisite: Pixi resolves and runs this exact package in its cached exec
# environment.
_UV = "uv=0.12.7"

# The worker's Pixi exec environment is independent of the uv tool directory, so these packages
# outlive the launcher while uv removes and rebuilds it.
_DEFERRED_SPECS = ("python=3.14", "psutil=7.2.2", "cyclopts=4.23")

# Set in the environment of the process a refresh re-executes, so an update that did not take
# answers from the snapshot it has rather than reinstalling in a loop.
REFRESHED = "MAINBOARD_REFRESHED"

# How long one process waits for another's reinstall of the same snapshot, which is a wheel
# build from a local tree, before answering from the snapshot it already has.
_LOCK_SECONDS = 300.0

# How long the reinstall itself may take before it is abandoned.
_INSTALL_SECONDS = 600.0

# How long a scheduled Windows update is trusted to still be on its way. Every command run
# meanwhile finds the same stale snapshot, and a worker for each would race uv installs over one
# tool directory.
_PENDING_SECONDS = 600.0


class Snapshot(FrozenModel):
    """What the running snapshot knows about its own source.

    installed: whether this process runs from a uv tool snapshot; a checkout has nothing to be
        stale against.
    uv: the reinstall as a bare uv argv, empty when nothing needs one; the deferred Windows
        worker runs exactly this.
    source: the absolute package directory the snapshot was installed from, where the deferred
        worker writes its log.
    tool: the uv tool directory holding the snapshot, its receipt and state.
    marker: the identity of the install this answer was read from, so a process that waited on
        another's reinstall knows the snapshot already moved.
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
        """found: a stale snapshot from `check()`."""
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
        """Run the Pixi-owned reinstall, answering why it failed ("" on success).

        Output is held back because stdout belongs to the verb about to run; a failure brings the
        installer's last line.
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

        One worker per stale snapshot: a marker beside its log says one is on its way until the
        worker removes it, so commands run meanwhile schedule no second install.
        """
        log = _refresh_log(self.found.source)
        pending = log.with_suffix(".pending")
        log.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            if time() - pending.stat().st_mtime < _PENDING_SECONDS:
                return
        pending.write_text(str(os.getpid()), encoding="utf-8")
        worker = str(Path(__file__).with_name("_refresh.py"))
        specs = [token for spec in (_UV, *_DEFERRED_SPECS) for token in ("--spec", spec)]
        pid = str(os.getpid())
        PixiEngine().defer("exec", *specs, "python", worker, pid, str(log), "--", *self.found.uv)
        say(f"{self.found.detail}; it updates itself once this command exits")


def current() -> None:
    """Keep this process on its source's newest code, the first thing every invocation does.

    A stale snapshot is refreshed and the command re-executed on it, with one line on stderr. The
    re-execution itself never refreshes again: still stale means the update did not take, and it
    says so instead of looping.
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
    """Durable deferred-update log under `source`'s workspace, the working directory's if None."""
    return (source or Path.cwd()) / Project().out_dir / "self-update.log"


def check(package: Path | None = None) -> Snapshot:
    """Compare the running snapshot against its recorded source tree, recording on first run.

    package: the installed package directory, this module's own when None.
    """
    root = tool_root(package or Path(__file__).resolve().parent)
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
    # Absolute, because the deferred Windows worker runs from wherever pixi's exec environment
    # stands, not the directory the receipt is relative to.
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
    tree = digest(source)
    marker = _marker(receipt)
    if _recorded(root / _STATE, marker=marker, current=tree) == tree:
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

    Reusing uv's managed-Python path keeps updates deterministic and avoids a download. A Pixi or
    Mainboard environment is replaceable workspace output the public launcher must not depend on;
    a missing interpreter is likewise left for uv to replace.
    """
    if not declared:
        return None
    interpreter = Path(declared)
    generated = {".mainboard", ".pixi"} & {part.casefold() for part in interpreter.parts}
    return None if generated or not interpreter.is_file() else interpreter


def digest(source: Path) -> str:
    """One cheap digest of runtime source and the package's pyproject, with contents unread."""
    package = source.parent
    files = sorted(
        path for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )
    if (metadata := package / "pyproject.toml").is_file():
        files.append(metadata)
    fingerprint = hashlib.sha256()
    for path in files:
        stat = path.stat()
        line = f"{path.relative_to(package)}:{stat.st_size}:{stat.st_mtime_ns}\n"
        fingerprint.update(line.encode())
    return fingerprint.hexdigest()


def tool_root(package: Path) -> Path | None:
    """The uv tool directory holding the imported `package`'s snapshot, None from a checkout."""
    return next((parent for parent in package.parents if (parent / _RECEIPT).is_file()), None)


def _marker(receipt: Path) -> str:
    """The identity of one install, so a reinstall invalidates what was recorded for the last."""
    return f"{Project().name}:{receipt.stat().st_mtime_ns}"


def _recorded(state: Path, *, marker: str, current: str) -> str:
    """The digest recorded for this install, `current` recorded fresh on a new install.

    A state file missing, torn or from another install is replaced, so the first run after an
    install is the baseline. An unwritable tool directory answers fresh rather than failing the
    command that asked.
    """
    with suppress(OSError, json.JSONDecodeError):
        held = json.loads(state.read_text(encoding="utf-8"))
        if held.get("marker") == marker:
            return str(held.get("digest", ""))
    with suppress(OSError):
        state.write_text(json.dumps({"marker": marker, "digest": current}), encoding="utf-8")
    return current
