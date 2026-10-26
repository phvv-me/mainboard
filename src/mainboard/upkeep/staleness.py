# Whether the installed snapshot still answers for its source, and `self update`'s reinstall.
#
# The CLI on PATH is a uv tool snapshot of the package source, so a source edit changes nothing
# until a reinstall. The uv receipt beside the snapshot names its source directory and extras, so
# the check needs no configuration. It digests source and pyproject names, sizes and mtimes (never
# contents, to stay cheap; pyproject because a runtime dependency changes what the command can
# do). The digest is recorded when `self update` finishes a reinstall; a snapshot with no record
# (installed by hand) counts as stale, since nothing proves what it was built from.

import hashlib
import json
import tomllib
from contextlib import suppress
from functools import partial
from pathlib import Path

from patos import FrozenModel
from plumbum import CommandNotFound
from plumbum.commands.processes import ProcessTimedOut

from ..core.errors import MissionError
from ..core.project import Project
from ..engines.compile.backend.pixi.engine import PixiEngine
from ..engines.compile.backend.process import Process

_RECEIPT = "uv-receipt.toml"
_STATE = "source-state.json"

# The extra a plain reinstall silently drops, so the named command always carries it.
_EXTRA = "plot"

# uv is never a host prerequisite: Pixi resolves and runs this exact package in its cached exec
# environment.
_UV = "uv=0.12.7"

# How long the reinstall itself may take before it is abandoned.
_INSTALL_SECONDS = 600.0


class Snapshot(FrozenModel):
    """What the running snapshot knows about its own source.

    installed: whether this process runs from a uv tool snapshot; a checkout has nothing to be
        stale against.
    uv: the reinstall as a bare uv argv, empty when the receipt names no source.
    source: the absolute package directory the snapshot was installed from.
    tool: the uv tool directory holding the snapshot, its receipt and state.
    digest: what the source digests to now, recorded once a reinstall from it succeeds.
    """

    installed: bool
    stale: bool = False
    detail: str = ""
    uv: tuple[str, ...] = ()
    source: Path | None = None
    tool: Path | None = None
    digest: str = ""

    @property
    def fix(self) -> tuple[str, ...]:
        """The reinstall as Pixi runs it, empty when nothing needs one."""
        return ("exec", "--spec", _UV, *self.uv) if self.stale else ()


def check(package: Path | None = None) -> Snapshot:
    """Compare the running snapshot against the source it was last reinstalled from.

    package: the installed package directory, the one holding this module when None.
    """
    root = next(
        (
            parent
            for parent in (package or Path(__file__).resolve().parents[1]).parents
            if (parent / _RECEIPT).is_file()
        ),
        None,
    )
    if root is None:
        return Snapshot(installed=False, detail="running from source")
    try:
        directory, extras, python = _receipt(root)
    except OSError, tomllib.TOMLDecodeError, KeyError, StopIteration:
        return Snapshot(installed=True, detail="the uv receipt names no source directory")
    package = root.joinpath(directory).resolve()
    source = package / "src"
    if not source.is_dir():
        return Snapshot(installed=True, detail=f"no source tree at {source}")
    tree = _digest(source)
    recorded = _recorded(root / _STATE)
    stale = recorded != tree
    return Snapshot(
        installed=True,
        stale=stale,
        detail=_detail(package, stale=stale, recorded=recorded),
        uv=_reinstall(package, extras, python),
        source=package,
        tool=root,
        digest=tree,
    )


def _detail(package: Path, *, stale: bool, recorded: str) -> str:
    """The state of `package` against the digest its snapshot was installed from."""
    if not stale:
        return f"current with {package}"
    if recorded:
        return f"the source at {package} changed since this snapshot was installed"
    return f"nothing records which state of {package} this snapshot was installed from"


def _receipt(root: Path) -> tuple[str, list[str], str | None]:
    """The source, extras and interpreter the uv receipt in `root` gives this package."""
    declared = tomllib.loads((root / _RECEIPT).read_text(encoding="utf-8"))
    requirement = next(
        entry
        for entry in declared["tool"]["requirements"]
        if entry.get("name") == Project().package and "directory" in entry
    )
    extras = requirement.get("extras") or [_EXTRA]
    return requirement["directory"], extras, declared["tool"].get("python")


def _reinstall(package: Path, extras: list[str], python: str | None) -> tuple[str, ...]:
    """The uv argv reinstalling `package` with `extras` into the snapshot's own tool directory."""
    interpreter = _durable_interpreter(python)
    return (
        "uv",
        "tool",
        "install",
        "--reinstall-package",
        Project().package,
        *(("--python", str(interpreter)) if interpreter else ()),
        "--from",
        f"{package}[{','.join(extras)}]",
        Project().package,
        "--force",
    )


def update(found: Snapshot) -> str:
    """Reinstall a stale snapshot from its source and record the digest it was built from."""
    try:
        result = PixiEngine().within_cwd(
            partial(Process.capture, timeout=_INSTALL_SECONDS), *found.fix
        )
    except (CommandNotFound, MissionError, OSError, ProcessTimedOut) as fault:
        raise MissionError(f"could not update: {' '.join(str(fault).split())[:200]}") from fault
    if not result.succeeded:
        said = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
        raise MissionError(f"could not update: {(said or [f'exit {result.returncode}'])[-1]}")
    _record(found)
    return f"updated from {found.source}"


def _record(found: Snapshot) -> None:
    """Record that the snapshot in `found.tool` was just installed from `found.digest`."""
    if found.tool is not None:
        (found.tool / _STATE).write_text(
            json.dumps({"digest": found.digest}), encoding="utf-8", newline="\n"
        )


def _recorded(state: Path) -> str:
    """The digest recorded by the last reinstall, empty when none was."""
    with suppress(OSError, json.JSONDecodeError, AttributeError):
        return str(json.loads(state.read_text(encoding="utf-8")).get("digest", ""))
    return ""


def _durable_interpreter(declared: str | None) -> Path | None:
    """An existing uv-tool interpreter that does not belong to generated project state.

    Reusing uv's managed-Python path keeps updates deterministic and avoids a download. A Pixi or
    Mainboard environment is replaceable workspace output the public launcher must not depend on;
    a missing interpreter is likewise left for uv to replace.
    """
    if not declared:
        return None
    interpreter = Path(declared)
    generated = {*Project().out_dirs, ".pixi"} & {part.casefold() for part in interpreter.parts}
    return None if generated or not interpreter.is_file() else interpreter


def _digest(source: Path) -> str:
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
