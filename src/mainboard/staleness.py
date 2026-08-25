# The installed snapshot's own freshness. The CLI on PATH is a uv tool snapshot of the package
# source, so an edit to that source silently changes nothing until someone reinstalls, and the
# trap has bitten enough times to earn a standing check. The uv receipt beside the installed
# environment says exactly which directory the snapshot was built from and with which extras, so
# the check needs no configuration: digest the source tree, remember what it looked like when
# this snapshot first ran, and say one line the moment the tree moves past it.
#
# The digest is over names, sizes and mtimes rather than contents, which keeps the whole check
# in the low milliseconds a CLI startup can afford, and it is recorded on the snapshot's first
# run rather than at install because uv owns the install and offers no hook. The one blind spot
# that buys is an edit landing between the install and the first run, which the next reinstall
# clears.

import hashlib
import json
import tomllib
from contextlib import suppress
from pathlib import Path

from patos import FrozenModel

from .core.project import Project

# The file uv writes beside every tool it installs, naming the source of the snapshot.
_RECEIPT = "uv-receipt.toml"

# Where this check remembers the source tree the running snapshot answered for.
_STATE = "source-state.json"

# The extra a plain reinstall silently drops, so the named command always carries it.
_EXTRA = "wandb"


class Snapshot(FrozenModel):
    """What the running snapshot knows about its own source.

    installed: whether this process runs from a uv tool snapshot at all; a checkout running
        its own source has nothing to be stale against.
    stale: whether the source tree has moved past what this snapshot was recorded against.
    detail: the one line behind the answer.
    fix: the reinstall command that refreshes the snapshot, empty when nothing needs one.
    """

    installed: bool
    stale: bool = False
    detail: str = ""
    fix: str = ""

    @property
    def warning(self) -> str:
        """The one line a CLI invocation prints, empty when there is nothing to say."""
        if not self.stale:
            return ""
        return f"{Project().name}: {self.detail}; reinstall: {self.fix}"


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
    source = root.joinpath(requirement["directory"]).resolve() / "src"
    if not source.is_dir():
        return Snapshot(installed=True, detail=f"no source tree at {source}")
    extras = ",".join(requirement.get("extras") or [_EXTRA])
    fix = f"uv tool install --from '{requirement['directory']}[{extras}]' {Project().name} --force"
    current = digest(source)
    recorded = _recorded(root / _STATE, marker=_marker(receipt), current=current)
    if recorded == current:
        return Snapshot(installed=True, detail="snapshot matches the source tree")
    return Snapshot(
        installed=True,
        stale=True,
        detail=f"the source at {source.parent} is newer than this installed snapshot",
        fix=fix,
    )


def digest(source: Path) -> str:
    """One cheap digest of a source tree: every file's path, size and mtime, contents unread.

    source: the tree to fingerprint.
    """
    fingerprint = hashlib.sha256()
    files = sorted(
        path for path in source.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        stat = path.stat()
        line = f"{path.relative_to(source)}:{stat.st_size}:{stat.st_mtime_ns}\n"
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
