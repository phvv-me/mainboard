# The spool a kept line shares between the center and its compute node, on the shared filesystem.
#
# Compute nodes refuse ssh, even from the login node, so the filesystem is the only path between a
# submission and the node that runs it. Every file is created under a unique temporary name and
# published by link(2), which fails when the name exists: a claim has exactly one winner, nothing
# is read half-written, and no lock can go stale. The center renders the same layout as shell for
# the login node (no Python there); the node reads and writes it through `Spool`.

import os
import secrets
import shlex
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import mkstemp
from typing import ClassVar

from patos import FrozenModel

from ..runtime.job import walltime_seconds
from .shared import state_dir

_VERSION = 1


class Submission(FrozenModel):
    """One queued job, immutable once published.

    label: the creation intent the dispatch registry holds for it, to reconcile a lost reply.
    script / cwd: the staged job script and the pinned tree it runs from.
    """

    version: int = _VERSION
    handle: str
    label: str
    script: str
    cwd: str
    walltime: str
    submitted: str


class Claim(FrozenModel):
    """Who owns a submission, written by the exclusive creation that makes the claim.

    cancelled: the center fenced the entry before any allocation took it, so it never runs.
    """

    version: int = _VERSION
    gen: int = 0
    alloc: str = ""
    node: str = ""
    pid: int = 0
    at: float = 0.0
    cancelled: bool = False


class LineSpec(FrozenModel):
    """What `host hold` asked for, the file that makes a line exist.

    walltime: the length of each allocation, `HH:MM:SS`.
    deadline: epoch seconds after which nothing is renewed, claimed or run.
    tool: the mainboard command the allocation runs `host serve` with.
    grace: what a job needs beyond its walltime inside an allocation, the runner's start and
        its kill.
    """

    grace: ClassVar[int] = 120
    version: int = _VERSION
    queue: str
    walltime: str
    mem_gb: int
    account: str
    deadline: float
    created: float
    tool: str


class Beat(FrozenModel):
    """An allocation's last word about itself."""

    at: float
    state: str
    running: str = ""


def handle() -> str:
    """A new submission id: epoch seconds in hex, so names sort by age, and three random bytes."""
    return f"h{int(time.time()):08x}{secrets.token_hex(3)}"


def spool_path(root: str) -> str:
    """The spool directory under a host workspace root.

    A pinned tree reaches the same one through its state-directory link, as its logs do.
    """
    return f"{root}/{state_dir()}/hold"


class Remote:
    """The spool at `path` on a host, as the shell the login node runs for the center."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.logs = f"{path.rsplit('/', 1)[0]}/logs"

    def put(self, directory: str, *, name: str) -> str:
        """Publish stdin as `directory/name` unless it exists; exit 0 created, 3 existing."""
        return "\n".join(
            (
                "umask 077",
                f"d={shlex.quote(f'{self.path}/{directory}')}",
                'mkdir -p "$d" && t=$(mktemp "$d/.tmp.XXXXXX") || exit 4',
                'cat > "$t"',
                f'if ln "$t" "$d"/{shlex.quote(name)} 2>/dev/null; then r=0; else r=3; fi',
                'rm -f "$t"',
                "exit $r",
            )
        )

    def write(self, name: str) -> str:
        """Replace `name` with stdin, atomically: a reader sees the old file or the new one."""
        return "\n".join(
            (
                "umask 077",
                f"d={shlex.quote(self.path)}",
                'mkdir -p "$d" && t=$(mktemp "$d/.tmp.XXXXXX") || exit 4',
                'cat > "$t" && mv -f "$t" "$d"/' + shlex.quote(name),
            )
        )

    def status(self) -> str:
        """The line's own files, one tagged row each.

        `T` login-node clock, `L` the line, `S` stopped, `K` the keeper, `B` its last beat, then
        the newest generation's `G`, `alloc`, `node`, `beat`, `ended`.
        """
        return "\n".join(
            (
                f"cd {shlex.quote(self.path)} || exit 4",
                'echo "T $(date +%s)"',
                '[ -f line.json ] && echo "L $(tr -d "\\n" < line.json)"',
                "[ -e stop ] && echo S",
                '[ -f keeper.json ] && echo "K $(tr -d "\\n" < keeper.json)"',
                '[ -f keeper.beat ] && echo "B $(tr -d "\\n" < keeper.beat)"',
                "g=$(ls gen 2>/dev/null | sort -n | tail -1)",
                'if [ -n "$g" ]; then',
                '  echo "G $g"',
                "  for f in alloc node beat ended; do",
                '    [ -f gen/"$g"/$f ] && echo "$f $(tr -d "\\n" < gen/"$g"/$f)"',
                "  done",
                "fi",
                "exit 0",
            )
        )

    def read(self, handles: Sequence[str]) -> str:
        """One tagged row per fact about each handle, then the line's.

        `T` login-node clock, `L` the line file, `S` stopped, `E` the runner's exit artifact,
        `C` the claim, `Q` still queued, `X` tombstoned. Only files named by a handle are opened.
        """
        logs = shlex.quote(self.logs)
        named = " ".join(shlex.quote(name) for name in handles)
        return "\n".join(
            (
                f"cd {shlex.quote(self.path)} || exit 4",
                'echo "T $(date +%s)"',
                '[ -f line.json ] && echo "L $(tr -d "\\n" < line.json)"',
                "[ -e stop ] && echo S",
                f"for h in {named}; do",
                f'  [ -f {logs}/"$h".exit ] && echo "E $h $(tr -d "\\n" < {logs}/"$h".exit)"',
                '  [ -f claims/"$h".json ] && echo "C $h $(tr -d "\\n" < claims/"$h".json)"',
                '  [ -f inbox/"$h".json ] && echo "Q $h"',
                '  [ -e cancel/"$h" ] && echo "X $h"',
                "done",
                "exit 0",
            )
        )

    def fence(self, name: str) -> str:
        """Tombstone `name`, then claim it as cancelled unless an allocation already did.

        Prints `exited` when the runner already finished it, `fenced` when it can no longer run,
        `claimed` when an allocation owns it and will end it on the tombstone.
        """
        claim = shlex.quote(Claim(cancelled=True, at=time.time()).model_dump_json())
        quoted = shlex.quote(name)
        return "\n".join(
            (
                f"umask 077; cd {shlex.quote(self.path)} || exit 4",
                "mkdir -p cancel claims || exit 4",
                f": > cancel/{quoted}",
                f"if [ -f {shlex.quote(self.logs)}/{quoted}.exit ]; then echo exited; exit 0; fi",
                "t=$(mktemp claims/.tmp.XXXXXX) || exit 4",
                f'printf %s {claim} > "$t"',
                f'if ln "$t" claims/{quoted}.json 2>/dev/null; then echo fenced; '
                "else echo claimed; fi",
                'rm -f "$t"',
            )
        )

    def stop(self) -> str:
        """End the line: the marker, then every unclaimed entry tombstoned and fenced."""
        claim = shlex.quote(Claim(cancelled=True, at=time.time()).model_dump_json())
        return "\n".join(
            (
                f"umask 077; cd {shlex.quote(self.path)} || exit 4",
                ": > stop",
                "mkdir -p cancel claims inbox || exit 4",
                "for f in inbox/*.json; do",
                '  [ -f "$f" ] || continue',
                '  h=$(basename "$f" .json)',
                '  : > cancel/"$h"',
                "  t=$(mktemp claims/.tmp.XXXXXX) || exit 4",
                f'  printf %s {claim} > "$t"',
                '  ln "$t" claims/"$h".json 2>/dev/null',
                '  rm -f "$t"',
                "done",
                "exit 0",
            )
        )


class Spool:
    """The spool at `path` on the machine running this code, the node's view of the layout."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.inbox, self.claims, self.cancel = (
            path / name for name in ("inbox", "claims", "cancel")
        )
        self.logs = path.parent / "logs"

    def publish(self, directory: Path, name: str, *, text: str) -> bool:
        """Create `directory/name` holding `text` unless it exists, never visible half-written."""
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = mkstemp(prefix=".tmp.", dir=directory)
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
        won = _is_linked(Path(temporary), target=directory / name)
        Path(temporary).unlink(missing_ok=True)
        return won

    def replace(self, path: Path, text: str) -> None:
        """Overwrite `path` with `text` atomically: a reader sees the old file or the new one."""
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = mkstemp(prefix=".tmp.", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(text)
        Path(temporary).replace(path)

    def line(self) -> LineSpec:
        """The line this spool serves."""
        return LineSpec.model_validate_json((self.path / "line.json").read_text(encoding="utf-8"))

    def stopped(self) -> bool:
        """Whether the line was released."""
        return (self.path / "stop").exists()

    def queued(self) -> list[Submission]:
        """Entries nobody has claimed or cancelled, oldest first."""
        waiting = []
        for entry in sorted(self.inbox.glob("h*.json")):
            if (self.claims / entry.name).exists() or (self.cancel / entry.stem).exists():
                continue
            waiting.append(Submission.model_validate_json(entry.read_text(encoding="utf-8")))
        return waiting

    def claim(self, submission: Submission, claim: Claim) -> bool:
        """Take `submission` for `claim`'s owner; False when someone else already did."""
        return self.publish(self.claims, f"{submission.handle}.json", text=claim.model_dump_json())

    def tombstoned(self, handle: str) -> bool:
        """Whether `handle` was cancelled."""
        return (self.cancel / handle).exists()

    def exit_of(self, handle: str) -> int | None:
        """The runner's recorded exit status, None until it wrote one."""
        try:
            said = (self.logs / f"{handle}.exit").read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return int(said.strip().removeprefix("exit="))

    def fits(self, submission: Submission, *, remaining: float) -> bool:
        """Whether the job and its grace fit in `remaining` seconds."""
        return walltime_seconds(submission.walltime) + LineSpec.grace <= remaining


def _is_linked(source: Path, *, target: Path) -> bool:
    """Whether `target` was created as a new name for `source`; link(2) fails if it exists."""
    try:
        os.link(source, target)
    except FileExistsError:
        return False
    return True
