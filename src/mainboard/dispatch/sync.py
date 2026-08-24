# Mirror a workspace onto a host: the `.gitignore` filter and the `rsync` command builder.

import fcntl
import hashlib
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Self, TextIO

import pathspec
from patos import StrFlag
from plumbum import CommandNotFound, local
from plumbum.commands.processes import ProcessExecutionError

from .shared import logger, state_dir, state_path
from .transport import HostUnreachable, is_transport_failure

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

# Always skipped regardless of `.gitignore`. `.git` has no trailing slash so it also matches
# a submodule's `.git` file, not just the superproject's `.git/` directory.
ALWAYS_EXCLUDE = (".git", ".env", f"{state_dir()}/", ".mainboard/", ".pixi/", "__pycache__/")

# macOS ships Apple's openrsync as /usr/bin/rsync; upstream rsync usually arrives via Homebrew
# or MacPorts at these roots, searched after PATH.
_UPSTREAM_FALLBACKS = ("/opt/homebrew/bin/rsync", "/usr/local/bin/rsync", "/opt/local/bin/rsync")


def binary(*, mirror: bool) -> str:
    """The local rsync to run, preferring upstream rsync over Apple's openrsync.

    openrsync cannot prune inside directories that exist on both ends of a remote `--relative`
    transfer, so a remote mirror silently keeps the stale files it was meant to remove. PATH is
    searched first, then the Homebrew/MacPorts roots.
    mirror: whether the transfer prunes a remote end (`--delete`), which demands upstream rsync
        and raises when only openrsync is installed.
    """
    for candidate in ("rsync", *_UPSTREAM_FALLBACKS):
        try:
            version = local[candidate]("--version")
        except CommandNotFound, OSError:
            continue
        if "openrsync" not in version:
            return candidate
    if mirror:
        raise RuntimeError(
            "mirroring needs upstream rsync, but only Apple's openrsync is installed "
            "and it cannot prune stale files on a remote host. Run `brew install rsync`"
        )
    return "rsync"


class Rsync(StrFlag):
    """rsync switches; OR-combine them and each member carries its literal flag."""

    ARCHIVE = "-a"  # recurse + preserve perms/times/symlinks/...
    COMPRESS = "-z"
    RELATIVE = "-R"  # recreate each source path under dest
    RECURSIVE = "-r"
    VERBOSE = "-v"
    DRY_RUN = "-n"
    CHECKSUM = "-c"  # compare by checksum, not size+mtime
    UPDATE = "-u"  # skip files newer on the receiver
    LINKS = "-l"
    PERMS = "-p"
    TIMES = "-t"
    HUMAN = "-h"
    DELETE = "--delete"  # mirror removals
    DELETE_AFTER = "--delete-after"  # apply transferred per-directory filters before pruning
    PARTIAL = "--partial"  # keep partially transferred files
    PROGRESS = "--progress"
    STATS = "--stats"


def _rsync_args(
    flags: Rsync | Sequence[Rsync],
    paths: Sequence[str],
    *,
    include: Sequence[str],
    exclude: Sequence[str],
    protect: Sequence[str],
    filters: Sequence[str],
    rsh: str | None,
    bwlimit: int | None,
    timeout: int | None,
    extra: Sequence[str],
) -> list[str]:
    """The rsync argv: combined flags, then filter rules in receiver order, then paths."""
    members = [*flags]
    short = "".join(member.string[1] for member in members if len(member.string) == 2)
    args: list[str] = [f"-{short}"] if short else []
    args += [member.string for member in members if member.string.startswith("--")]
    if rsh is not None:
        args += ["-e", rsh]
    if bwlimit is not None:
        args.append(f"--bwlimit={bwlimit}")
    if timeout is not None:
        args.append(f"--timeout={timeout}")
    for pattern in protect:
        args += ["--filter", f"protect {pattern}"]
    for pattern in include:
        args += ["--include", pattern]
    for rule in filters:
        args += ["--filter", rule]
    for pattern in exclude:
        args += ["--exclude", pattern]
    args += [*extra, *paths]
    return args


def rsync(
    sources: str | Sequence[str],
    dest: str,
    flags: Rsync | Sequence[Rsync] = Rsync.ARCHIVE | Rsync.COMPRESS,
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    protect: Sequence[str] = (),
    filters: Sequence[str] = (),
    rsh: str | None = None,
    bwlimit: int | None = None,
    timeout: int | None = None,
    extra: Sequence[str] = (),
    allow_vanished: bool = True,
    host: str = "",
    cwd: Path | None = None,
) -> str:
    """Run `rsync` locally (it connects to remote hosts itself); return its stdout.

    The binary comes from `binary`, which prefers upstream rsync and refuses to mirror a remote
    end with Apple's openrsync.
    sources: one path or many. dest: `host:/path/` or a local dir.
    flags: combined `Rsync` flag or sequence of members; single-letter ones merge into one `-azR`
        group.
    include / exclude: filter patterns emitted before and after `filters`.
    protect: receiver-side `protect` filter rules emitted before include/exclude, shielding
        remote-only paths from `--delete` pruning.
    filters: ordered rsync filter rules, such as Git ignore merge rules.
    rsh: remote shell (`-e`). bwlimit: KB/s cap. timeout: seconds.
    extra: raw flags for anything not covered above.
    allow_vanished: accept rsync code 24 for ordinary changing mirrors. Required job-script
        transfers disable this so submission cannot continue after a partial sync.
    cwd: the directory relative source paths are read from, this process's own when None. A
        workspace mirror passes its root, since `--relative` rebuilds each source path under the
        destination and a path read from a subdirectory would name a different file or none.
    """
    paths = [*([sources] if isinstance(sources, str) else sources), dest]
    mirror = Rsync.DELETE in [*flags] and any(":" in path for path in paths)
    args = _rsync_args(
        flags,
        paths,
        include=include,
        exclude=exclude,
        protect=protect,
        filters=filters,
        rsh=rsh,
        bwlimit=bwlimit,
        timeout=timeout,
        extra=extra,
    )
    command = local[binary(mirror=mirror)][args]
    if cwd is not None:
        command = command.with_cwd(cwd)
    try:
        output = str(command())
    except ProcessExecutionError as error:
        output = _absorb_or_raise(error, host=host, allow_vanished=allow_vanished)
    else:
        _log_deletions(output)
    return output


def _absorb_or_raise(error: ProcessExecutionError, *, host: str, allow_vanished: bool) -> str:
    """The partial stdout for an absorbed transport blip or a code-24 vanished-file mirror.

    Every other failure re-raises, a translated `HostUnreachable` for a transport-shaped one so
    a dead host reads the same way every scheduler backend already reports it.
    """
    output = str(error.stdout)
    _log_deletions(output)
    stderr = str(error.stderr)
    if host and (is_transport_failure(error.retcode, stderr) or error.retcode == 30):
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        detail = next(
            (
                line
                for line in lines
                if "connection" in line.lower() or "timed out" in line.lower()
            ),
            lines[-1] if lines else "transport timed out",
        )
        raise HostUnreachable(f"rsync to {host!r} failed: {detail}") from error
    if error.retcode != 24 or not allow_vanished:
        raise error
    return output


def _log_deletions(output: str) -> None:
    """Note every receiver path rsync reports deleting, one warning per invocation."""
    deleted = [
        line.removeprefix("deleting ")
        for line in output.splitlines()
        if line.startswith("deleting ")
    ]
    if deleted:
        logger.warning("rsync deleted %d path(s): %s", len(deleted), ", ".join(deleted))


class SyncLock:
    """Serialize destructive mirrors to one target across local dispatch processes.

    target: SSH alias whose remote tree is being mirrored.
    root: local workspace root that owns the generated state directory, discovered upward from
        the current directory when None, so two processes started in different subdirectories
        still queue behind the same lock file.
    """

    def __init__(self, target: str, root: Path | None = None) -> None:
        digest = hashlib.blake2s(target.encode(), digest_size=8).hexdigest()
        self.path = state_path(root) / "locks" / f"sync-{digest}.lock"
        self.file: TextIO | None = None

    def __enter__(self) -> Self:
        """Wait for this target's mirror lock and hold it until context exit."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as undo:
            file = undo.enter_context(self.path.open("a+"))
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            undo.pop_all()
        self.file = file
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the kernel lock even when rsync raises."""
        del exc_type, exc_value, traceback
        file = self.file
        if file is None:
            return
        self.file = None
        with file:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class GitignoreFilter:
    """Git ignore rules for rsync.

    Rsync reads the workspace root ignore file once as a global rule set, then discovers every
    nested `.gitignore` as it descends. This preserves each file's directory context and lets
    rsync apply the same excludes while pruning the receiver.

    root: the repo whose `.gitignore` drives the denylist; defaults to the current working
        directory, the repo you dispatch from.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        gitignore = self.root / ".gitignore"
        lines = self.__lines(gitignore)
        # pyrefly: ignore  reason=pathspec from_lines stub over-narrows to AnyStr since=2026-08-16
        self.spec = pathspec.GitIgnoreSpec.from_lines(lines)
        self.filters = [
            *(["merge,- .gitignore"] if gitignore.is_file() else []),
            ":- .gitignore",
        ]
        self.excludes = list(ALWAYS_EXCLUDE)

    def control_files(self, sources: Sequence[str]) -> list[str]:
        """Ignore files above source roots that the receiver needs before deletion.

        A narrow source such as `research/projects` inherits `research/.gitignore` on the
        sender, but that parent file is outside the transferred subtree. Shipping each existing
        ancestor from the workspace root downward lets `--delete-after` apply the same
        directory-relative rules on the receiver.
        """
        files: list[str] = []
        seen: set[Path] = set()
        for source in sources:
            path = Path(source)
            if path.is_absolute():
                path = path.relative_to(self.root)
            ancestors = reversed((path.parent, *path.parent.parents))
            for ancestor in ancestors:
                gitignore = ancestor / ".gitignore"
                if gitignore in seen or not (self.root / gitignore).is_file():
                    continue
                seen.add(gitignore)
                files.append(str(gitignore))
        return files

    def ignored(self, path: str | Path) -> bool:
        """Whether `path` (absolute or repo-relative) is git-ignored."""
        candidate = Path(path)
        if candidate.is_absolute() and candidate.is_relative_to(self.root):
            candidate = candidate.relative_to(self.root)
        return self.spec.match_file(candidate)

    @staticmethod
    def __lines(gitignore: Path) -> list[str]:
        return gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
