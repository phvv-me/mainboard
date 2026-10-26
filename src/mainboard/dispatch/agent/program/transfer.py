"""The requests a target carries out: survey what it holds, receive a mirror, pin a tree; and
`main`, which reads one request from standard input and answers on standard output."""

import hashlib
import json
import os
import stat
import tarfile
import tempfile
import threading
from contextlib import suppress
from typing import IO, TYPE_CHECKING, TypedDict

from .constants import CHUNK
from .exceptions import Refusal
from .snapshot import PinSpec, Snapshot, locked
from .walking import Digests, Entry, Scope, ScopeSpec, checked, digested, native, walk

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from .walking import Json


class SurveySpec(TypedDict):
    """What a target is asked to describe: its scopes walked and its named files stated."""

    root: str
    state: str
    scopes: list[ScopeSpec]
    named: list[str]


class ReceiveSpec(TypedDict):
    """What a target changes before and while the tar stream arrives.

    files: each file the stream carries, as `[mtime_ns, executable]`, executable False when the
        center has no mode bits to give.
    """

    root: str
    state: str
    delete: list[str]
    directories: list[str]
    links: dict[str, str]
    files: dict[str, tuple[int, bool]]


class Request(TypedDict, total=False):
    """One request, naming its operation by the one key it carries."""

    survey: SurveySpec
    receive: ReceiveSpec
    pin: PinSpec


# How often a working request tells the center it is alive, well inside its patience.
_PULSE_SECONDS = 10.0


class Emitter:
    """Answer records to the center, one JSON line each, flushed so a quiet link means a stall.

    While a request works, a blank line goes out every `PULSE_SECONDS`: pinning a whole mirror
    image can take minutes without a record to send, and the center read the silent link as a
    dead host (gold, 2026-09-29). The center counts the line as movement and keeps no record.
    """

    def __call__(self, record: Json) -> None:
        line = json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
        with self.lock:
            self.stdout.write(line)
            self.stdout.flush()

    def __init__(self, stdout: IO[bytes]) -> None:
        self.stdout = stdout
        self.lock = threading.Lock()
        self.done = threading.Event()

    def pulse(self) -> None:
        """Write the blank line every `PULSE_SECONDS` until the request is done."""
        while not self.done.wait(_PULSE_SECONDS):
            with self.lock:
                self.stdout.write(b"\n")
                self.stdout.flush()


def _survey(spec: SurveySpec, emit: Emitter) -> None:
    """Describe what this end holds: its capabilities first, then every scoped and named entry.

    The root and its state directory are made on the way, since a fresh machine holds neither and
    this is the first thing a mirror asks of it.
    """
    root = spec["root"]
    state = native(root, relative=spec["state"])
    os.makedirs(state, exist_ok=True)
    emit({"fold": _folds(state)})
    digests = Digests(os.path.join(state, "digests.json"))
    for entry in _held(spec):
        if (found := digested(entry, root, digests)) is not None:
            emit(list(found))
    digests.save(prune=True)


def _held(spec: SurveySpec) -> Iterator[Entry]:
    """Every entry the scopes cover under the root, then every named path that is a file there."""
    root = spec["root"]
    for scope in spec["scopes"]:
        yield from walk(root, Scope.of(scope))
    for relative in spec["named"]:
        with suppress(FileNotFoundError):
            status = os.lstat(native(root, relative=checked(relative)))
            if stat.S_ISREG(status.st_mode):
                yield Entry.stated(relative, status)


def _carried(request: Request, stdin: IO[bytes], emit: Emitter) -> None:
    """Carry out `request`: a survey, a receive reading its stream off `stdin`, or a pin."""
    if "survey" in request:
        _survey(request["survey"], emit)
    elif "receive" in request:
        _receive(request["receive"], stdin, emit)
    else:
        emit({"path": Snapshot(request["pin"]).pinned()})


def _folds(directory: str) -> bool:
    """Whether names in `directory` compare without case, probed with a file of its own."""
    handle, probe = tempfile.mkstemp(prefix=".mainboard-Case-", dir=directory)
    os.close(handle)
    head, name = os.path.split(probe)
    try:
        return os.path.exists(os.path.join(head, name.swapcase()))
    finally:
        os.remove(probe)


def _receive(spec: ReceiveSpec, stream: IO[bytes], emit: Emitter) -> None:
    """Apply one mirror under this end's mirror lock: prune, make, link, then unpack the stream.

    Pruning comes first so a name that only changed case lands on a file system that folds case,
    and every file is written beside its final name and renamed over it, so a reader holding the
    old file keeps the bytes it opened.
    """
    root = spec["root"]
    state = native(root, relative=spec["state"])
    os.makedirs(state, exist_ok=True)
    with locked(os.path.join(state, "mirror.lock")):
        deleted, kept = _prune(root, spec["delete"])
        for relative in spec["directories"]:
            os.makedirs(native(root, relative=checked(relative)), exist_ok=True)
        for relative, target in spec["links"].items():
            path = native(root, relative=checked(relative))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.symlink(target, path)
        digests = Digests(os.path.join(state, "digests.json"))
        written, size = _unpack(root, stream, spec["files"], digests)
        digests.save()
    emit({"written": written, "bytes": size, "deleted": deleted, "kept": kept})


def _prune(root: str, paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """Remove `paths` deepest first; a directory still holding what the rules keep stays."""
    deleted: list[str] = []
    kept: list[str] = []
    for relative in sorted(paths, key=lambda path: path.count("/"), reverse=True):
        path = native(root, relative=checked(relative))
        remove = os.rmdir if os.path.isdir(path) and not os.path.islink(path) else os.remove
        try:
            remove(path)
        except FileNotFoundError:
            continue
        except OSError:
            kept.append(relative)
            continue
        deleted.append(relative)
    return deleted, kept


def _unpack(
    root: str, stream: IO[bytes], files: Mapping[str, tuple[int, bool]], digests: Digests
) -> tuple[int, int]:
    """Write every file the stream carries, refusing one the request did not announce."""
    written = size = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            relative = checked(member.name)
            source = archive.extractfile(member)
            if relative not in files or source is None:
                raise Refusal(f"unannounced entry in the stream: {relative!r}")
            mtime, executable = files[relative]
            path = native(root, relative=relative)
            placed = _place(path, source, mtime=mtime, executable=executable)
            digests.note(path, key=relative, found=placed)
            written += 1
            size += member.size
    return written, size


def _place(path: str, source: IO[bytes], *, mtime: int, executable: bool) -> str:
    """Write `source` beside `path`, rename it into place, and answer the digest of its bytes.

    The file is created with the mode its execute bit asks for, which this host's umask then
    narrows.
    """
    staged = os.path.join(
        os.path.dirname(path), f".{os.path.basename(path)}.mainboard-{os.getpid()}"
    )
    try:
        return _written(
            source, staged=staged, path=path, mtime=mtime, mode=0o777 if executable else 0o666
        )
    except OSError as fault:
        if os.path.lexists(staged):
            os.remove(staged)
        raise Refusal(f"could not place {path}: {fault}") from fault


def _written(source: IO[bytes], *, staged: str, path: str, mtime: int, mode: int) -> str:
    """`source` written to `staged` under `mode` and `mtime`, renamed over `path`; its digest."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hashed = hashlib.sha256()
    with os.fdopen(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "wb") as out:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            hashed.update(chunk)
            out.write(chunk)
    os.utime(staged, ns=(mtime, mtime))
    os.replace(staged, path)
    return hashed.hexdigest()


def main(stdin: IO[bytes], *, stdout: IO[bytes], stderr: IO[str]) -> int:
    """Carry out the one request `stdin` holds, answering records on `stdout`; the exit status.

    A refusal is one `mainboard:` line on `stderr` and status 3; anything else escapes as itself.
    """
    request: Request = json.loads(stdin.readline())
    emit = Emitter(stdout)
    threading.Thread(target=emit.pulse, daemon=True).start()
    try:
        _carried(request, stdin, emit)
    except Refusal as refusal:
        stderr.write(f"mainboard: {refusal}\n")
        stderr.flush()
        return 3
    finally:
        emit.done.set()
    return 0
