"""Pinned source trees: the mirrored or sealed image a job runs from, built once under a
host's pin lock and verified on every reuse."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypedDict

from .constants import CLOSURE, DIRECTORY, LINK, NAMES, PINS, RESERVED, STAMP, WRAPPERS
from .exceptions import Refusal
from .walking import (
    Entry,
    Scope,
    ScopeSpec,
    ancestors,
    checked,
    is_unsafe,
    native,
    sha256_of,
    walk,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence


def _marked(suffix: str) -> tuple[str, ...]:
    """A control file `.<name>-<suffix>` under every name, primary first."""
    return tuple(f".{name}-{suffix}" for name in NAMES)


class ImageSpec(TypedDict, total=False):
    """What a snapshot copies: a mirrored scope, or a sealed closure listing and its needs."""

    kind: str
    scope: ScopeSpec
    listing: str
    digest: str
    needs: list[str]
    pins: list[str]
    staging: str
    live: list[str]


class PinSpec(TypedDict):
    """Everything one pin needs, resolved on the center down to host path strings."""

    root: str
    base: str
    key: str
    stamp: str
    out: str
    prefix: str
    environment: str
    results: str
    script: str
    wrapper: str
    image: ImageSpec


def closure_path(pinned: str) -> str:
    """Where the pin at `pinned` keeps the listing of the closure it runs."""
    return f"{pinned}/{CLOSURE}"


def wrapper_path(name: str) -> str:
    """Where a pin keeps the staged job wrapper `name`."""
    return f"{WRAPPERS}/{name}"


def reserved(relative: str) -> bool:
    """Whether `relative` names one of a pinned tree's control files, under any of its names."""
    wrappers = tuple(f"{name}/" for name in _marked("jobs"))
    return relative in RESERVED or relative.startswith(wrappers)


def _present(directory: str, names: Sequence[str], written: str) -> str:
    """The first of `names` `directory` holds, else where `written`, its written name, would be."""
    found = (os.path.join(directory, name) for name in names)
    return next(
        (path for path in found if os.path.lexists(path)), os.path.join(directory, written)
    )


@contextmanager
def locked(path: str) -> Generator[None]:
    """Hold an exclusive kernel lock on `path` for the block, released however it ends.

    A kernel lock rather than a lock file's mere existence, so a process killed mid-transfer
    leaves nothing behind that the next one would have to be told about.
    """
    with open(path, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _pending(directory: str) -> Generator[str]:
    """A fresh file in `directory` for the block to fill and rename away, else removed."""
    handle, pending = tempfile.mkstemp(prefix=".pending.", dir=directory)
    os.close(handle)
    try:
        yield pending
    finally:
        if os.path.lexists(pending):
            os.remove(pending)


def _remove(path: str) -> None:
    """Remove whatever stands at `path`, a link itself rather than what it points at."""
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _relink(source: str, *, path: str) -> None:
    """Point a link at `path` to `source`, its parent made and a link already there replaced."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.islink(path):
        os.remove(path)
    os.symlink(source, path)


def _hardlink(source: str, *, path: str) -> None:
    """Share `source`'s inode at `path`, or copy it where the file system cannot share one."""
    try:
        os.link(source, path)
    except OSError:
        shutil.copy2(source, path)


class Mirrored:
    """The whole synced scope, what a command that ships the mirror runs from.

    scope: the roots the mirror shipped and the rules it shipped them by, so the snapshot holds
        the shipped file set and not the artifacts the host wrote beside it.
    """

    @staticmethod
    def _carried(entry: Entry, *, source: str, placed: str) -> None:
        """A link `entry` made again at `placed`, a file hardlinked there from `source`."""
        if entry.kind == LINK:
            os.symlink(entry.detail, placed)
        else:
            _hardlink(source, path=placed)

    def __init__(self, spec: ImageSpec) -> None:
        self.scope = Scope.of(spec["scope"])

    def copy(self, root: str, *, snap: str) -> None:
        """Hardlink the shipped set out of the mirror, links carried as links."""
        missing = [
            top for top in self.scope.roots if not os.path.lexists(native(root, relative=top))
        ]
        if missing:
            raise Refusal(f"missing source: {missing[0]}")
        for entry in walk(root, self.scope):
            placed = native(snap, relative=entry.path)
            os.makedirs(os.path.dirname(placed), exist_ok=True)
            if entry.kind == DIRECTORY:
                os.makedirs(placed, exist_ok=True)
            else:
                Mirrored._carried(entry, source=native(root, relative=entry.path), placed=placed)

    def fill(self, root: str, *, snap: str) -> None:
        """Link back whatever the mirror holds in each directory the copy made and did not fill."""
        # Every directory the copy made to hold the roots, the tree root included: links back to
        # the mirror are how a job reaches the environment, the dispatch state and the data
        # beside its source uncopied.
        roots = self.scope.roots
        for directory in sorted({".", *(folder for top in roots for folder in ancestors(top))}):
            source, target = native(root, relative=directory), native(snap, relative=directory)
            os.makedirs(target, exist_ok=True)
            for name in sorted(os.listdir(source)):
                entry, placed = os.path.join(source, name), os.path.join(target, name)
                if os.path.exists(entry) and not os.path.lexists(placed):
                    os.symlink(entry, placed)

    def link(self, root: str, *, snap: str) -> None:
        """Nothing: a mirrored tree already reaches everything the mirror holds."""

    def verify(self, snap: str) -> None:
        """Nothing: a mirrored tree carries no listing to check."""


class Sealed:
    """A job's closure and nothing beside it, what a job spelled by file runs from.

    listing: the closure listing on the mirror, whose first column names every shipped file and
        whose second holds each file's SHA-256.
    digest: the listing's own SHA-256, which the frozen copy must match.
    needs / pins: data paths linked back to the mirror on every dispatch, pins reached through
        their shared `staging` directory.
    live: every path linked back on this dispatch, which no listed file may sit beneath.
    """

    @staticmethod
    def __placed(snap: str, *, need: str) -> str:
        """Where `need` lands in the tree.

        A parent reached through a link is refused, since placing the need there would replace the
        mirror's own file.
        """
        placed = native(snap, relative=need)
        os.makedirs(os.path.dirname(placed), exist_ok=True)
        inside = os.path.normpath(
            native(os.path.realpath(snap), relative=os.path.dirname(need) or ".")
        )
        if os.path.realpath(os.path.dirname(placed)) != inside:
            raise Refusal(f"the need {need} sits under a path already linked to the mirror")
        return placed

    def __fault(self, snap: str, *, real: str, relative: str, blob: str) -> str:
        """Why the listed file `relative` cannot stand in the tree at `snap`, empty when it can.

        real: the tree's own path with every link resolved, which the file's must sit under.
        """
        path = native(snap, relative=relative)
        overlapping = [
            live for live in self.live if relative == live or relative.startswith(live + "/")
        ]
        if overlapping:
            return f"live path overlaps source: {overlapping[0]}"
        if os.path.islink(path) or not os.path.isfile(path):
            return f"missing or linked source: {relative}"
        if os.path.realpath(path) != native(real, relative=relative):
            return f"linked source parent: {relative}"
        return f"source blob mismatch: {relative}" if sha256_of(path) != blob else ""

    def __init__(self, spec: ImageSpec) -> None:
        self.listing = spec["listing"]
        self.digest = spec["digest"]
        self.needs = spec["needs"]
        self.pins = spec["pins"]
        self.staging = spec["staging"]
        self.live = spec["live"]

    def __rows(self, frozen: str) -> Iterator[tuple[str, str]]:
        """Each listed path and blob, the listing's own digest and every path checked first."""
        if sha256_of(frozen) != self.digest:
            raise Refusal("closure listing digest mismatch")
        with open(frozen, encoding="utf-8", newline="") as listing:
            rows = [line.split("\t") for line in listing.read().split("\n") if line]
        for fields in rows:
            relative = fields[0]
            if is_unsafe(relative):
                raise Refusal(f"invalid closure path: {relative}")
            if reserved(relative):
                raise Refusal(f"reserved closure path: {relative}")
            yield relative, fields[1]

    def copy(self, root: str, *, snap: str) -> None:
        """Freeze the listing, then hardlink exactly the files it names, links resolved."""
        frozen = os.path.join(snap, CLOSURE)
        shutil.copyfile(native(root, relative=checked(self.listing)), frozen)
        for relative, _ in self.__rows(frozen):
            source = native(root, relative=relative)
            if not os.path.isfile(source):
                raise Refusal(f"missing or linked source: {relative}")
            placed = native(snap, relative=relative)
            os.makedirs(os.path.dirname(placed), exist_ok=True)
            _hardlink(os.path.realpath(source), path=placed)

    def fill(self, root: str, *, snap: str) -> None:
        """Nothing: a sealed tree reaches the mirror only through what it declared."""

    def link(self, root: str, *, snap: str) -> None:
        """Check each pin and need on the mirror and link it into the tree."""
        for need in (*self.pins, *self.needs):
            if not os.path.exists(native(root, relative=need)):
                raise Refusal(f"the need {need} is not on the mirror")
        if self.pins and not os.path.exists(native(snap, relative=self.staging)):
            _relink(native(root, relative=self.staging), path=native(snap, relative=self.staging))
        for need in self.needs:
            _relink(native(root, relative=need), path=self.__placed(snap, need=need))

    def verify(self, snap: str) -> None:
        """Check the frozen listing and every listed file's place and bytes."""
        real = os.path.realpath(snap)
        for relative, blob in self.__rows(_present(snap, _marked("closure"), CLOSURE)):
            if fault := self.__fault(snap, real=real, relative=relative, blob=blob):
                raise Refusal(fault)


class Snapshot:
    """One pinned source tree: built once under the host's pin lock, verified on every reuse.

    A build happens in a private directory that becomes the tree by one rename after its source
    verifies, so a failed build never becomes a completed snapshot, and an older incomplete tree
    refuses for inspection. The live paths, the needs, the results path and the frozen wrapper,
    belong to the dispatch rather than to the tree, so they are linked on every reuse too.
    """

    @staticmethod
    def __freeze(source: str, *, frozen: str, expected: str) -> None:
        """Copy `source` to `frozen` by a rename, refusing bytes not hashing to `expected`."""
        with _pending(os.path.dirname(frozen)) as pending:
            shutil.copy(source, pending)
            if sha256_of(pending) != expected:
                raise Refusal("wrapper digest mismatch")
            os.replace(pending, frozen)

    @staticmethod
    def __sync(directory: str) -> None:
        """Make the entries of `directory` survive a crash."""
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)

    def __build(self) -> None:
        if os.path.lexists(self.final):
            raise Refusal(f"incomplete snapshot requires inspection: {self.final}")
        pending = tempfile.mkdtemp(prefix=".pending.", dir=self.base)
        try:
            self.__made(pending)
        finally:
            if os.path.lexists(pending):
                shutil.rmtree(pending)

    def __environment(self, snap: str) -> None:
        """Point the tree's environment at the immutable prefix it names, once, at build time."""
        prefix = self.spec["prefix"]
        if not prefix:
            return
        where = native(snap, relative=f"{self.spec['out']}/envs/{self.spec['environment']}")
        os.makedirs(where, exist_ok=True)
        _remove(os.path.join(where, ".pixi"))
        os.symlink(f"{prefix}/.pixi", os.path.join(where, ".pixi"))

    def __generated(self, snap: str) -> None:
        """Rebuild the generated tree as this snapshot's own.

        The environment is the mirror's and the code is the snapshot's, and the generated tree is
        where the two meet, so its directories down to each environment are real here, its files
        hardlinked, and everything heavy beneath them a link back to the mirror. A job that
        recompiles its manifest then writes into this tree and never over the description every
        other job on the host activates through. A directory the image already copied into is
        left as the image made it.
        """
        out = self.spec["out"]
        envs = native(self.root, relative=f"{out}/envs")
        listed = sorted(os.listdir(envs)) if os.path.isdir(envs) else []
        names = [name for name in listed if os.path.isdir(os.path.join(envs, name))]
        for directory in [f"{out}/envs", *(f"{out}/envs/{name}" for name in names)]:
            os.makedirs(native(snap, relative=directory), exist_ok=True)
        for directory in [out, *(f"{out}/envs/{name}" for name in names)]:
            self.__shared(directory, snap=snap)

    def __init__(self, spec: PinSpec) -> None:
        self.spec = spec
        self.root = spec["root"]
        self.base = spec["base"]
        self.final = os.path.join(self.base, spec["key"])
        image = spec["image"]
        self.image: Mirrored | Sealed = (
            Sealed(image) if image["kind"] == "sealed" else Mirrored(image)
        )

    def __live(self, snap: str) -> None:
        self.image.link(self.root, snap=snap)
        self.__results(snap)
        self.__wrapper(snap)

    def __made(self, pending: str) -> None:
        """Build the tree in `pending`, verify it, stamp it and rename it into place."""
        self.image.copy(self.root, snap=pending)
        self.__generated(pending)
        self.image.fill(self.root, snap=pending)
        self.__environment(pending)
        self.image.verify(pending)
        self.__live(pending)
        with open(os.path.join(pending, STAMP), "wb") as stamp:
            stamp.write(self.spec["stamp"].encode("utf-8"))
        os.rename(pending, self.final)

    def __results(self, snap: str) -> None:
        """Point the dispatch's results path back at the mirror, where the pull already looks.

        A real directory the copy made there is replaced by the link. When the path's parent is
        itself reached through a link into the mirror, the path already is the mirror's and is
        left alone, since clearing it would clear the mirror's own results.
        """
        relative = self.spec["results"]
        if not relative:
            return
        os.makedirs(native(self.root, relative=relative), exist_ok=True)
        placed = native(snap, relative=relative)
        parent = os.path.dirname(placed)
        os.makedirs(parent, exist_ok=True)
        inside = native(os.path.realpath(snap), relative=os.path.dirname(relative) or ".")
        if os.path.realpath(parent) != os.path.normpath(inside):
            return
        if not os.path.islink(placed):
            _remove(placed)
        _relink(native(self.root, relative=relative), path=placed)

    def __retain(self) -> None:
        """Publish this tree and its prefix as roots to keep, before anything launches.

        The record is durable, immutable and named by its content, so a repeated pin adds
        nothing, a failed preparation leaves its pin behind and a lost submit reply loses
        nothing. The center's retention authority alone retires one; no scheduler verdict does.
        The reader is `core.retention.RetentionPins`.
        """
        prefix = self.spec["prefix"]
        pin = json.dumps(
            {
                "version": 1,
                "sources": [self.final],
                "prefixes": [prefix] if prefix else [],
                "proofs": [self.final],
            },
            sort_keys=True,
        ).encode("utf-8")
        directory = os.path.join(self.base, PINS)
        os.makedirs(directory, exist_ok=True)
        handle, pending = tempfile.mkstemp(prefix=".pending.", dir=directory)
        with os.fdopen(handle, "wb") as sink:
            sink.write(pin)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(pending, os.path.join(directory, hashlib.sha256(pin).hexdigest() + ".json"))
        # Each directory that may have gained an entry: the pins, the sources, the state, the tree.
        for _ in range(4):
            self.__sync(directory)
            directory = os.path.dirname(directory)

    def __reuse(self) -> None:
        with open(_present(self.final, _marked("source"), STAMP), "rb") as stamp:
            if stamp.read() != self.spec["stamp"].encode("utf-8"):
                raise Refusal("snapshot stamp mismatch")
        self.image.verify(self.final)
        self.__live(self.final)
        # Used again now: pruning measures a tree's age from its last dispatch, not its first.
        os.utime(self.final)

    def __shared(self, directory: str, *, snap: str) -> None:
        """Link what the mirror's `directory` holds into the tree at `snap`.

        A directory goes in as a link and a file hardlinked; whatever the tree holds there stays.
        """
        source = native(self.root, relative=directory)
        if not os.path.isdir(source):
            return
        for name in sorted(os.listdir(source)):
            entry, placed = (
                os.path.join(source, name),
                native(snap, relative=f"{directory}/{name}"),
            )
            if os.path.isdir(entry) and not os.path.lexists(placed):
                os.symlink(entry, placed)
            elif os.path.exists(entry) and not os.path.lexists(placed):
                _hardlink(entry, path=placed)

    def __wrapper(self, snap: str) -> None:
        """Freeze the staged job wrapper by its bytes, then check the frozen copy on every pin."""
        staged, where = self.spec["script"], self.spec["wrapper"]
        if not staged:
            return
        expected = where.rsplit("/", 1)[-1][len("job-") : -len(".sh")]
        wrappers = os.path.join(snap, WRAPPERS)
        if os.path.islink(wrappers):
            raise Refusal("linked wrapper directory")
        os.makedirs(wrappers, exist_ok=True)
        frozen = native(snap, relative=where)
        if not os.path.lexists(frozen):
            Snapshot.__freeze(
                native(self.root, relative=checked(staged)), frozen=frozen, expected=expected
            )
        linked = os.path.islink(frozen) or not os.path.isfile(frozen)
        if linked or sha256_of(frozen) != expected:
            raise Refusal("missing or linked wrapper" if linked else "wrapper digest mismatch")

    def pinned(self) -> str:
        """Build or verify the tree under the pin lock, and answer where it stands."""
        os.makedirs(self.base, exist_ok=True)
        with locked(os.path.join(self.base, ".pin.lock")):
            self.__retain()
            if os.path.isfile(_present(self.final, _marked("source"), STAMP)):
                self.__reuse()
            else:
                self.__build()
        return self.final
