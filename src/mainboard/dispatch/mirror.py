# Bring a target's copy of the workspace up to date with this one, on any OS at either end.
#
# Both ends walk the same scopes under the same compiled rules and describe every file by its
# size and SHA-256, each end remembering its digests by size and timestamp so an unchanged tree
# is never reread. The difference between the two descriptions is the whole transfer: the files
# that differ travel as one compressed tar stream over the one SSH channel, the paths the center
# no longer holds are pruned unless a protect rule keeps them, and the target applies all of it
# under its own mirror lock. Content decides what differs, never a timestamp, so two machines
# whose clocks or file systems keep time differently still agree on what is already there.

import os
import tarfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import TypeAdapter

from ..core.errors import MissionError
from .agent import Digests, Entry, walk
from .agent.program import CHUNK, DIRECTORY, FILE, LINK, native
from .shared import logger, state_dir, state_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from .agent import Agent, Rules, Scope
    from .agent.program import ReceiveSpec
    from .agent.runner import Sink

# The compression the stream is packed at: the fastest gzip level, since source text still
# shrinks severalfold while Parquet and weights shrink at no level, and a slower level would cap
# a fast link at what one core compresses.
_LEVEL = 1

# One survey record: path, kind, size, nanosecond mtime, execute bit, and digest or link target.
_RECORD = TypeAdapter(tuple[str, str, int, int, bool | None, str])


class _Capabilities(FrozenModel):
    """What the target's file system can hold, the first record of every survey."""

    links: bool
    modes: bool
    fold: bool


class _Received(FrozenModel):
    """What the target reports once it applied a mirror."""

    written: int
    bytes: int
    deleted: tuple[str, ...]
    kept: tuple[str, ...]


class Pushed(FrozenModel):
    """What one mirror did.

    files: how many files the center holds in scope.
    sent: how many of them crossed the wire, and their `bytes` before compression.
    deleted: what the target pruned; `kept`, what it could not prune because the rules keep
        something beneath it.
    """

    files: int = 0
    sent: int = 0
    bytes: int = 0
    deleted: tuple[str, ...] = ()
    kept: tuple[str, ...] = ()


class Mirror:
    """Mirror scoped trees and named files of this workspace onto one target.

    workspace: the local workspace root every path is relative to.
    agent: the target's agent.
    """

    def __init__(self, workspace: Path, agent: Agent) -> None:
        self.workspace = workspace
        self.agent = agent

    def push(
        self,
        root: str,
        *,
        scopes: Sequence[Scope],
        named: Sequence[str] = (),
        protected: Rules,
    ) -> Pushed:
        """Bring `root` on the target up to date and answer what that took.

        scopes: the trees mirrored whole, each pruned on the target down to what it holds here.
        named: files shipped by name whatever the rules say, never pruned, each required to
            exist here and to still exist when the stream reaches it.
        protected: what the target never prunes, whether or not this workspace holds it.
        """
        digests = Digests(str(state_path(self.workspace) / "digests.json"))
        local = self.__local(scopes, named, digests=digests)
        first, *records = self.agent.ask(
            {
                "survey": {
                    "root": root,
                    "state": state_dir(),
                    "scopes": [scope.spec() for scope in scopes],
                    "named": list(named),
                }
            }
        )
        capabilities = _Capabilities.model_validate(first)
        if not capabilities.links:
            local = self.__dereferenced(local, digests=digests)
        digests.save()
        if capabilities.fold and (clashes := clashing(local)):
            raise MissionError(
                f"{self.agent.host} folds case, and these paths differ only in case: "
                + ", ".join(clashes)
            )
        held = {
            entry.path: entry
            for entry in (Entry(*_RECORD.validate_python(record)) for record in records)
        }
        delta = Delta.between(local, held, protected=protected, modes=capabilities.modes)
        logger.info(
            "mirroring %d of %d path(s) to %s:%s",
            len(delta.files),
            len(local),
            self.agent.host,
            root,
        )
        [answer] = self.agent.ask(
            {"receive": delta.request(root)},
            payload=partial(self.__pack, delta.files, named=frozenset(named)),
        )
        received = _Received.model_validate(answer)
        if received.deleted:
            logger.warning(
                "mirror deleted %d path(s): %s", len(received.deleted), ", ".join(received.deleted)
            )
        return Pushed(
            files=sum(entry.kind == FILE for entry in local.values()),
            sent=received.written,
            bytes=received.bytes,
            deleted=received.deleted,
            kept=received.kept,
        )

    def __local(
        self, scopes: Sequence[Scope], named: Sequence[str], *, digests: Digests
    ) -> dict[str, Entry]:
        """Every entry this workspace holds in scope and by name, files hashed through memory."""
        root = str(self.workspace)
        found: dict[str, Entry] = {}
        for scope in scopes:
            for entry in walk(root, scope):
                found[entry.path] = self.__hashed(entry, digests)
        for relative in named:
            status = os.stat(native(root, relative))
            found[relative] = self.__hashed(Entry.stated(relative, status), digests)
        return found

    def __dereferenced(self, local: dict[str, Entry], *, digests: Digests) -> dict[str, Entry]:
        """`local` as a target that holds no links takes it: each link to a file sent as that file.

        A link to a directory has no file to become and is left behind with a warning, since
        walking into it could carry a whole second tree under a name nobody declared.
        """
        adapted: dict[str, Entry] = {}
        for path, entry in local.items():
            referent = native(str(self.workspace), path)
            if entry.kind != LINK:
                adapted[path] = entry
            elif os.path.isfile(referent):
                adapted[path] = self.__hashed(Entry.stated(path, os.stat(referent)), digests)
            else:
                logger.warning("%s cannot hold the link %s; left behind", self.agent.host, path)
        return adapted

    def __hashed(self, entry: Entry, digests: Digests) -> Entry:
        if entry.kind == FILE:
            path = native(str(self.workspace), entry.path)
            entry.detail = digests.of(path, key=entry.path)
        return entry

    def __pack(self, files: Sequence[Entry], sink: Sink, *, named: frozenset[str]) -> None:
        """Stream `files` into `sink` as one gzip tar, a file at a time and a chunk at a time.

        A file that vanished since it was hashed is skipped with a warning, unless it was named,
        since a named file is one the dispatch cannot run without.
        """
        with tarfile.open(
            fileobj=sink, mode="w|gz", bufsize=CHUNK, compresslevel=_LEVEL
        ) as archive:
            for entry in files:
                try:
                    with open(native(str(self.workspace), entry.path), "rb") as source:
                        info = tarfile.TarInfo(entry.path)
                        info.size = os.fstat(source.fileno()).st_size
                        info.mtime = entry.mtime // 1_000_000_000
                        info.mode = 0o755 if entry.executable else 0o644
                        archive.addfile(info, source)
                except FileNotFoundError as vanished:
                    if entry.path in named:
                        raise MissionError(
                            f"{entry.path} vanished before it could be sent"
                        ) from vanished
                    logger.warning("skipping %s, gone before the stream reached it", entry.path)


class Delta(FrozenModel):
    """The difference between two descriptions of one tree, as the target has to apply it.

    files: what the stream carries, in path order.
    directories: what the target makes, empty ones included.
    links: what the target links, path to target.
    delete: what the target prunes, deepest first on its side.
    """

    files: tuple[Entry, ...] = ()
    directories: tuple[str, ...] = ()
    links: tuple[tuple[str, str], ...] = ()
    delete: tuple[str, ...] = ()

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def between(
        cls,
        local: Mapping[str, Entry],
        held: Mapping[str, Entry],
        *,
        protected: Rules,
        modes: bool,
    ) -> Delta:
        """What turns `held` into `local`, leaving alone whatever `protected` claims.

        A file differs by size or digest, and by its execute bit when both ends keep one. A
        path whose kind changed, or a link whose target did, is pruned before it is remade,
        since neither a directory nor a link is replaced by a rename.

        held: what the target described.
        protected: paths the target keeps whether or not this workspace holds them.
        modes: whether the target keeps execute bits.
        """
        files = tuple(
            entry
            for path, entry in sorted(local.items())
            if entry.kind == FILE and _differs(entry, held.get(path), modes=modes)
        )
        stale = tuple(
            path
            for path, theirs in held.items()
            if _stale(local.get(path), theirs)
            and not protected.matches(path, directory=theirs.kind == DIRECTORY)
        )
        remade = set(stale)
        placed = [
            (path, entry)
            for path, entry in sorted(local.items())
            if path not in held or path in remade
        ]
        return cls(
            files=files,
            directories=tuple(path for path, entry in placed if entry.kind == DIRECTORY),
            links=tuple((path, entry.detail) for path, entry in placed if entry.kind == LINK),
            delete=stale,
        )

    def request(self, root: str) -> ReceiveSpec:
        """This delta as the receive request the target applies under `root`."""
        return {
            "root": root,
            "state": state_dir(),
            "delete": list(self.delete),
            "directories": list(self.directories),
            "links": dict(self.links),
            "files": {entry.path: (entry.mtime, entry.executable) for entry in self.files},
        }


def clashing(paths: Iterable[str]) -> list[str]:
    """The paths differing from another only in case, which a case-folding target holds as one."""
    folded: dict[str, list[str]] = {}
    for path in paths:
        folded.setdefault(path.casefold(), []).append(path)
    return sorted(path for group in folded.values() if len(group) > 1 for path in group)


def _differs(mine: Entry, theirs: Entry | None, *, modes: bool) -> bool:
    """Whether the target's copy of a file is missing or holds other bytes or another mode."""
    if theirs is None or theirs.kind != FILE:
        return True
    return (mine.size, mine.detail) != (theirs.size, theirs.detail) or (
        modes
        and None not in (mine.executable, theirs.executable)
        and mine.executable != theirs.executable
    )


def _stale(mine: Entry | None, theirs: Entry) -> bool:
    """Whether the target's entry has to go: gone here, of another kind, or linked elsewhere."""
    if mine is None or mine.kind != theirs.kind:
        return True
    return mine.kind == LINK and mine.detail != theirs.detail
