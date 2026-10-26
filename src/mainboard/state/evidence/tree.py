"""The evidence one directory sees: its files on disk and what the lakes holding it index there."""

import hashlib
import logging
import os
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from functools import cache, partial
from pathlib import Path, PurePosixPath

from ...core.errors import MissionError
from ...core.project import Project
from ..lake import Lake
from ..storage.owned import link_in
from .core import Evidence
from .keeping import posix
from .models import Kept, Located

logger = logging.getLogger(__name__)


class EvidenceTree:
    """The evidence visible from one directory: its files on disk, and every file the lakes of
    the workspaces holding it index there, whether or not the file is still on disk."""

    @property
    def keepers(self) -> list[Evidence]:
        """The evidence of every workspace holding `base` that keeps a lake, nearest first."""
        found: dict[str, Evidence] = {}
        project = Project()
        for directory in (self.base, *self.base.parents):
            if project.manifest(directory).is_file():
                evidence = _kept(directory)
                if evidence.lake.exists():
                    found.setdefault(evidence.lake.served or str(evidence.lake.catalog), evidence)
        return list(found.values())

    def __init__(self, base: Path) -> None:
        self.base = Path(os.path.abspath(base))

    def _indexed(self) -> Iterator[tuple[Evidence, str, Located]]:
        """Each indexed file under `base`, with its keeper and its path relative to `base`."""
        for evidence in self.keepers:
            prefix = posix(self.base.relative_to(evidence.root))
            for row in evidence.indexed(prefix):
                yield evidence, row.path.removeprefix(prefix).removeprefix("/"), row

    def directories(self, pattern: str) -> list[Path]:
        """Every directory under `base` matching the glob `pattern`, in path order.

        A directory counts when it is on disk or holds a file a lake indexes, and is named where it
        would stand in the tree.
        """
        found = {
            path.relative_to(self.base).as_posix()
            for path in self.base.glob(pattern)
            if path.is_dir()
        }
        held = {
            parent
            for _, relative, _ in self._indexed()
            for parent in PurePosixPath(relative).parents
            if parent.parts
        }
        found |= {parent.as_posix() for parent in held if parent.full_match(pattern)}
        return [self.base / relative for relative in sorted(found)]

    def files(self, pattern: str) -> list[Path]:
        """Every file under `base` matching the glob `pattern`, in path order.

        Files come from disk, else from the read-through cache of the lake indexing them.
        """
        found = {path.relative_to(self.base).as_posix(): path for path in self.base.glob(pattern)}
        for evidence, relative, row in self._indexed():
            if relative not in found and PurePosixPath(relative).full_match(pattern):
                found[relative] = evidence.cached(row)
        return [found[relative] for relative in sorted(found)]

    def holds(self, digests: Collection[str]) -> set[str]:
        """The digests among `digests` some lake holding `base` keeps an object for."""
        return {
            digest
            for evidence in self.keepers
            for digest in evidence.session.run(partial(evidence.blobs.held, digests=digests))
        }

    def keep(self, paths: Sequence[Path]) -> Kept | None:
        """Move `paths` into the center's lake, the nearest one holding `base`.

        They are kept byte for byte, then their tree copies dropped, since the lake is the
        evidence's home and the tree only its cache. The caller is their producer and has closed
        them (a finished run's receipts), which is what lets the copies go; one that was written
        since is kept in the tree. A path behind a link is left where it is, in the tree. None when
        the nearest lake is no center's: a dispatch mirror's staging lake is not a home, and the
        center collects the mirror's tree.
        """
        keepers = self.keepers[:1]
        if not (paths and keepers and keepers[0].lake.central):
            return None
        keeper = keepers[0]
        owned = [path for path in paths if link_in(path, root=keeper.root) is None]
        if len(owned) < len(paths):
            logger.warning(
                "%d of %d paths lie behind a link and stay", len(paths) - len(owned), len(paths)
            )
        if not owned:
            return None
        closed = keeper.finalize(owned)
        kept = keeper.ingest(owned)
        keeper.evict(closed)
        return kept

    def read(self, path: Path, sha256: str) -> bytes:
        """The bytes pinned at `path` by `sha256`: from disk, else from a lake by digest.

        Raises FileNotFoundError when neither holds them, ValueError when the bytes found differ.
        """
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = self.recall(sha256)
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"{path} does not match its pinned digest")
        return data

    def recall(self, sha256: str) -> bytes:
        """The object with `sha256` from the first lake holding it intact.

        Raises FileNotFoundError when none does.
        """
        for evidence in self.keepers:
            data = evidence.recall(sha256)
            if data is not None:
                return data
        raise FileNotFoundError(f"no lake around {self.base} holds {sha256}")

    def recalled(self, sizes: Mapping[str, int]) -> Iterator[tuple[str, bytes]]:
        """Each object among `sizes` from the first lake holding it intact, a window at a time.

        `sizes` maps digest to size, as `Evidence.recalled` reads it; an object no lake holds is
        not yielded.
        """
        wanted = dict(sizes)
        for evidence in self.keepers:
            for digest, data in evidence.recalled(dict(wanted)):
                del wanted[digest]
                yield digest, data

    def restore(self, paths: Iterable[Path]) -> list[Path]:
        """Write back the files under `paths` the tree lacks from the nearest lake keeping them.

        It answers the files written. A lake keeps a file when it holds `base`; a path no lake
        indexes is left missing, for its reader to fail on.
        """
        written: list[Path] = []
        for path in paths:
            for keeper in self.keepers:
                with suppress(MissionError):
                    written += keeper.materialize([path])
                    break
        return written


@cache
def _kept(root: Path) -> Evidence:
    """One `Evidence` per workspace for the life of the process.

    Its lake session (an attach costs a tenth of a second) then outlives the reader that asked, as
    a registry's does.
    """
    return Evidence(Lake.at(root))
