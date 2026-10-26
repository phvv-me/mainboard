"""What the lake keeps and reports about evidence files."""

from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Literal

from patos import FrozenModel

# A file as the filesystem knows it: device, inode, size and modification time in nanoseconds.
type Identity = tuple[int, int, int, int]


# The folder `trials.archive.ParquetArtifacts` packs artifacts into. Readers address their members
# by path, opening its parts in the tree and never recalling them by digest, so the parts stay.
_ARCHIVE = "artifacts.parquet"


class Kept(FrozenModel):
    """What one `ingest` did.

    files: evidence files read.
    indexed: paths given a new index row, their content new or changed.
    objects: objects new to the lake, and `size` how many bytes those were.
    """

    files: int = 0
    indexed: int = 0
    objects: int = 0
    size: int = 0


class Located(FrozenModel):
    """One path's latest index row: which object it holds."""

    path: str
    sha256: str
    size: int


class Inventory(FrozenModel):
    """Where one indexed file stands, its tree copy against the lake's, as observed once.

    sha256: the digest the lake indexes at the path, and `size` its bytes.
    local_sha256: the digest of the file now at the path, empty where there is none to read:
        no file at all, a link, or anything but a regular file.
    local: whether the tree holds no file, the same bytes or other ones.
    lake: whether the lake holds no row of the object, an intact object or a damaged one.
    """

    path: str
    sha256: str
    size: int
    local_sha256: str
    local: Literal["missing", "identical", "changed"]
    lake: Literal["missing", "intact", "corrupt"]

    @property
    def archived(self) -> bool:
        """Whether the file is a part of an artifact archive, which readers open by path."""
        return is_archived(self.path)

    @property
    def evictable(self) -> bool:
        """Whether the tree copy is the lake's whole object and nothing reads it by path."""
        return self.local == "identical" and self.lake == "intact" and not self.archived


class Finalized(FrozenModel):
    """Files their producer closed, as they stood when it said so, the only ones to evict.

    files: each file with its identity at that moment. The producer promises that no writer
        remains. A file that no longer matches has broken the promise, so its bytes stay.
    """

    files: dict[Path, Identity]


def is_archived(path: str) -> bool:
    """Whether the workspace-relative `path` is a file inside an artifact archive's folder."""
    return _ARCHIVE in PurePosixPath(path).parts[:-1]


def located(rows: Iterable[tuple[str, str, int]]) -> list[Located]:
    return [Located(path=path, sha256=digest, size=size) for path, digest, size in rows]
