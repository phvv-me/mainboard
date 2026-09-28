# The center's file digests in its lake's `digests`: the agent's `Digests`, whose far side keeps
# a JSON file since it runs on a bare interpreter, with the workspace lake behind it here. Only a
# file whose stamp moved costs a row, and a path a pruning save no longer met costs one dropped
# row, so a mirror of an unchanged tree appends nothing.

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ...state.lake import Lake
from ..agent import Digests

if TYPE_CHECKING:
    from pathlib import Path

# The stamp fields a digest is remembered by, in the agent's order, then the digest itself.
_FIELDS = ("size", "inode", "mtime_ns", "ctime_ns", "sha256")


class KeptDigests(Digests):
    """One kind of digest memory (`mirror`, `collection`) of the workspace at `root`."""

    def __init__(self, root: Path, kind: str) -> None:  # noqa: PLW0231 - no file to read
        self.session = Lake.at(root).session()
        self.kind = kind
        rows = self.session.rows(
            f"SELECT path, {', '.join(_FIELDS)} FROM (SELECT * FROM lake.digests WHERE kind = ? "
            "QUALIFY row_number() OVER (PARTITION BY path ORDER BY rowid DESC) = 1) "
            "WHERE dropped IS NOT TRUE",
            [kind],
        )
        # A row an older release wrote without the inode or change time is simply rehashed.
        self.held: dict[str, list[int | str]] = {
            path: list(kept) for path, *kept in rows if None not in kept
        }
        self.kept = dict(self.held)
        self.seen: set[str] = set()

    def save(self, *, prune: bool = False) -> None:
        """Append what changed since the last save.

        prune: forget every path this memory was not asked about, as the agent's own save does.
        """
        if prune:
            self.held = {key: value for key, value in self.held.items() if key in self.seen}
        stamp = datetime.now(UTC)
        changed = [
            {
                "ts": stamp,
                "kind": self.kind,
                "path": path,
                **dict(zip(_FIELDS, value, strict=True)),
            }
            for path, value in self.held.items()
            if self.kept.get(path) != value
        ]
        gone = [
            {"ts": stamp, "kind": self.kind, "path": path, "dropped": True}
            for path in self.kept
            if path not in self.held
        ]
        self.session.append("digests", [*changed, *gone])
        self.kept = dict(self.held)
