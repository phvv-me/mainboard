"""A second copy of the lake's evidence on another disk."""

import hashlib
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path

from ...core.errors import MissionError
from .models import Located


class DirectoryReplica:
    """A replica on a mounted disk: each object one file at `<root>/<first two hex>/<sha256>`,
    written beside its place and renamed in only once its digest checks, so an interrupted copy
    never leaves wrong bytes under a good name; the index is `<root>/index.jsonl`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _object(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def holds(self, digests: Collection[str]) -> set[str]:
        return {digest for digest in digests if self._object(digest).is_file()}

    def index(self, rows: Sequence[Located]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        partial = self.root / "index.jsonl.partial"
        partial.write_text(
            "".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8", newline="\n"
        )
        partial.replace(self.root / "index.jsonl")

    def put(self, digest: str, chunks: Iterator[bytes]) -> None:
        target = self._object(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        hasher = hashlib.sha256()
        with partial.open("wb") as sink:
            for chunk in chunks:
                hasher.update(chunk)
                sink.write(chunk)
        if hasher.hexdigest() != digest:
            partial.unlink()
            raise MissionError(
                f"object {digest} read back as {hasher.hexdigest()}; not replicated"
            )
        partial.replace(target)
