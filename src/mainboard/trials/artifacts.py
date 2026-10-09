"""Immutable trial artifacts and explicitly pinned inputs."""

import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from patos import FrozenModel
from pydantic import Field

from ..state.evidence import EvidenceTree
from .archive import ParquetArtifacts

# What a read of table artifacts answers when none holds the schema: no rows, and the `_trial`
# provenance column every such read adds beside the table's own.
NO_TABLE = "SELECT NULL::VARCHAR AS _trial WHERE false"

# A file in the Hugging Face hub cache, `<kind>s--<repo, "/" spelled "--">/snapshots/<revision>/`;
# a repository name may not hold `--`, so the spelling reads back exactly.
_HUB = re.compile(r"(models|datasets|spaces)--([^/]+)/snapshots/([0-9a-f]{40})/(.+)$")


class Artifact(FrozenModel):
    """A portable content reference, relative to its declared project root.

    path: where the bytes stand: the node's content store (`objects/<sha256[:2]>/<sha256>` beside
        its receipts) since 2026-10-09, before that a copy in each run's own `objects/`.
    source: where pinned bytes were downloaded from, `hf://<repo>@<revision>/<file>` for a Hugging
        Face file; empty for bytes a trial made.
    """

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    schema_name: str = ""
    source: str = ""

    @property
    def relative(self) -> PurePosixPath:
        """Validate canonical portable paths without interpreting the source machine's OS."""
        return relative_path(self.path)

    def read(self, root: Path) -> bytes:
        """Read pinned bytes through the project's logical storage mounts.

        Dispatch mounts result directories outside its source snapshot. References remain
        project-relative across that mount and after fetching; their hash verifies the bytes.
        Found on disk, else in a Parquet archive under `root`, else by digest in the lake of a
        workspace holding `root`, once the file has left the tree.
        """
        path = root / self.relative
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            try:
                data = ParquetArtifacts.read(path, boundary=root, digest=self.sha256)
            except FileNotFoundError:
                data = EvidenceTree(root).recall(self.sha256)
        if len(data) != self.size or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"artifact content changed: {self.path}")
        return data


def relative_path(value: str) -> PurePosixPath:
    """Validate a portable project-relative reference, independent of the reader's OS."""
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
    ):
        raise ValueError(f"artifact path must stay canonical and project-relative: {value}")
    return path


def pinned(path: Path) -> str:
    """Where `path` was downloaded from when it is a Hugging Face hub cache file, else empty."""
    found = _HUB.search(Path(os.path.abspath(path)).as_posix())
    if found is None:
        return ""
    kind, repository, revision, name = found.groups()
    prefix = "" if kind == "models" else f"{kind}/"
    return f"hf://{prefix}{repository.replace('--', '/')}@{revision}/{name}"


class Artifacts:
    """A content store under `directory/objects`, each artifact's bytes held once however many
    trials and runs reference them; never an ambient latest store, since a reference names its
    digest."""

    def __init__(self, root: Path, directory: Path) -> None:
        self.root = Path(os.path.abspath(root))
        self.directory = Path(os.path.abspath(directory))
        self.directory.relative_to(self.root)
        self.written: set[Path] = set()

    @staticmethod
    def verify(receipts: Iterable[str], *, directory: Path, boundary: Path) -> None:
        """Verify receipt references within the fetched directory, without importing a project.

        References are project-relative; the fetch directory is workspace-relative. Match
        their common suffix to recover the logical root, bounded by this workspace. Never
        accept a same-named artifact from a different project's result directory.
        """
        directory = directory.absolute()
        boundary = boundary.absolute()
        directory.relative_to(boundary)
        roots = [
            root
            for root in reversed([directory, *directory.parents])
            if root.is_relative_to(boundary)
        ]
        references = {
            Artifact.model_validate(value)
            for line in receipts
            for value in json.loads(line)["trial_receipt"].get("artifacts", {}).values()
            if isinstance(value, dict)
        }
        # A file the center did not need sent is indexed in its lake only; its bytes are one
        # object there however many runs name it, so each digest is read back once.
        absent: dict[str, tuple[Artifact, Path]] = {}
        for reference in references:
            relative = reference.relative
            # The fetch directory is always a root and a validated reference never climbs out of
            # it, so some root always matches; the outermost one is the project.
            root = next(root for root in roots if (root / relative).is_relative_to(directory))
            if not (root / relative).resolve().is_relative_to(directory.resolve()):
                raise ValueError(f"artifact link leaves the declared fetch: {relative}")
            if (root / relative).exists():
                reference.read(root)
            else:
                absent[reference.sha256] = (reference, root)
        for reference, root in absent.values():
            reference.read(root)

    def write(
        self, data: bytes, *, media_type: str, schema_name: str = "", source: str = ""
    ) -> Artifact:
        """Hold bytes in the store, once, before returning their immutable reference.

        Bytes already held are read back and compared rather than written again; a held copy
        that differs is damaged and replaced by these, which hash to its name.
        """
        digest = hashlib.sha256(data).hexdigest()
        target = self.directory / "objects" / digest[:2] / digest
        try:
            held = target.read_bytes() == data
        except FileNotFoundError:
            held = False
        if not held:
            _publish(target, data)
        self.written.add(target)
        return Artifact(
            path=target.relative_to(self.root).as_posix(),
            sha256=digest,
            size=len(data),
            media_type=media_type,
            schema_name=schema_name,
            source=source,
        )


def _publish(target: Path, data: bytes) -> None:
    """Write `data` beside `target` and link it into place.

    A racing writer's copy is kept when it holds the same bytes, and a damaged one is replaced.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError:
        if target.read_bytes() != data:
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
