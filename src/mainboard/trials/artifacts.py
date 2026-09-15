"""Immutable trial artifacts and explicitly pinned inputs."""

import hashlib
import json
import os
from collections.abc import Iterable
from ntpath import isreserved
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from patos import FrozenModel
from pydantic import Field

from .archive import ParquetArtifacts


class Artifact(FrozenModel):
    """A portable content reference, relative to its declared project root."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    schema_name: str = ""

    @property
    def relative(self) -> PurePosixPath:
        """Validate canonical portable paths without interpreting the source machine's OS."""
        return relative_path(self.path)

    def read(self, root: Path) -> bytes:
        """Read pinned bytes through the project's logical storage mounts.

        Dispatch mounts result directories outside its source snapshot. References remain
        project-relative across that mount and after fetching; their hash verifies the bytes.
        """
        try:
            data = (root / self.relative).read_bytes()
        except FileNotFoundError:
            data = ParquetArtifacts.read(root / self.relative, boundary=root, digest=self.sha256)
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
        or ":" in value
        or isreserved(value)
    ):
        raise ValueError(f"artifact path must stay canonical and project-relative: {value}")
    return path


class Artifacts:
    """One trial's content-addressed output directory, never an ambient latest store."""

    def __init__(self, root: Path, directory: Path) -> None:
        self.root = Path(os.path.abspath(root))
        self.directory = Path(os.path.abspath(directory))
        self.directory.relative_to(self.root)

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
        roots = [directory, *directory.parents]
        roots = [root for root in reversed(roots) if root.is_relative_to(boundary)]
        for line in receipts:
            payload = json.loads(line)["trial_receipt"]
            for value in payload.get("artifacts", {}).values():
                if not isinstance(value, dict):
                    continue
                reference = Artifact.model_validate(value)
                relative = reference.relative
                root = next(
                    (root for root in roots if (root / relative).is_relative_to(directory)),
                    None,
                )
                if root is None:
                    raise ValueError(f"artifact is outside the declared fetch: {relative}")
                if not (root / relative).resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"artifact link leaves the declared fetch: {relative}")
                reference.read(root)

    def write(self, data: bytes, *, media_type: str, schema_name: str = "") -> Artifact:
        """Publish bytes before returning their immutable reference."""
        digest = hashlib.sha256(data).hexdigest()
        target = self.directory / "objects" / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise ValueError(f"artifact collision or incomplete write: {target}") from None
        finally:
            temporary.unlink()
        return Artifact(
            path=target.relative_to(self.root).as_posix(),
            sha256=digest,
            size=len(data),
            media_type=media_type,
            schema_name=schema_name,
        )
