"""Immutable trial artifacts and explicitly pinned inputs."""

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from patos import FrozenModel
from pydantic import Field


class Artifact(FrozenModel):
    """A portable content reference, relative to its declared project root."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    media_type: str = "application/octet-stream"
    schema_name: str = ""

    def read(self, root: Path) -> bytes:
        """Read pinned bytes through the project's logical storage mounts.

        Dispatch mounts result directories outside its source snapshot. References remain
        project-relative across that mount and after fetching; their hash verifies the bytes.
        """
        relative = Path(self.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact path must stay project-relative: {self.path}")
        data = (root / relative).read_bytes()
        if len(data) != self.size or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"artifact content changed: {self.path}")
        return data


class Artifacts:
    """One trial's content-addressed output directory, never an ambient latest store."""

    def __init__(self, root: Path, directory: Path) -> None:
        self.root = Path(os.path.abspath(root))
        self.directory = Path(os.path.abspath(directory))
        self.directory.relative_to(self.root)

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
