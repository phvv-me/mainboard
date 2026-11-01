import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Literal, Self

from patos import FrozenModel
from pydantic import Field, model_validator

from ....core.constants.names import SHA256
from .enums import ArtifactKind, Confidence, Coverage, Provenance

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


def digest(value: JsonValue) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Artifact(FrozenModel):
    path: str
    sha256: str = Field(pattern=SHA256)
    size: int = Field(ge=0)
    kind: ArtifactKind

    @classmethod
    def at(cls, path: Path) -> Self:
        path = path.resolve(strict=True)
        with path.open("rb") as stream:
            header = stream.read(64)
            stream.seek(0)
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            size = stream.tell()
        return cls(path=str(path), sha256=sha256, size=size, kind=cls._kind_of(header, path))

    @staticmethod
    def _elf_kind(header: bytes, path: Path) -> ArtifactKind:
        if len(header) < 20 or header[5] not in {1, 2}:
            raise ValueError(f"truncated or invalid ELF header: {path}")
        byteorder: Literal["little", "big"] = "little" if header[5] == 1 else "big"
        if int.from_bytes(header[18:20], byteorder) == 190:
            return ArtifactKind.cubin
        return ArtifactKind.native_elf

    @staticmethod
    def _kind_of(header: bytes, path: Path) -> ArtifactKind:
        magic = header[:4]
        if magic == b"\x7fELF":
            return Artifact._elf_kind(header, path)
        if magic in {b"\xb1CbF", b"\x50\xed\x55\xba"}:
            return ArtifactKind.fatbin
        if magic in {
            b"\xcf\xfa\xed\xfe",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xfe\xed\xfa\xce",
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xba\xbf",
            b"\xbf\xba\xfe\xca",
        }:
            return ArtifactKind.native_macho
        if path.suffix.lower() == ".ptx":
            return ArtifactKind.ptx
        raise ValueError(f"unrecognized device artifact: {path}")


class Limitation(FrozenModel):
    kind: str
    subject: str
    detail: str
    remediation: str


class ResidualUnknown(FrozenModel):
    site: str
    question: str
    candidates: tuple[str, ...] = ()
    reason: str


class Fact[T](FrozenModel):
    value: T
    confidence: Confidence
    provenance: Provenance


class AnalysisProfile(FrozenModel):
    """Completeness is relative to this provider, release and canonical options digest."""

    provider: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    options_digest: str = Field(pattern=SHA256)

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class Evidence[T](FrozenModel):
    artifact: Artifact
    provider: str
    tool_version: str
    operation: str
    params: Mapping[str, JsonValue]
    result: T
    coverage: Coverage
    analysis_profile: AnalysisProfile
    limitations: tuple[Limitation, ...] = ()
    unresolved: tuple[ResidualUnknown, ...] = ()
    raw: JsonValue = None

    @model_validator(mode="after")
    def profile_matches(self) -> Self:
        expected = AnalysisProfile(
            provider=self.provider,
            tool_version=self.tool_version,
            options_digest=digest(dict(self.params)),
        )
        if self.analysis_profile != expected:
            raise ValueError("analysis profile must identify the evidence provider and options")
        return self

    def key(self) -> str:
        """The shared lake/cache address; paths and findings do not identify an analysis."""
        return digest(
            [
                self.artifact.sha256,
                self.operation,
                dict(self.params),
                self.provider,
                self.tool_version,
            ]
        )


def coverage_of(values: Collection[Coverage]) -> Coverage:
    for status in (Coverage.failed, Coverage.truncated, Coverage.partial, Coverage.not_run):
        if status in values:
            return status
    return Coverage.complete
