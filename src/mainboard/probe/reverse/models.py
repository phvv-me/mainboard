import hashlib
import json
from collections.abc import Mapping
from enum import IntEnum, StrEnum, auto
from pathlib import Path
from typing import Literal, Self

from patos import FrozenModel
from pydantic import Field, model_validator

type Address = int
type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


def digest(value: JsonValue) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ArtifactKind(StrEnum):
    native_elf = auto()
    native_macho = auto()
    fatbin = auto()
    cubin = auto()
    ptx = auto()
    # Breadth kinds (dotnet, asar, apk) belong to the later REA phase.


class Artifact(FrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
        match header[:4]:
            case b"\x7fELF":
                if len(header) < 20 or header[5] not in {1, 2}:
                    raise ValueError(f"truncated or invalid ELF header: {path}")
                byteorder: Literal["little", "big"] = "little" if header[5] == 1 else "big"
                kind = (
                    ArtifactKind.cubin
                    if int.from_bytes(header[18:20], byteorder) == 190
                    else ArtifactKind.native_elf
                )
            case b"\xb1CbF" | b"\x50\xed\x55\xba":
                kind = ArtifactKind.fatbin
            case (
                b"\xcf\xfa\xed\xfe"
                | b"\xce\xfa\xed\xfe"
                | b"\xfe\xed\xfa\xcf"
                | b"\xfe\xed\xfa\xce"
                | b"\xca\xfe\xba\xbe"
                | b"\xbe\xba\xfe\xca"
                | b"\xca\xfe\xba\xbf"
                | b"\xbf\xba\xfe\xca"
            ):
                kind = ArtifactKind.native_macho
            case _ if path.suffix.lower() == ".ptx":
                kind = ArtifactKind.ptx
            case _:
                raise ValueError(f"unrecognized device artifact: {path}")
        return cls(path=str(path), sha256=sha256, size=size, kind=kind)


class Coverage(StrEnum):
    complete = auto()
    partial = auto()
    truncated = auto()
    failed = auto()
    not_run = auto()


class Confidence(StrEnum):
    observed = auto()
    inferred = auto()
    uncertain = auto()


class Provenance(IntEnum):
    """Merge competing provenance by max; ANALYSIS includes AI-derived findings."""

    DEFAULT = 0
    IMPORTED = 1
    ANALYSIS = 2
    USER_DEFINED = 3


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


class LiftedRef(FrozenModel):
    tier: str
    form: str
    producer: str
    blob_hash: str


class AnalysisProfile(FrozenModel):
    """Completeness is relative to this provider, release and canonical options digest."""

    provider: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    options_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

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


_UNKNOWN_COUNT = Fact[int | None](
    value=None, confidence=Confidence.uncertain, provenance=Provenance.DEFAULT
)


class Kernel(FrozenModel):
    name: str
    arch: str
    # Missing measurements remain explicit uncertain facts, never fabricated zeros.
    registers: Fact[int | None] = _UNKNOWN_COUNT
    shared_bytes: Fact[int | None] = _UNKNOWN_COUNT


class KernelChange(FrozenModel):
    before: Kernel
    after: Kernel


class InventoryDiff(FrozenModel):
    """Changes to reported metadata, not proof that equal kernels have equal machine code."""

    added: tuple[Kernel, ...]
    removed: tuple[Kernel, ...]
    changed: tuple[KernelChange, ...]
    added_arches: tuple[str, ...] = ()
    removed_arches: tuple[str, ...] = ()


class KernelInventory(FrozenModel):
    kernels: tuple[Kernel, ...]
    arches: tuple[str, ...]

    @model_validator(mode="after")
    def unique_kernels(self) -> Self:
        keys = {(kernel.name, kernel.arch) for kernel in self.kernels}
        if len(keys) != len(self.kernels):
            raise ValueError("kernel identities (name, arch) must be unique")
        return self

    def diff(self, newer: KernelInventory) -> InventoryDiff:
        before = {(kernel.name, kernel.arch): kernel for kernel in self.kernels}
        after = {(kernel.name, kernel.arch): kernel for kernel in newer.kernels}
        return InventoryDiff(
            added=tuple(after[key] for key in sorted(after.keys() - before.keys())),
            removed=tuple(before[key] for key in sorted(before.keys() - after.keys())),
            changed=tuple(
                KernelChange(before=before[key], after=after[key])
                for key in sorted(before.keys() & after.keys())
                if before[key] != after[key]
            ),
            added_arches=tuple(sorted(set(newer.arches) - set(self.arches))),
            removed_arches=tuple(sorted(set(self.arches) - set(newer.arches))),
        )


class SassInsn(FrozenModel):
    address: Address
    opcode: str
    operands: str
    predicate: str
    raw: str


class BranchSummary(FrozenModel):
    resolved: int = 0
    unresolved: int = 0


class SassListing(FrozenModel):
    kernel: str
    arch: str
    instructions: tuple[SassInsn, ...]

    @property
    def brx_sites(self) -> tuple[Address, ...]:
        # BRA/BRX spellings vary by generation; none of these indirect targets is resolved
        # merely because a tool printed the instruction or a register operand.
        return tuple(
            insn.address
            for insn in self.instructions
            if insn.opcode.split(".")[0] in {"BRX", "BRXU", "JMX", "JMXU"}
        )

    @property
    def branches(self) -> BranchSummary:
        return BranchSummary(unresolved=len(self.brx_sites))


class SassResult(FrozenModel):
    """All selected functions, retaining separate address spaces for each architecture."""

    listings: tuple[SassListing, ...]

    @property
    def branches(self) -> BranchSummary:
        return BranchSummary(
            resolved=sum(item.branches.resolved for item in self.listings),
            unresolved=sum(item.branches.unresolved for item in self.listings),
        )


class CubinEntry(FrozenModel):
    name: str
    arch: str


class ArchCoverage(FrozenModel):
    arches: tuple[str, ...]
    ptx_arches: tuple[str, ...] = ()
    cubins: tuple[CubinEntry, ...] = ()


type DeviceResult = KernelInventory | SassResult | ArchCoverage | InventoryDiff


class Parsed[T](FrozenModel):
    result: T
    coverage: Coverage
    unresolved: tuple[ResidualUnknown, ...] = ()


def coverage_of(values: tuple[Coverage, ...]) -> Coverage:
    for status in (Coverage.failed, Coverage.truncated, Coverage.partial, Coverage.not_run):
        if status in values:
            return status
    return Coverage.complete
