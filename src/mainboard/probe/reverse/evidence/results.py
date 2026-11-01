from patos import FrozenModel
from pydantic import field_validator

from .core import Fact, ResidualUnknown
from .enums import Confidence, Coverage, Provenance

type Address = int

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

    @field_validator("kernels")
    @classmethod
    def unique_kernels(cls, kernels: tuple[Kernel, ...]) -> tuple[Kernel, ...]:
        if len({(kernel.name, kernel.arch) for kernel in kernels}) != len(kernels):
            raise ValueError("kernel identities (name, arch) must be unique")
        return kernels

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
    def branches(self) -> BranchSummary:
        return BranchSummary(unresolved=len(self.brx_sites))

    @property
    def brx_sites(self) -> tuple[Address, ...]:
        # BRA/BRX spellings vary by generation; none of these indirect targets is resolved
        # merely because a tool printed the instruction or a register operand.
        return tuple(
            insn.address
            for insn in self.instructions
            if insn.opcode.split(".")[0] in {"BRX", "BRXU", "JMX", "JMXU"}
        )


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
