from pathlib import Path

from ...core.errors import MissionError
from ...render.values import Node
from ...state.lake import Lake
from .models import (
    AnalysisProfile,
    ArchCoverage,
    Artifact,
    DeviceResult,
    Evidence,
    InventoryDiff,
    JsonValue,
    KernelInventory,
    Limitation,
    SassResult,
    coverage_of,
    digest,
)
from .providers import Request, ReverseProvider
from .store import EvidenceStore


class DeviceProbe:
    @staticmethod
    def _refuse_misuse(operation: str, *, function: str, against: str) -> None:
        if operation == "compare" and not against:
            raise MissionError("compare requires --against <newer-library>")
        if operation != "compare" and against:
            raise MissionError("--against belongs to compare")
        if operation == "arch" and function:
            raise MissionError("--function cannot filter an image's architecture coverage")

    def __init__(self, lake: Lake) -> None:
        self.store = EvidenceStore(lake)

    def _analyze(
        self, artifact: Artifact, operation: str, *, arch: str, function: str
    ) -> Evidence[DeviceResult]:
        provider = ReverseProvider.supporting("device", artifact.kind, operation)
        evidence = provider.analyze(
            Request(artifact=artifact, operation=operation, arch=arch, function=function)
        )
        self.store.ingest(evidence)
        return evidence

    def _compare(
        self,
        library: Path,
        artifact: Artifact,
        against: str,
        *,
        arch: str,
        function: str,
    ) -> Evidence[DeviceResult]:
        before = self.run(library, "inventory", arch=arch, function=function)
        after = self.run(Path(against), "inventory", arch=arch, function=function)
        match before.result, after.result:
            case KernelInventory() as older, KernelInventory() as newer:
                result = older.diff(newer)
            case _:
                raise MissionError("compare requires two kernel inventories")
        params: dict[str, JsonValue] = {
            "before": before.key(),
            "after": after.key(),
            "against_sha256": after.artifact.sha256,
            "arch": arch,
            "function": function,
        }
        evidence = Evidence[DeviceResult](
            artifact=artifact,
            provider="mainboard.inventory-diff",
            tool_version="1",
            operation="compare",
            params=params,
            result=result,
            coverage=coverage_of((before.coverage, after.coverage)),
            analysis_profile=AnalysisProfile(
                provider="mainboard.inventory-diff",
                tool_version="1",
                options_digest=digest(params),
            ),
            limitations=before.limitations
            + after.limitations
            + (
                Limitation(
                    kind="metadata_only",
                    subject="compare",
                    detail="Changes cover kernel identity and resources, not code equivalence",
                    remediation="Inspect SASS evidence for changes to instruction bodies",
                ),
            ),
            unresolved=before.unresolved + after.unresolved,
        )
        self.store.ingest(evidence)
        return evidence

    def run(
        self,
        library: Path,
        operation: str,
        *,
        arch: str = "",
        function: str = "",
        against: str = "",
    ) -> Evidence[DeviceResult]:
        self._refuse_misuse(operation, function=function, against=against)
        artifact = Artifact.at(library)
        if operation == "compare":
            return self._compare(library, artifact, against, arch=arch, function=function)
        return self._analyze(artifact, operation, arch=arch, function=function)


def compact(evidence: Evidence[DeviceResult]) -> list[dict[str, Node]]:
    """Every caveat survives the compact rendering, including empty and failed results."""
    rows: list[dict[str, Node]] = [
        {
            "record": "evidence",
            "evidence_id": evidence.key(),
            "coverage": evidence.coverage.value,
            "provider": evidence.provider,
            "operation": evidence.operation,
        }
    ]
    match evidence.result:
        case KernelInventory() as result:
            rows.extend(
                {
                    "record": "kernel",
                    "kernel": kernel.name,
                    "arch": kernel.arch,
                    "registers": kernel.registers.value,
                    "shared_bytes": kernel.shared_bytes.value,
                }
                for kernel in result.kernels
            )
        case ArchCoverage() as result:
            rows.extend({"record": "cubin_arch", "arch": arch} for arch in result.arches)
            rows.extend({"record": "ptx_arch", "arch": arch} for arch in result.ptx_arches)
        case SassResult() as result:
            rows.append(
                {
                    "record": "indirect_branches",
                    "resolved": result.branches.resolved,
                    "unresolved": result.branches.unresolved,
                }
            )
            rows.extend(
                {
                    "record": "instruction",
                    "kernel": listing.kernel,
                    "arch": listing.arch,
                    "address": insn.address,
                    "opcode": insn.opcode,
                    "operands": insn.operands,
                    "predicate": insn.predicate,
                }
                for listing in result.listings
                for insn in listing.instructions
            )
        case InventoryDiff() as result:
            rows.extend(
                {"record": "added", **kernel.model_dump(mode="json")} for kernel in result.added
            )
            rows.extend(
                {"record": "removed", **kernel.model_dump(mode="json")}
                for kernel in result.removed
            )
            rows.extend(
                {
                    "record": "changed",
                    "kernel": change.after.name,
                    "arch": change.after.arch,
                    "before": change.before.model_dump(mode="json"),
                    "after": change.after.model_dump(mode="json"),
                }
                for change in result.changed
            )
            rows.extend({"record": "added_arch", "arch": arch} for arch in result.added_arches)
            rows.extend({"record": "removed_arch", "arch": arch} for arch in result.removed_arches)
    rows.extend(
        {"record": "limitation", **item.model_dump(mode="json")} for item in evidence.limitations
    )
    rows.extend(
        {"record": "unknown", **item.model_dump(mode="json")} for item in evidence.unresolved
    )
    return rows
