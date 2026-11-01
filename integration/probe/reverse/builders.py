from mainboard.probe.reverse import Evidence
from mainboard.probe.reverse.evidence import (
    AnalysisProfile,
    Artifact,
    ArtifactKind,
    Confidence,
    Coverage,
    Fact,
    JsonValue,
    Kernel,
    KernelInventory,
    Provenance,
    digest,
)


def artifact() -> Artifact:
    return Artifact(path="/fixture/lib.so", sha256="a" * 64, size=64, kind=ArtifactKind.native_elf)


def evidence(params: dict[str, JsonValue] | None = None) -> Evidence[KernelInventory]:
    options: dict[str, JsonValue] = {} if params is None else params
    return Evidence[KernelInventory](
        artifact=artifact(),
        provider="fixture",
        tool_version="13.4.0",
        operation="inventory",
        params=options,
        result=KernelInventory(kernels=(Kernel(name="k", arch="sm_90a"),), arches=("sm_90a",)),
        coverage=Coverage.complete,
        analysis_profile=AnalysisProfile(
            provider="fixture", tool_version="13.4.0", options_digest=digest(options)
        ),
        raw={"stdout": "fixture"},
    )


def inventories() -> tuple[KernelInventory, KernelInventory]:
    """An inventory and its successor, in kernel order (kept, changed, then removed or added).

    The successor observes 32 registers of the changed kernel, drops `old`, adds `new` on sm_100.
    """
    unchanged = Kernel(name="same", arch="sm_90a")
    older = Kernel(name="changed", arch="sm_90a")
    newer = older.model_copy(
        update={
            "registers": Fact[int | None](
                value=32,
                confidence=Confidence.observed,
                provenance=Provenance.IMPORTED,
            )
        }
    )
    removed, added = Kernel(name="old", arch="sm_90a"), Kernel(name="new", arch="sm_100")
    before = KernelInventory(kernels=(unchanged, older, removed), arches=("sm_90a",))
    after = KernelInventory(kernels=(added, newer, unchanged), arches=("sm_90a", "sm_100"))
    return before, after
