from mainboard.probe.reverse import Evidence
from mainboard.probe.reverse.models import (
    AnalysisProfile,
    Artifact,
    ArtifactKind,
    Coverage,
    JsonValue,
    Kernel,
    KernelInventory,
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
