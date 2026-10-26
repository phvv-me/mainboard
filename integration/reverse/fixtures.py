from collections.abc import Sequence
from pathlib import Path

from mainboard.core import Runner, ToolOutput
from mainboard.probe.reverse import Evidence
from mainboard.probe.reverse.constants import STANDARD_FLAGS
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

from ..constants import ELF, NVDISASM, RESOURCE, SASS

_NO_PTX = "cuobjdump info    : No PTX file found to extract from 'lib.so'\n"


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


class FixtureRunner(Runner):
    def __init__(self, outputs: dict[str, ToolOutput] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, tuple[str, ...], Path | None]] = []

    def run(self, tool: str, args: Sequence[str], *, cwd: Path | None = None) -> ToolOutput:
        self.calls.append((tool, tuple(args), cwd))
        if args[0] in self.outputs:
            return self.outputs[args[0]]
        match args[0]:
            case "--version":
                return ToolOutput(stdout="Cuda compilation tools, release 13.4, V13.4.48\n")
            case "--help":
                return ToolOutput(stdout="\n".join(STANDARD_FLAGS[tool]))
            case "--list-elf":
                return ToolOutput(stdout=ELF)
            case "--list-ptx":
                return ToolOutput(stdout=_NO_PTX)
            case "--dump-resource-usage":
                return ToolOutput(stdout=RESOURCE)
            case "--dump-sass":
                return ToolOutput(stdout=SASS)
            case "--extract-elf":
                return ToolOutput()
            case "-c":
                return ToolOutput(stdout=NVDISASM)
            case _:
                raise AssertionError(f"unexpected fixture command: {tool} {args}")
