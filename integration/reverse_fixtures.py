from pathlib import Path

from mainboard.core.tool import ToolOutput
from mainboard.probe.reverse.device import STANDARD_FLAGS
from mainboard.probe.reverse.models import (
    AnalysisProfile,
    Artifact,
    ArtifactKind,
    Coverage,
    Evidence,
    JsonValue,
    Kernel,
    KernelInventory,
    digest,
)

ELF = """ELF file    1: lib.1.sm_90a.cubin
ELF file    2: lib.2.sm_100.cubin
"""
PTX = "PTX file    1: lib.1.sm_90.ptx\n"
NO_PTX = "cuobjdump info    : No PTX file found to extract from 'lib.so'\n"
RESOURCE = """Fatbin elf code:
================
arch = sm_90a
code version = [1,7]
host = linux
compile_size = 64bit
Resource usage:
 Common:
  GLOBAL:0 CONSTANT[3]:0
 Function _Z6kernelPi:
  REG:32 STACK:0 SHARED:128 LOCAL:0 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z5otherPi:
  REG:16 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:360
"""
SASS = """Fatbin elf code:
================
arch = sm_90a
code version = [1,7]
host = linux
compile_size = 64bit
    Function : _Z6kernelPi
.headerflags    @\"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM90\"
        /*0000*/                   MOV R1, c[0x0][0x28] ; /* 0x00000a0000017a02 */
                                                         /* 0x000fc40000000f00 */
        /*0010*/              @!P0 BRX R2 -0x10 ;         /* 0x0000000002007950 */
        /*0020*/                   EXIT ;                /* 0x000000000000794d */
"""
# A real `cuobjdump --dump-sass` fatbin dump: an empty compressed section, then a populated one,
# carrying the per-image headers a live toolkit prints (`compressed`, a `ptxasOptions` line, the
# tab-indented `code for`/`.target` directives) and the trailing progress dots. None is an
# instruction; a regression that stops treating them as metadata resurfaces them as false unknowns.
SASS_REAL = "\n".join(
    [
        "",
        "Fatbin elf code:",
        "================",
        "arch = sm_75",
        "code version = [1,8]",
        "host = linux",
        "compile_size = 64bit",
        "compressed",
        "",
        "\tcode for sm_75",
        "\t.target\tsm_75",
        "",
        "",
        "Fatbin elf code:",
        "================",
        "arch = sm_90a",
        "code version = [1,8]",
        "host = linux",
        "compile_size = 64bit",
        "compressed",
        "ptxasOptions = --allow-expensive-optimizations=true",
        "",
        "\tcode for sm_90a",
        "\t.target\tsm_90a",
        "\t\tFunction : _Z6kernelPi",
        '.headerflags\t@"EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)"',
        "        /*0000*/                   MOV R1, c[0x0][0x28] ;  /* 0x00000a0000017a02 */",
        "                                                          /* 0x000fc60000000f00 */",
        "        /*0010*/                   EXIT ;                  /* 0x000000000000794d */",
        "                                                          /* 0x000fea0003800000 */",
        "        /*0020*/                   BRA 0x20 ;              /* 0xfffffff000007947 */",
        "                                                          /* 0x000fc0000383ffff */",
        "\t\t..........",
        "",
    ]
)
NVDISASM = """.headerflags @\"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM90\"
.elftype @\"ET_EXEC\"
.section .text._Z6kernelPi,\"ax\",@progbits
.sectioninfo @\"SHI_REGISTERS=32\"
.align 128
.global _Z6kernelPi
.type _Z6kernelPi,@function
_Z6kernelPi:
/*0000*/ MOV R1, c[0x0][0x28] ;
/*0010*/ @P0 BRX R2 ;
/*0020*/ EXIT ;
"""


def artifact() -> Artifact:
    return Artifact(path="/fixture/lib.so", sha256="a" * 64, size=64, kind=ArtifactKind.native_elf)


def evidence(params: dict[str, JsonValue] | None = None) -> Evidence[KernelInventory]:
    options = {} if params is None else params
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


class FixtureRunner:
    def __init__(self, outputs: dict[str, ToolOutput] | None = None) -> None:
        self.outputs = outputs or {}
        self.calls: list[tuple[str, tuple[str, ...], Path | None]] = []

    def run(self, tool: str, args: tuple[str, ...], *, cwd: Path | None = None) -> ToolOutput:
        self.calls.append((tool, args, cwd))
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
                return ToolOutput(stdout=NO_PTX)
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
