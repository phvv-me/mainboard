import abc
import re
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

from patos import FrozenModel

from ...core.errors import MissionError
from .models import (
    AnalysisProfile,
    ArchCoverage,
    ArtifactKind,
    Coverage,
    CubinEntry,
    DeviceResult,
    Evidence,
    JsonValue,
    KernelInventory,
    Limitation,
    Parsed,
    ResidualUnknown,
    SassListing,
    SassResult,
    coverage_of,
    digest,
)
from .parsers import normalize_arch, parse_list_elf, parse_list_ptx, parse_resources, parse_sass
from .providers import Availability, Effects, Request, ReverseProvider
from .runner import Runner, ToolOutput, ToolRunner

# These standard toolkit spellings are checked against the installed tool's --help before
# analysis. TODO(lead): exercise these contracts on real CUDA 13.x binaries and library fixtures.
STANDARD_FLAGS = {
    "cuobjdump": (
        "--list-elf",
        "--list-ptx",
        "--dump-sass",
        "--function",
        "--dump-resource-usage",
        "--extract-elf",
    ),
    # -c is --print-code, not a CFG output flag; this provider parses SASS text only.
    "nvdisasm": ("-c",),
}
_READ_ONLY = Effects(read_only=True, executes_foreign_code=False, destructive=False)
_IMAGES = frozenset(
    {
        ArtifactKind.native_elf,
        ArtifactKind.native_macho,
        ArtifactKind.fatbin,
        ArtifactKind.cubin,
    }
)


class _Identity(FrozenModel):
    availability: Availability
    version: str = "unknown"


class _CudaProvider(ReverseProvider, abc.ABC):
    kinds: ClassVar[frozenset[str]] = frozenset({"device"})

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner if runner is not None else ToolRunner()

    @cached_property
    def identity(self) -> _Identity:
        version = self.runner.run(self.tool, ("--version",))
        help_ = self.runner.run(self.tool, ("--help",))
        for output in (version, help_):
            if output.returncode or output.truncated:
                return _Identity(
                    availability=Availability(
                        ok=False,
                        reason=output.stderr or "tool did not return complete version/help",
                        remediation=f"Install the CUDA toolkit and put {self.tool} on PATH",
                    )
                )
        help_text = help_.stdout + help_.stderr
        missing = tuple(
            flag
            for flag in STANDARD_FLAGS[self.tool]
            if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text)
        )
        found = re.search(r"\bV(\d+(?:\.\d+)+)", version.stdout + version.stderr)
        if not found:
            found = re.search(r"\brelease\s+(\d+(?:\.\d+)+)", version.stdout + version.stderr)
        if missing or not found:
            return _Identity(
                availability=Availability(
                    ok=False,
                    reason=f"unrecognized toolkit version or flags: {', '.join(missing)}",
                    remediation=f"Inspect {self.tool} --help and --version; update the contract",
                )
            )
        return _Identity(availability=Availability(ok=True), version=found[1])

    def available(self) -> Availability:
        return self.identity.availability

    def _validate(self, request: Request) -> None:
        if request.artifact.kind not in self.artifacts or request.operation not in self.operations:
            raise MissionError(
                f"{self.tool} cannot analyze {request.artifact.kind}/{request.operation}"
            )
        if request.arch:
            normalize_arch(request.arch)
        status = self.available()
        if not status.ok:
            raise MissionError(f"{status.reason}; {status.remediation}")

    def _evidence(
        self,
        request: Request,
        parsed: Parsed[DeviceResult],
        outputs: tuple[ToolOutput, ...],
        *,
        options: Mapping[str, JsonValue] | None = None,
    ) -> Evidence[DeviceResult]:
        result = parsed.result
        match result:
            case KernelInventory():
                kernels = tuple(
                    kernel
                    for kernel in result.kernels
                    if (not request.arch or kernel.arch == request.arch)
                    and (not request.function or kernel.name == request.function)
                )
                result = KernelInventory(
                    kernels=kernels,
                    arches=tuple(
                        target
                        for target in result.arches
                        if not request.arch or target == request.arch
                    ),
                )
                empty = not kernels
            case SassResult():
                result = SassResult(
                    listings=tuple(
                        listing
                        for listing in result.listings
                        if (not request.arch or listing.arch == request.arch)
                        and (not request.function or listing.kernel == request.function)
                    )
                )
                empty = not result.listings
            case ArchCoverage():
                result = ArchCoverage(
                    arches=tuple(
                        target
                        for target in result.arches
                        if not request.arch or target == request.arch
                    ),
                    ptx_arches=tuple(
                        target
                        for target in result.ptx_arches
                        if not request.arch or target == request.arch
                    ),
                    cubins=tuple(
                        entry
                        for entry in result.cubins
                        if not request.arch or entry.arch == request.arch
                    ),
                )
                empty = not result.arches and not result.ptx_arches
            case _:
                raise ValueError("CUDA providers do not produce inventory diffs")
        statuses = [parsed.coverage]
        limitations: list[Limitation] = []
        unknowns = list(parsed.unresolved)
        for output in outputs:
            if output.truncated:
                statuses.append(Coverage.truncated)
                limitations.append(
                    Limitation(
                        kind="truncated",
                        subject=self.tool,
                        detail="Tool timed out or exceeded the retained-output limit",
                        remediation="Narrow the selection or increase the runner limits and rerun",
                    )
                )
            elif output.returncode:
                statuses.append(Coverage.failed)
            elif output.stderr:
                statuses.append(Coverage.partial)
            if output.stderr or output.returncode:
                limitations.append(
                    Limitation(
                        kind="tool_diagnostic",
                        subject=self.tool,
                        detail=f"exit={output.returncode}: {output.stderr}",
                        remediation="Inspect the diagnostic; rerun with a supported artifact",
                    )
                )
        if empty and (request.arch or request.function):
            statuses.append(Coverage.partial)
            unknowns.append(
                ResidualUnknown(
                    site=f"{request.arch}/{request.function}",
                    question="Does the selected code exist?",
                    reason="The tool output contains no matching selection",
                )
            )
        params = dict(options) if options is not None else request.options()
        return Evidence[DeviceResult](
            artifact=request.artifact,
            provider=self.tool,
            tool_version=self.identity.version,
            operation=request.operation,
            params=params,
            result=result,
            coverage=coverage_of(tuple(statuses)),
            analysis_profile=AnalysisProfile(
                provider=self.tool,
                tool_version=self.identity.version,
                options_digest=digest(params),
            ),
            limitations=tuple(limitations),
            unresolved=tuple(unknowns),
            raw=[output.model_dump(mode="json") for output in outputs],
        )


class CuobjdumpProvider(_CudaProvider):
    tool: ClassVar[str] = "cuobjdump"
    artifacts: ClassVar[frozenset[ArtifactKind]] = _IMAGES | {ArtifactKind.ptx}
    operations: ClassVar[Mapping[str, Effects]] = dict.fromkeys(
        ("inventory", "sass", "arch", "resource"),
        _READ_ONLY,
    )

    def analyze(self, request: Request) -> Evidence[DeviceResult]:
        self._validate(request)
        outputs: list[ToolOutput] = []
        path = request.artifact.path
        match request.operation:
            case "sass":
                args: tuple[str, ...] = ("--dump-sass",)
                if request.function:
                    args += ("--function", request.function)
                output = self.runner.run(self.tool, (*args, path))
                outputs.append(output)
                sass = parse_sass(output.stdout, truncated=output.truncated)
                parsed = Parsed[DeviceResult](
                    result=sass.result,
                    coverage=sass.coverage,
                    unresolved=sass.unresolved,
                )
            case "resource":
                output = self.runner.run(self.tool, ("--dump-resource-usage", path))
                outputs.append(output)
                resources = parse_resources(output.stdout, truncated=output.truncated)
                parsed = Parsed[DeviceResult](
                    result=resources.result,
                    coverage=resources.coverage,
                    unresolved=resources.unresolved,
                )
            case "arch" | "inventory":
                elf_out = self.runner.run(self.tool, ("--list-elf", path))
                ptx_out = self.runner.run(self.tool, ("--list-ptx", path))
                outputs.extend((elf_out, ptx_out))
                elf = parse_list_elf(elf_out.stdout, truncated=elf_out.truncated)
                ptx = parse_list_ptx(ptx_out.stdout, truncated=ptx_out.truncated)
                arches = ArchCoverage(
                    arches=elf.result.arches,
                    ptx_arches=ptx.result.ptx_arches,
                    cubins=elf.result.cubins,
                )
                statuses: tuple[Coverage, ...] = (elf.coverage, ptx.coverage)
                unknowns = elf.unresolved + ptx.unresolved
                result: DeviceResult = arches
                if request.operation == "inventory":
                    output = self.runner.run(self.tool, ("--dump-resource-usage", path))
                    outputs.append(output)
                    inventory = parse_resources(output.stdout, truncated=output.truncated)
                    result = KernelInventory(
                        kernels=inventory.result.kernels,
                        arches=tuple(sorted(set(arches.arches + inventory.result.arches))),
                    )
                    statuses += (inventory.coverage,)
                    unknowns += inventory.unresolved
                    if arches.ptx_arches:
                        statuses += (Coverage.partial,)
                        unknowns += (
                            ResidualUnknown(
                                site="ptx",
                                question="Which kernels exist only as PTX?",
                                candidates=arches.ptx_arches,
                                reason="PTX was listed; only compiled kernels were inventoried",
                            ),
                        )
                parsed = Parsed[DeviceResult](
                    result=result,
                    coverage=coverage_of(statuses),
                    unresolved=unknowns,
                )
            case _:
                raise MissionError(f"unsupported operation {request.operation}")
        return self._evidence(request, parsed, tuple(outputs))

    def extract(
        self, request: Request, entry: CubinEntry, directory: Path
    ) -> tuple[Path, ToolOutput]:
        """Write into a caller-owned temporary directory, never beside the library."""
        if Path(entry.name).name != entry.name or entry.name in {"", ".", ".."}:
            raise MissionError(f"unsafe embedded image name: {entry.name!r}")
        if "\\" in entry.name:
            raise MissionError(f"unsafe embedded image name: {entry.name!r}")
        output = self.runner.run(
            self.tool,
            ("--extract-elf", entry.name, request.artifact.path),
            cwd=directory,
        )
        return directory / entry.name, output


class NvdisasmProvider(_CudaProvider):
    tool: ClassVar[str] = "nvdisasm"
    artifacts: ClassVar[frozenset[ArtifactKind]] = _IMAGES
    operations: ClassVar[Mapping[str, Effects]] = {"sass": _READ_ONLY}

    def analyze(self, request: Request) -> Evidence[DeviceResult]:
        self._validate(request)
        outputs: list[ToolOutput] = []
        options = request.options()
        if request.artifact.kind == ArtifactKind.cubin:
            outputs.append(self.runner.run(self.tool, ("-c", request.artifact.path)))
            sass = parse_sass(outputs[-1].stdout, truncated=outputs[-1].truncated)
        else:
            extractor = CuobjdumpProvider(self.runner)
            status = extractor.available()
            if not status.ok:
                raise MissionError(f"cubin extraction: {status.reason}; {status.remediation}")
            options["extractor_version"] = extractor.identity.version
            listing = self.runner.run(extractor.tool, ("--list-elf", request.artifact.path))
            outputs.append(listing)
            entries = parse_list_elf(listing.stdout, truncated=listing.truncated)
            selected = tuple(
                entry
                for entry in entries.result.cubins
                if not request.arch or entry.arch == request.arch
            )
            listings: list[SassListing] = []
            unknowns = list(entries.unresolved)
            statuses = [entries.coverage]
            for entry in selected:
                with TemporaryDirectory(prefix="mb-cubin-") as directory:
                    cubin, extracted = extractor.extract(request, entry, Path(directory))
                    outputs.append(extracted)
                    if extracted.returncode or extracted.truncated:
                        continue
                    output = self.runner.run(self.tool, ("-c", str(cubin)), cwd=Path(directory))
                    outputs.append(output)
                    parsed = parse_sass(
                        output.stdout,
                        truncated=output.truncated,
                        arch=entry.arch if entry.arch != "unknown" else "",
                    )
                    listings.extend(parsed.result.listings)
                    unknowns.extend(parsed.unresolved)
                    statuses.append(parsed.coverage)
            if not selected:
                statuses.append(Coverage.partial)
                unknowns.append(
                    ResidualUnknown(
                        site="cubin",
                        question="Which embedded cubin should be disassembled?",
                        reason="No matching ELF image was listed",
                    )
                )
            sass = Parsed[SassResult](
                result=SassResult(listings=tuple(listings)),
                coverage=coverage_of(tuple(statuses)),
                unresolved=tuple(unknowns),
            )
        return self._evidence(
            request,
            Parsed[DeviceResult](
                result=sass.result, coverage=sass.coverage, unresolved=sass.unresolved
            ),
            tuple(outputs),
            options=options,
        )
