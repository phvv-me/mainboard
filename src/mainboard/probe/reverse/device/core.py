import abc
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

from patos import FrozenModel

from ....core.errors import MissionError
from ....core.execution.tool import Runner, ToolOutput, ToolRunner
from ..constants import STANDARD_FLAGS
from ..models import (
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
    SassResult,
    coverage_of,
    digest,
)
from ..providers import Availability, Effects, Request, ReverseProvider
from .parsers import normalize_arch, parse_list_elf, parse_list_ptx, parse_resources, parse_sass

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

    @staticmethod
    def _narrowed(result: DeviceResult, *, arch: str, function: str) -> tuple[DeviceResult, bool]:
        """`result` narrowed to the chosen architecture and function, and whether none is left."""
        match result:
            case KernelInventory():
                kernels = tuple(
                    kernel
                    for kernel in result.kernels
                    if _wanted(kernel.arch, selected=arch)
                    and _wanted(kernel.name, selected=function)
                )
                arches = tuple(
                    target for target in result.arches if _wanted(target, selected=arch)
                )
                return KernelInventory(kernels=kernels, arches=arches), not kernels
            case SassResult():
                listings = tuple(
                    listing
                    for listing in result.listings
                    if _wanted(listing.arch, selected=arch)
                    and _wanted(listing.kernel, selected=function)
                )
                return SassResult(listings=listings), not listings
            case ArchCoverage():
                narrowed = ArchCoverage(
                    arches=tuple(
                        target for target in result.arches if _wanted(target, selected=arch)
                    ),
                    ptx_arches=tuple(
                        target for target in result.ptx_arches if _wanted(target, selected=arch)
                    ),
                    cubins=tuple(
                        entry for entry in result.cubins if _wanted(entry.arch, selected=arch)
                    ),
                )
                return narrowed, not narrowed.arches and not narrowed.ptx_arches
        raise ValueError("CUDA providers do not produce inventory diffs")

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner if runner is not None else ToolRunner()

    def _assessed(self, outputs: Iterable[ToolOutput]) -> tuple[list[Coverage], list[Limitation]]:
        """What each tool run says of the coverage, and the limitations it brings."""
        statuses: list[Coverage] = []
        limitations: list[Limitation] = []
        for output in outputs:
            status = (
                Coverage.truncated
                if output.truncated
                else Coverage.failed
                if output.returncode
                else Coverage.partial
                if output.stderr
                else None
            )
            statuses += [status] if status is not None else []
            limitations += [
                *([self._truncation()] if output.truncated else []),
                *([self._diagnostic(output)] if output.stderr or output.returncode else []),
            ]
        return statuses, limitations

    def _diagnostic(self, output: ToolOutput) -> Limitation:
        return Limitation(
            kind="tool_diagnostic",
            subject=self.tool,
            detail=f"exit={output.returncode}: {output.stderr}",
            remediation="Inspect the diagnostic; rerun with a supported artifact",
        )

    def _evidence(
        self,
        request: Request,
        parsed: Parsed[DeviceResult],
        outputs: Iterable[ToolOutput],
        *,
        options: Mapping[str, JsonValue] | None = None,
    ) -> Evidence[DeviceResult]:
        result, empty = _CudaProvider._narrowed(
            parsed.result, arch=request.arch, function=request.function
        )
        statuses, limitations = self._assessed(outputs)
        statuses.insert(0, parsed.coverage)
        unknowns = list(parsed.unresolved)
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

    def _truncation(self) -> Limitation:
        return Limitation(
            kind="truncated",
            subject=self.tool,
            detail="Tool timed out or exceeded the retained-output limit",
            remediation="Narrow the selection or increase the runner limits and rerun",
        )

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

    def available(self) -> Availability:
        return self.identity.availability


class CuobjdumpProvider(_CudaProvider):
    tool: ClassVar[str] = "cuobjdump"
    artifacts: ClassVar[frozenset[ArtifactKind]] = _IMAGES | {ArtifactKind.ptx}
    operations: ClassVar[Mapping[str, Effects]] = dict.fromkeys(
        ("inventory", "sass", "arch", "resource"),
        _READ_ONLY,
    )

    def _inventory(self, request: Request) -> tuple[Parsed[DeviceResult], list[ToolOutput]]:
        """The compiled kernels with their resources, beside the listed architectures.

        Kernels that exist only as PTX are not inventoried, which the coverage says.
        """
        listed, outputs = self._listed(request)
        arches = listed.result
        assert isinstance(arches, ArchCoverage)
        output = self.runner.run(self.tool, ("--dump-resource-usage", request.artifact.path))
        inventory = parse_resources(output.stdout, truncated=output.truncated)
        only_ptx = (
            ResidualUnknown(
                site="ptx",
                question="Which kernels exist only as PTX?",
                candidates=arches.ptx_arches,
                reason="PTX was listed; only compiled kernels were inventoried",
            ),
        )
        parsed = Parsed[DeviceResult](
            result=KernelInventory(
                kernels=inventory.result.kernels,
                arches=tuple(sorted(set(arches.arches + inventory.result.arches))),
            ),
            coverage=coverage_of(
                (
                    listed.coverage,
                    inventory.coverage,
                    *((Coverage.partial,) if arches.ptx_arches else ()),
                )
            ),
            unresolved=(
                *listed.unresolved,
                *inventory.unresolved,
                *(only_ptx if arches.ptx_arches else ()),
            ),
        )
        return parsed, [*outputs, output]

    def _listed(self, request: Request) -> tuple[Parsed[DeviceResult], list[ToolOutput]]:
        """The architectures the ELF images and the PTX listings carry."""
        path = request.artifact.path
        elf_out = self.runner.run(self.tool, ("--list-elf", path))
        ptx_out = self.runner.run(self.tool, ("--list-ptx", path))
        elf = parse_list_elf(elf_out.stdout, truncated=elf_out.truncated)
        ptx = parse_list_ptx(ptx_out.stdout, truncated=ptx_out.truncated)
        parsed = Parsed[DeviceResult](
            result=ArchCoverage(
                arches=elf.result.arches,
                ptx_arches=ptx.result.ptx_arches,
                cubins=elf.result.cubins,
            ),
            coverage=coverage_of((elf.coverage, ptx.coverage)),
            unresolved=elf.unresolved + ptx.unresolved,
        )
        return parsed, [elf_out, ptx_out]

    def _resource(self, request: Request) -> tuple[Parsed[DeviceResult], list[ToolOutput]]:
        output = self.runner.run(self.tool, ("--dump-resource-usage", request.artifact.path))
        resources = parse_resources(output.stdout, truncated=output.truncated)
        parsed = Parsed[DeviceResult](
            result=resources.result, coverage=resources.coverage, unresolved=resources.unresolved
        )
        return parsed, [output]

    def _sass(self, request: Request) -> tuple[Parsed[DeviceResult], list[ToolOutput]]:
        selected = ("--function", request.function) if request.function else ()
        output = self.runner.run(self.tool, ("--dump-sass", *selected, request.artifact.path))
        sass = parse_sass(output.stdout, truncated=output.truncated)
        parsed = Parsed[DeviceResult](
            result=sass.result, coverage=sass.coverage, unresolved=sass.unresolved
        )
        return parsed, [output]

    def analyze(self, request: Request) -> Evidence[DeviceResult]:
        self._validate(request)
        reads = {
            "sass": self._sass,
            "resource": self._resource,
            "arch": self._listed,
            "inventory": self._inventory,
        }
        try:
            read = reads[request.operation]
        except KeyError:
            raise MissionError(f"unsupported operation {request.operation}") from None
        parsed, outputs = read(request)
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

    @staticmethod
    def _joined(
        entries: Parsed[ArchCoverage], found: Sequence[Parsed[SassResult]], *, listed: bool
    ) -> Parsed[SassResult]:
        """The listings of every disassembled image as one result.

        A selection that listed no image is partial, and asks which one was meant.
        """
        unmatched = ResidualUnknown(
            site="cubin",
            question="Which embedded cubin should be disassembled?",
            reason="No matching ELF image was listed",
        )
        return Parsed[SassResult](
            result=SassResult(
                listings=tuple(item for parsed in found for item in parsed.result.listings)
            ),
            coverage=coverage_of(
                (
                    entries.coverage,
                    *(parsed.coverage for parsed in found),
                    *(() if listed else (Coverage.partial,)),
                )
            ),
            unresolved=(
                *entries.unresolved,
                *(unknown for parsed in found for unknown in parsed.unresolved),
                *(() if listed else (unmatched,)),
            ),
        )

    def _disassembled(
        self, extractor: CuobjdumpProvider, request: Request, entry: CubinEntry
    ) -> tuple[Parsed[SassResult] | None, list[ToolOutput]]:
        """One embedded image extracted and disassembled; None when it would not extract."""
        with TemporaryDirectory(prefix="mb-cubin-") as directory:
            cubin, extracted = extractor.extract(request, entry, Path(directory))
            if extracted.returncode or extracted.truncated:
                return None, [extracted]
            output = self.runner.run(self.tool, ("-c", str(cubin)), cwd=Path(directory))
        parsed = parse_sass(
            output.stdout,
            truncated=output.truncated,
            arch=entry.arch if entry.arch != "unknown" else "",
        )
        return parsed, [extracted, output]

    def _extracted(
        self, request: Request, options: dict[str, JsonValue]
    ) -> tuple[Parsed[SassResult], list[ToolOutput]]:
        """Every embedded image of the selected architecture, extracted by cuobjdump."""
        extractor = self._extractor(options)
        listing = self.runner.run(extractor.tool, ("--list-elf", request.artifact.path))
        entries = parse_list_elf(listing.stdout, truncated=listing.truncated)
        selected = [
            entry for entry in entries.result.cubins if _wanted(entry.arch, selected=request.arch)
        ]
        read = [self._disassembled(extractor, request, entry) for entry in selected]
        found = [parsed for parsed, _ in read if parsed is not None]
        sass = NvdisasmProvider._joined(entries, found, listed=bool(selected))
        return sass, [listing, *(output for _, outputs in read for output in outputs)]

    def _extractor(self, options: dict[str, JsonValue]) -> CuobjdumpProvider:
        """cuobjdump, refused when absent; its version joins `options`, shaping the evidence."""
        extractor = CuobjdumpProvider(self.runner)
        status = extractor.available()
        if not status.ok:
            raise MissionError(f"cubin extraction: {status.reason}; {status.remediation}")
        options["extractor_version"] = extractor.identity.version
        return extractor

    def analyze(self, request: Request) -> Evidence[DeviceResult]:
        self._validate(request)
        outputs: list[ToolOutput] = []
        options = request.options()
        if request.artifact.kind == ArtifactKind.cubin:
            outputs.append(self.runner.run(self.tool, ("-c", request.artifact.path)))
            sass = parse_sass(outputs[-1].stdout, truncated=outputs[-1].truncated)
        else:
            sass, extracted = self._extracted(request, options)
            outputs.extend(extracted)
        return self._evidence(
            request,
            Parsed[DeviceResult](
                result=sass.result, coverage=sass.coverage, unresolved=sass.unresolved
            ),
            tuple(outputs),
            options=options,
        )


def _wanted(value: str, *, selected: str) -> bool:
    """Whether `value` is the one selected, any value when nothing is."""
    return not selected or value == selected
