import re
from collections.abc import Sequence
from typing import Self

from ....dispatch.jobs.arch import arch as architecture
from ....dispatch.jobs.arch import sm
from ..models import (
    ArchCoverage,
    Confidence,
    Coverage,
    CubinEntry,
    Fact,
    Kernel,
    KernelInventory,
    Parsed,
    Provenance,
    ResidualUnknown,
    SassInsn,
    SassListing,
    SassResult,
)

_ARCH = re.compile(r"\b(?:sm_?|EF_CUDA_SM)(\d{2,3}[af]?)\b", re.IGNORECASE)
_ENTRY = re.compile(r"(?:ELF|PTX) file\s+\d+\s*:\s*(.+)", re.IGNORECASE)
_FUNCTION = re.compile(r"Function\s*:?\s+(.+?)\s*:?$")
_SECTION = re.compile(r"\.section\s+\.text\.([^,\s]+)")
_INSN = re.compile(
    r"\s*/\*\s*([0-9a-fA-F]+)\s*\*/\s*"
    r"(?:(@!?[A-Za-z][\w]*)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
)
_ENCODING = re.compile(r"(?:\s*/\*\s*(?:0x)?[0-9a-fA-F]+\s*\*/\s*)+")
_RESOURCE = re.compile(r"\b(REG|SHARED):\s*(\d+)(?=\s|$)")
_RESOURCE_TOKEN = re.compile(
    r"\b(?:REG|SHARED|STACK|LOCAL|GLOBAL|CONSTANT\[\d+\]|TEXTURE|SURFACE|SAMPLER):\s*\d+(?=\s|$)"
)
_CUT = re.compile(r"\btruncated\b|\[\.\.\.\]|^\s*\.\.\.\s*$", re.IGNORECASE | re.MULTILINE)
# cuobjdump prints a run of progress dots per image; four or more never spells the three-dot
# truncation ellipsis `_CUT` watches for, so this stays noise rather than a missing-content signal.
# What a header or listing naming no known target asks.
_WHICH_TARGET = "Which CUDA target is this?"
_PROGRESS = re.compile(r"\.{4,}")
_METADATA = (
    "Fatbin ",
    "arch =",
    "code version",
    "host =",
    "compile_size",
    "compressed",
    "identifier",
    # Per-architecture fatbin section headers a real cuobjdump --dump-sass prints before each
    # image: `code for sm_90`, a ptxas `.target` line, and the `ptxasOptions = ...` the image
    # was built with. They carry no instruction, and the arch already arrives via `arch =`.
    "code for",
    ".target",
    "ptxasOptions",
    ".headerflags",
    ".elftype",
    ".sectioninfo",
    ".sectionflags",
    ".align",
    ".global",
    ".type",
    ".size",
    ".other",
    "//",
    "Resource usage:",
    "Common:",
)


def normalize_arch(value: str) -> str:
    """Reuse dispatch's capability parser, retaining CUDA's architecture-specific suffix."""
    spelled = value.strip().lower()
    parsed = architecture(spelled)
    if parsed.low != parsed.high or not re.fullmatch(r"sm_\d{2,3}[af]?", spelled):
        raise ValueError(f"expected an exact CUDA target, got {value!r}")
    suffix = spelled[-1] if spelled[-1] in "af" else ""
    return sm(parsed.low) + suffix


def _arch(line: str) -> str:
    found = _ARCH.search(line)
    return normalize_arch(f"sm_{found[1].lower()}") if found else "unknown"


def _unknown(
    site: str, *, reason: str, question: str = "What does this output describe?"
) -> ResidualUnknown:
    return ResidualUnknown(site=site, question=question, reason=reason)


def _coverage(text: str, unknowns: Sequence[ResidualUnknown], *, truncated: bool) -> Coverage:
    if truncated or _CUT.search(text):
        return Coverage.truncated
    return Coverage.partial if unknowns else Coverage.complete


def _is_metadata(line: str) -> bool:
    return (
        not line
        or line.startswith(_METADATA)
        or set(line) <= {"=", "-", " "}
        or bool(_PROGRESS.fullmatch(line))
        or bool(_ENCODING.fullmatch(line))
    )


class _Entries:
    """The images one `cuobjdump --list-elf` or `--list-ptx` output names, read line by line."""

    @classmethod
    def of(cls, text: str) -> Self:
        """The images `text` lists."""
        read = cls()
        for number, raw in enumerate(text.splitlines(), 1):
            read.read(number, raw)
        return read

    def __init__(self) -> None:
        self.entries: list[CubinEntry] = []
        self.unknowns: list[ResidualUnknown] = []
        self.empty = False

    def arches(self) -> tuple[str, ...]:
        return tuple(sorted({entry.arch for entry in self.entries if entry.arch != "unknown"}))

    def read(self, number: int, raw: str) -> None:
        line = raw.strip()
        if match := _ENTRY.fullmatch(line):
            self.entries.append(CubinEntry(name=match[1], arch=_arch(match[1])))
            if self.entries[-1].arch == "unknown":
                self.unknowns.append(
                    _unknown(f"line:{number}", reason=line, question=_WHICH_TARGET)
                )
        elif re.search(r"\bno (?:elf|ptx) (?:file|code)", line, re.IGNORECASE):
            self.empty = True
        elif not _is_metadata(line):
            self.unknowns.append(_unknown(f"line:{number}", reason=line))

    def unresolved(self) -> tuple[ResidualUnknown, ...]:
        """What the output left unanswered, a silent output included."""
        if not self.entries and not self.empty:
            reason = "No image entries or explicit empty result reported"
            return (*self.unknowns, _unknown("inventory", reason=reason))
        return tuple(self.unknowns)


def parse_list_elf(text: str, *, truncated: bool = False) -> Parsed[ArchCoverage]:
    """A listed ELF is an embedded image, never a kernel name."""
    read = _Entries.of(text)
    unresolved = read.unresolved()
    return Parsed(
        result=ArchCoverage(arches=read.arches(), ptx_arches=(), cubins=tuple(read.entries)),
        coverage=_coverage(text, unresolved, truncated=truncated),
        unresolved=unresolved,
    )


def parse_list_ptx(text: str, *, truncated: bool = False) -> Parsed[ArchCoverage]:
    read = _Entries.of(text)
    unresolved = read.unresolved()
    return Parsed(
        result=ArchCoverage(arches=(), ptx_arches=read.arches(), cubins=()),
        coverage=_coverage(text, unresolved, truncated=truncated),
        unresolved=unresolved,
    )


class _Listings:
    """The functions one `cuobjdump --dump-sass` output lists, read line by line.

    arch: the target the output is known to describe, empty until a header names one.
    truncated: whether the output is already known to be cut short.
    """

    def __init__(self, arch: str, *, truncated: bool) -> None:
        self.listings: list[SassListing] = []
        self.unknowns: list[ResidualUnknown] = []
        self.instructions: list[SassInsn] = []
        self.kernel, self.target = "unknown", normalize_arch(arch) if arch else "unknown"
        self.started, self.truncated = False, truncated

    def _closed(self, reason: str) -> bool:
        """End the function being read; answer whether it held no instruction.

        `reason` is noted when it held none.
        """
        listing = SassListing(
            kernel=self.kernel, arch=self.target, instructions=tuple(self.instructions)
        )
        self.listings.append(listing)
        empty = not self.instructions
        if empty:
            self.unknowns.append(_unknown(f"{self.target}/{self.kernel}", reason=reason))
        self.instructions, self.started = [], False
        return empty

    def _flagged(self, arch: str) -> None:
        """Take the target ELF header flags name when it is of another generation.

        The flags can omit the a/f suffix the fatbin entry or arch header carries.
        """
        if self.target == "unknown" or architecture(self.target).low != architecture(arch).low:
            self.target = arch

    def _headed(self, number: int, raw: str, *, function: re.Match[str] | None, arch: str) -> None:
        """End the function before a header and begin what the header names.

        `arch` is the header's target, empty for a function header.
        """
        if self.started:
            self._closed("Function has no instructions")
        if raw.strip().startswith("arch ="):
            self.target, self.kernel = arch, "unknown"
            if arch == "unknown":
                self.unknowns.append(
                    _unknown(f"line:{number}", reason=raw, question=_WHICH_TARGET)
                )
        if function:
            self.kernel, self.started = function[1], True

    def _instruction(self, number: int, raw: str, found: re.Match[str]) -> None:
        address, predicate, opcode, operands = found.groups()
        insn = SassInsn(
            address=int(address, 16),
            opcode=opcode,
            operands=operands,
            predicate=predicate or "",
            raw=raw,
        )
        self.instructions.append(insn)
        self.started = True
        line = raw.strip()
        if self.kernel == "unknown" or self.target == "unknown":
            question = "Which function and target own this instruction?"
            self.unknowns.append(_unknown(f"line:{number}", reason=line, question=question))
        if opcode.split(".")[0] in {"BRX", "BRXU", "JMX", "JMXU"}:
            self.unknowns.append(
                _unknown(
                    f"{self.target}/{self.kernel}/0x{insn.address:x}",
                    reason=(
                        "Static SASS lists an indirect target; no target analysis was performed"
                    ),
                    question="Which instruction addresses can this indirect branch reach?",
                )
            )
        tail = line[found.end() :].strip()
        if tail and not _ENCODING.fullmatch(tail) and not tail.startswith("//"):
            self.unknowns.append(_unknown(f"line:{number}", reason=tail))

    def _unrecognized(self, number: int, raw: str) -> None:
        """A line that is no metadata, brace or label is unresolved; a cut address truncates."""
        line = raw.strip()
        if _is_metadata(line) or line in {"{", "}"} or re.fullmatch(r"[.$\w]+:", line):
            return
        self.unknowns.append(_unknown(f"{self.target}/{self.kernel}/line:{number}", reason=raw))
        if re.match(r"/\*\s*[0-9a-fA-F]+", line) and ";" not in line:
            self.truncated = True

    def parsed(self, text: str) -> Parsed[SassResult]:
        if self.started and self._closed("Output ended before function instructions"):
            self.truncated = True
        if not self.listings:
            self.unknowns.append(_unknown("sass", reason="No function instructions reported"))
        return Parsed(
            result=SassResult(listings=tuple(self.listings)),
            coverage=_coverage(text, self.unknowns, truncated=self.truncated),
            unresolved=tuple(self.unknowns),
        )

    def read(self, number: int, raw: str) -> None:
        line = raw.strip()
        function = _FUNCTION.fullmatch(line) or _SECTION.match(line)
        next_arch = _arch(line)
        if ".headerflags" in line and next_arch != "unknown":
            self._flagged(next_arch)
        elif function or line.startswith("arch ="):
            self._headed(number, raw, function=function, arch=next_arch)
        elif found := _INSN.match(line):
            self._instruction(number, raw, found)
        else:
            self._unrecognized(number, raw)


def parse_sass(text: str, *, truncated: bool = False, arch: str = "") -> Parsed[SassResult]:
    read = _Listings(arch, truncated=truncated)
    for number, raw in enumerate(text.splitlines(), 1):
        read.read(number, raw)
    return read.parsed(text)


class _Resources:
    """The registers and shared memory each kernel of one `cuobjdump --dump-resource-usage`
    output declares, read line by line."""

    def __init__(self) -> None:
        self.kernels: dict[tuple[str, str], Kernel] = {}
        self.unknowns: list[ResidualUnknown] = []
        self.target, self.name = "unknown", ""
        self.resources: dict[str, int] = {}

    def _counted(self, number: int, raw: str) -> None:
        """Keep the counts for the kernel being read; anything else on the line is unresolved."""
        line = raw.strip()
        if self.name:
            self.resources.update({key: int(value) for key, value in _RESOURCE.findall(line)})
        if _RESOURCE_TOKEN.sub("", line).strip():
            self.unknowns.append(_unknown(f"{self.target}/{self.name}/line:{number}", reason=raw))

    def _headed(self, number: int, raw: str, *, function: re.Match[str] | None) -> None:
        """Keep the kernel before a header and begin what the header names."""
        line = raw.strip()
        if self.name:
            self._kept()
        self.resources, self.name = {}, function[1] if function else ""
        if line.startswith("arch ="):
            self.target = _arch(line)
            if self.target == "unknown":
                self.unknowns.append(
                    _unknown(f"line:{number}", reason=raw, question=_WHICH_TARGET)
                )

    def _kept(self) -> None:
        """Keep the kernel just read, noting a conflicting record or a missing count."""
        facts = {
            field: Fact[int | None](
                value=self.resources.get(key),
                confidence=Confidence.observed if key in self.resources else Confidence.uncertain,
                provenance=Provenance.IMPORTED if key in self.resources else Provenance.DEFAULT,
            )
            for key, field in (("REG", "registers"), ("SHARED", "shared_bytes"))
        }
        kernel = Kernel(name=self.name, arch=self.target, **facts)
        site = f"{self.target}/{self.name}"
        if self.kernels.setdefault((self.name, self.target), kernel) != kernel:
            reason = "Conflicting resource records for this kernel"
            self.unknowns.append(_unknown(site, reason=reason))
        if self.target == "unknown" or any(key not in self.resources for key in ("REG", "SHARED")):
            reason = "Architecture or resource counts were not reported"
            self.unknowns.append(_unknown(site, reason=reason))

    def parsed(self, text: str, *, truncated: bool) -> Parsed[KernelInventory]:
        if not self.kernels:
            self.unknowns.append(_unknown("resource", reason="No kernel resources reported"))
        kernels = [self.kernels[key] for key in sorted(self.kernels)]
        return Parsed(
            result=KernelInventory(
                kernels=tuple(kernels),
                arches=tuple(sorted({k.arch for k in kernels if k.arch != "unknown"})),
            ),
            coverage=_coverage(text, self.unknowns, truncated=truncated),
            unresolved=tuple(self.unknowns),
        )

    def read(self, number: int, raw: str) -> None:
        line = raw.strip()
        function = _FUNCTION.fullmatch(line)
        if function or line.startswith("arch ="):
            self._headed(number, raw, function=function)
        elif _RESOURCE_TOKEN.search(line):
            self._counted(number, raw)
        elif not _is_metadata(line):
            self.unknowns.append(_unknown(f"line:{number}", reason=raw))


def parse_resources(text: str, *, truncated: bool = False) -> Parsed[KernelInventory]:
    read = _Resources()
    # The sentinel flushes the final function using the same path as every preceding one.
    for number, raw in enumerate([*text.splitlines(), "Function : __end__"], 1):
        read.read(number, raw)
    return read.parsed(text, truncated=truncated)
