import re

from ...dispatch.arch import arch as architecture
from ...dispatch.arch import sm
from .models import (
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
    site: str, reason: str, question: str = "What does this output describe?"
) -> ResidualUnknown:
    return ResidualUnknown(site=site, question=question, reason=reason)


def _coverage(text: str, unknowns: list[ResidualUnknown], truncated: bool) -> Coverage:
    if truncated or _CUT.search(text):
        return Coverage.truncated
    return Coverage.partial if unknowns else Coverage.complete


def _metadata(line: str) -> bool:
    return (
        not line
        or line.startswith(_METADATA)
        or set(line) <= {"=", "-", " "}
        or bool(_PROGRESS.fullmatch(line))
        or bool(_ENCODING.fullmatch(line))
    )


def parse_list_elf(text: str, *, truncated: bool = False) -> Parsed[ArchCoverage]:
    """A listed ELF is an embedded image, never a kernel name."""
    return _parse_entries(text, ptx=False, truncated=truncated)


def parse_list_ptx(text: str, *, truncated: bool = False) -> Parsed[ArchCoverage]:
    return _parse_entries(text, ptx=True, truncated=truncated)


def _parse_entries(text: str, *, ptx: bool, truncated: bool) -> Parsed[ArchCoverage]:
    entries: list[CubinEntry] = []
    unknowns: list[ResidualUnknown] = []
    empty = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if match := _ENTRY.fullmatch(line):
            target = _arch(match[1])
            entries.append(CubinEntry(name=match[1], arch=target))
            if target == "unknown":
                unknowns.append(_unknown(f"line:{number}", line, "Which CUDA target is this?"))
        elif re.search(r"\bno (?:elf|ptx) (?:file|code)", line, re.IGNORECASE):
            empty = True
        elif not _metadata(line):
            unknowns.append(_unknown(f"line:{number}", line))
    if not entries and not empty:
        unknowns.append(
            _unknown("inventory", "No image entries or explicit empty result reported")
        )
    arches = tuple(sorted({entry.arch for entry in entries if entry.arch != "unknown"}))
    return Parsed(
        result=ArchCoverage(
            arches=() if ptx else arches,
            ptx_arches=arches if ptx else (),
            cubins=() if ptx else tuple(entries),
        ),
        coverage=_coverage(text, unknowns, truncated),
        unresolved=tuple(unknowns),
    )


def parse_sass(text: str, *, truncated: bool = False, arch: str = "") -> Parsed[SassResult]:
    listings: list[SassListing] = []
    unknowns: list[ResidualUnknown] = []
    instructions: list[SassInsn] = []
    kernel, target = "unknown", normalize_arch(arch) if arch else "unknown"
    started = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        function = _FUNCTION.fullmatch(line) or _SECTION.match(line)
        next_arch = _arch(line)
        if ".headerflags" in line and next_arch != "unknown":
            # ELF flags can omit the a/f suffix carried by the fatbin entry or arch header.
            if target == "unknown" or architecture(target).low != architecture(next_arch).low:
                target = next_arch
            continue
        arch_header = line.startswith("arch =")
        if function or arch_header:
            if started:
                listings.append(
                    SassListing(kernel=kernel, arch=target, instructions=tuple(instructions))
                )
                if not instructions:
                    unknowns.append(_unknown(f"{target}/{kernel}", "Function has no instructions"))
                instructions = []
                started = False
            if arch_header:
                target, kernel = next_arch, "unknown"
                if target == "unknown":
                    unknowns.append(_unknown(f"line:{number}", raw, "Which CUDA target is this?"))
            if function:
                kernel = function[1]
                started = True
            continue
        if found := _INSN.match(line):
            address, predicate, opcode, operands = found.groups()
            insn = SassInsn(
                address=int(address, 16),
                opcode=opcode,
                operands=operands,
                predicate=predicate or "",
                raw=raw,
            )
            instructions.append(insn)
            started = True
            if kernel == "unknown" or target == "unknown":
                unknowns.append(
                    _unknown(
                        f"line:{number}", line, "Which function and target own this instruction?"
                    )
                )
            if opcode.split(".")[0] in {"BRX", "BRXU", "JMX", "JMXU"}:
                unknowns.append(
                    _unknown(
                        f"{target}/{kernel}/0x{insn.address:x}",
                        "Static SASS lists an indirect target; no target analysis was performed",
                        "Which instruction addresses can this indirect branch reach?",
                    )
                )
            tail = line[found.end() :].strip()
            if tail and not _ENCODING.fullmatch(tail) and not tail.startswith("//"):
                unknowns.append(_unknown(f"line:{number}", tail))
        elif _metadata(line) or line in {"{", "}"} or re.fullmatch(r"[.$\w]+:", line):
            continue
        else:
            unknowns.append(_unknown(f"{target}/{kernel}/line:{number}", raw))
            if re.match(r"/\*\s*[0-9a-fA-F]+", line) and ";" not in line:
                truncated = True
    if started:
        listings.append(SassListing(kernel=kernel, arch=target, instructions=tuple(instructions)))
        if not instructions:
            unknowns.append(
                _unknown(f"{target}/{kernel}", "Output ended before function instructions")
            )
            truncated = True
    if not listings:
        unknowns.append(_unknown("sass", "No function instructions reported"))
    return Parsed(
        result=SassResult(listings=tuple(listings)),
        coverage=_coverage(text, unknowns, truncated),
        unresolved=tuple(unknowns),
    )


def parse_resources(text: str, *, truncated: bool = False) -> Parsed[KernelInventory]:
    kernels: dict[tuple[str, str], Kernel] = {}
    unknowns: list[ResidualUnknown] = []
    target = "unknown"
    name = ""
    resources: dict[str, int] = {}
    # The sentinel flushes the final function using the same path as every preceding one.
    for number, raw in enumerate([*text.splitlines(), "Function : __end__"], 1):
        line = raw.strip()
        function = _FUNCTION.fullmatch(line)
        next_arch = _arch(line)
        arch_header = line.startswith("arch =")
        if function or arch_header:
            if name:
                facts = {
                    field: Fact[int | None](
                        value=resources.get(key),
                        confidence=Confidence.observed
                        if key in resources
                        else Confidence.uncertain,
                        provenance=Provenance.IMPORTED if key in resources else Provenance.DEFAULT,
                    )
                    for key, field in (("REG", "registers"), ("SHARED", "shared_bytes"))
                }
                kernel = Kernel(name=name, arch=target, **facts)
                key = (name, target)
                if key in kernels and kernels[key] != kernel:
                    unknowns.append(
                        _unknown(
                            f"{target}/{name}", "Conflicting resource records for this kernel"
                        )
                    )
                else:
                    kernels[key] = kernel
                if target == "unknown" or any(key not in resources for key in ("REG", "SHARED")):
                    unknowns.append(
                        _unknown(
                            f"{target}/{name}", "Architecture or resource counts were not reported"
                        )
                    )
            resources = {}
            name = function[1] if function else ""
            if arch_header:
                target = next_arch
                if target == "unknown":
                    unknowns.append(_unknown(f"line:{number}", raw, "Which CUDA target is this?"))
            continue
        if _RESOURCE_TOKEN.search(line):
            if name:
                resources.update({key: int(value) for key, value in _RESOURCE.findall(line)})
            if _RESOURCE_TOKEN.sub("", line).strip():
                unknowns.append(_unknown(f"{target}/{name}/line:{number}", raw))
        elif not _metadata(line):
            unknowns.append(_unknown(f"line:{number}", raw))
    if not kernels:
        unknowns.append(_unknown("resource", "No kernel resources reported"))
    return Parsed(
        result=KernelInventory(
            kernels=tuple(kernels[key] for key in sorted(kernels)),
            arches=tuple(
                sorted({kernel.arch for kernel in kernels.values() if kernel.arch != "unknown"})
            ),
        ),
        coverage=_coverage(text, unknowns, truncated),
        unresolved=tuple(unknowns),
    )
