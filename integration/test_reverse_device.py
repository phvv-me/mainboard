from pathlib import Path

import pytest

from mainboard.core.errors import MissionError
from mainboard.core.tool import ToolOutput
from mainboard.probe.reverse.device import CuobjdumpProvider, NvdisasmProvider
from mainboard.probe.reverse.models import ArchCoverage, Coverage, KernelInventory, SassResult
from mainboard.probe.reverse.parsers import (
    normalize_arch,
    parse_list_elf,
    parse_list_ptx,
    parse_resources,
    parse_sass,
)
from mainboard.probe.reverse.providers import Request, ReverseProvider

from .reverse_fixtures import (
    ELF,
    NVDISASM,
    PTX,
    RESOURCE,
    SASS,
    SASS_REAL,
    FixtureRunner,
    artifact,
)


def test_listed_images_are_architecture_coverage_not_kernel_names() -> None:
    found = parse_list_elf(ELF)
    assert found.coverage == Coverage.complete
    assert found.result.arches == ("sm_100", "sm_90a")
    assert [entry.name for entry in found.result.cubins] == [
        "lib.1.sm_90a.cubin",
        "lib.2.sm_100.cubin",
    ]
    assert parse_list_ptx(PTX).result.ptx_arches == ("sm_90",)
    unknown = parse_list_elf("ELF file 1: unnamed.cubin\nnot a supported record\n")
    assert unknown.coverage == Coverage.partial
    assert len(unknown.unresolved) == 2
    assert unknown.result.cubins[0].arch == "unknown"


@pytest.mark.parametrize("target", ["sm_75", "sm_90a", "sm_100f", "sm_120"])
def test_cuda_suffixes_survive_dispatch_arch_naming(target: str) -> None:
    assert normalize_arch(target) == target


def test_resources_keep_observed_zero_distinct_from_unknown() -> None:
    found = parse_resources(RESOURCE)
    assert found.coverage == Coverage.complete
    assert {kernel.name for kernel in found.result.kernels} == {"_Z6kernelPi", "_Z5otherPi"}
    assert found.result.kernels[0].shared_bytes.value == 0
    missing = parse_resources("arch = sm_90\nFunction missing:\nREG:16\n")
    assert missing.result.kernels[0].shared_bytes.value is None
    assert missing.coverage == Coverage.partial and missing.unresolved


def test_real_fatbin_preamble_lines_are_metadata_not_unknowns() -> None:
    parsed = parse_sass(SASS_REAL)
    assert parsed.coverage is Coverage.complete
    assert parsed.unresolved == ()
    assert [listing.kernel for listing in parsed.result.listings] == ["_Z6kernelPi"]
    only = parsed.result.listings[0]
    assert only.arch == "sm_90a"
    assert [insn.opcode for insn in only.instructions] == ["MOV", "EXIT", "BRA"]
    assert parsed.result.branches.unresolved == 0


@pytest.mark.parametrize("text", [SASS, NVDISASM])
def test_sass_retains_indirect_sites_and_instruction_fields(text: str) -> None:
    found = parse_sass(text)
    assert len(found.result.listings) == 1
    listing = found.result.listings[0]
    assert listing.kernel == "_Z6kernelPi"
    assert len(listing.instructions) == 3
    assert listing.instructions[1].opcode == "BRX"
    assert listing.instructions[1].predicate in {"@P0", "@!P0"}
    assert listing.branches.unresolved == 1 and listing.branches.resolved == 0
    assert listing.brx_sites == (16,)
    assert found.coverage == Coverage.partial
    assert len(found.unresolved) == 1
    assert found.unresolved[0].site.endswith("/0x10")


@pytest.mark.parametrize("tail", ["/*0030*/ MOV R", "[output truncated]", "..."])
def test_cut_sass_is_truncated_without_discarding_prior_branches(tail: str) -> None:
    found = parse_sass(SASS + tail)
    assert found.coverage == Coverage.truncated
    assert found.result.branches.unresolved == 1
    assert len(found.unresolved) >= 2


def test_an_unparsed_instruction_is_retained_as_an_unknown() -> None:
    found = parse_sass(SASS + "/*0030*/ ?NEWOP R1;\n")
    assert found.coverage == Coverage.partial
    assert "?NEWOP" in found.unresolved[-1].reason


def test_unknown_architecture_never_inherits_the_preceding_image() -> None:
    found = parse_sass(SASS + "arch = future\nFunction : new\n/*0000*/ EXIT;\n")
    assert found.result.listings[-1].arch == "unknown"
    assert found.coverage == Coverage.partial
    resources = parse_resources(RESOURCE + "arch = future\nFunction new:\nREG:1 SHARED:0\n")
    assert (
        next(kernel for kernel in resources.result.kernels if kernel.name == "new").arch
        == "unknown"
    )


def test_unknown_resource_tokens_are_not_silently_discarded() -> None:
    found = parse_resources(RESOURCE.replace("REG:32", "REG:32 NEW_FIELD:7"))
    assert found.coverage == Coverage.partial
    assert any("NEW_FIELD:7" in unknown.reason for unknown in found.unresolved)
    malformed = parse_resources("arch = sm_90\nFunction broken:\nREG:32oops SHARED:0\n")
    assert malformed.coverage == Coverage.partial
    assert malformed.result.kernels[0].registers.value is None


def test_multiple_functions_and_architectures_keep_separate_address_spaces() -> None:
    found = parse_sass(SASS + "arch = sm_100\nFunction : second\n/*0000*/ EXIT;\n")
    assert [(item.kernel, item.arch) for item in found.result.listings] == [
        ("_Z6kernelPi", "sm_90a"),
        ("second", "sm_100"),
    ]
    assert found.result.listings[1].instructions[0].address == 0


def test_registered_provider_constructs_profile_and_uses_literal_arguments() -> None:
    runner = FixtureRunner()
    provider = CuobjdumpProvider(runner)
    assert CuobjdumpProvider in ReverseProvider.implementations()
    request = Request(artifact=artifact(), operation="sass", arch="sm_90a", function="_Z6kernelPi")
    found = provider.analyze(request)
    assert found.tool_version == "13.4.48"
    assert found.params == {"arch": "sm_90a", "function": "_Z6kernelPi"}
    assert found.analysis_profile.provider == provider.tool
    assert (
        provider.tool,
        ("--dump-sass", "--function", "_Z6kernelPi", artifact().path),
        None,
    ) in runner.calls
    assert found.coverage == Coverage.partial
    assert provider.operations["sass"].read_only
    assert not provider.operations["sass"].executes_foreign_code


@pytest.mark.parametrize("operation", ["inventory", "arch", "resource"])
def test_provider_returns_typed_findings(operation: str) -> None:
    found = CuobjdumpProvider(FixtureRunner()).analyze(
        Request(artifact=artifact(), operation=operation)
    )
    match found.result:
        case KernelInventory():
            assert len(found.result.kernels) == 2
        case ArchCoverage():
            assert found.result.arches == ("sm_100", "sm_90a")
        case _:
            pytest.fail("unexpected result model")
    assert found.coverage == Coverage.complete


def test_failure_and_runner_truncation_never_report_complete() -> None:
    request = Request(artifact=artifact(), operation="sass")
    for output, expected in (
        (ToolOutput(stderr="unsupported binary", returncode=1), Coverage.failed),
        (ToolOutput(stdout=SASS, returncode=124, truncated=True), Coverage.truncated),
    ):
        found = CuobjdumpProvider(FixtureRunner({"--dump-sass": output})).analyze(request)
        assert found.coverage == expected
        assert found.limitations


def test_availability_explains_missing_tool_and_help_contract() -> None:
    for output in (
        ToolOutput(stderr="not found", returncode=127),
        ToolOutput(stdout="wrong flags"),
    ):
        provider = CuobjdumpProvider(FixtureRunner({"--help": output}))
        assert not provider.available().ok
        assert provider.available().reason and provider.available().remediation
        with pytest.raises(MissionError):
            provider.analyze(Request(artifact=artifact(), operation="inventory"))


def test_nvdisasm_extracts_in_a_temporary_directory_and_records_both_versions() -> None:
    runner = FixtureRunner()
    result = NvdisasmProvider(runner).analyze(
        Request(artifact=artifact(), operation="sass", arch="sm_90a")
    )
    assert result.params["extractor_version"] == "13.4.48"
    extraction = [call for call in runner.calls if call[1][0] == "--extract-elf"]
    assert len(extraction) == 1
    directory = extraction[0][2]
    assert directory is not None and not directory.exists()
    match result.result:
        case SassResult():
            assert result.result.branches.unresolved == 1
            assert result.result.listings[0].arch == "sm_90a"
        case _:
            pytest.fail("expected SASS")


def test_extraction_refuses_path_traversal(tmp_path: Path) -> None:
    entry = parse_list_elf("ELF file 1: ../bad.sm_90.cubin").result.cubins[0]
    with pytest.raises(MissionError, match="unsafe"):
        CuobjdumpProvider(FixtureRunner()).extract(
            Request(artifact=artifact(), operation="sass"),
            entry,
            tmp_path,
        )
