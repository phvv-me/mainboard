"""Grid-shape parsing, wave math, and the rendered launch-efficiency report."""

from rich.console import Console

from mainboard.profiling.efficiency import (
    EfficiencyReport,
    KernelEfficiency,
    grid_blocks,
    readable,
)
from mainboard.profiling.trace import KernelTrace


def _kernel(name: str, ns: int, **shape: object) -> KernelTrace:
    """A `KernelTrace` of `name` lasting `ns` nanoseconds from t=0."""
    return KernelTrace(name=name, start_ns=0, end_ns=ns, **shape)  # pyrefly: ignore


def render(report: EfficiencyReport) -> str:
    """Render a report to plain text for content assertions."""
    console = Console(no_color=True, width=120, record=True)
    console.print(report)
    return console.export_text()


def test_grid_blocks_parses_cupti_and_alternate_spellings() -> None:
    assert grid_blocks("110x1x1") == 110
    assert grid_blocks("(8, 2, 1)") == 16
    assert grid_blocks("") == 0
    assert grid_blocks("garbage") == 0


def test_readable_trims_mangled_names_and_passes_plain_ones_through() -> None:
    assert readable("relu_kernel") == "relu_kernel"
    # Itanium mangling: each `<length><identifier>` segment, so "5numba" is "numba".
    assert readable("_ZN5numba5tests3jitE") == "numba.tests.jit"
    assert readable("_ZN") == "_ZN"


def test_aggregate_discards_zero_duration_kernels_and_ranks_by_total_time() -> None:
    kernels = [
        _kernel("gemm", 100, grid="3x1x1", block="256x1x1", registers=32),
        _kernel("gemm", 100, grid="3x1x1", block="256x1x1", registers=32),
        _kernel("relu", 500, grid="4x1x1", block="128x1x1", registers=16),
        KernelTrace(name="stalled", start_ns=10, end_ns=10),
    ]
    rows = KernelEfficiency.aggregate(kernels, sm_count=2)
    assert [row.name for row in rows] == ["relu", "gemm"]

    relu, gemm = rows
    assert relu.calls == 1
    assert relu.waves == 2.0
    assert relu.tail_waste_pct == 0.0

    assert gemm.calls == 2
    assert gemm.block_threads == 256
    assert gemm.registers == 32
    assert gemm.waves == 1.5
    assert gemm.tail_waste_pct == 25.0


def test_report_build_ranks_and_truncates_to_top() -> None:
    kernels = [
        _kernel("a", 100, grid="1x1x1"),
        _kernel("b", 300, grid="1x1x1"),
        _kernel("c", 200, grid="1x1x1"),
    ]
    report = EfficiencyReport.build(kernels, sm_count=1, top=2)
    assert [row.name for row in report.rows] == ["b", "c"]
    assert report.sm_count == 1


def test_achieved_bandwidth_is_zero_without_a_payload_or_any_device_time() -> None:
    empty = EfficiencyReport(bytes_moved=10**9)
    assert empty.achieved_gbs == 0.0

    row = KernelEfficiency(
        name="k",
        calls=1,
        total_ms=1000.0,
        blocks=1,
        block_threads=1,
        registers=0,
        waves=1.0,
        tail_waste_pct=0.0,
    )
    no_payload = EfficiencyReport(rows=(row,), bytes_moved=0)
    assert no_payload.achieved_gbs == 0.0


def test_achieved_bandwidth_and_utilisation_from_a_known_payload() -> None:
    row = KernelEfficiency(
        name="k",
        calls=1,
        total_ms=1000.0,
        blocks=1,
        block_threads=1,
        registers=0,
        waves=1.0,
        tail_waste_pct=0.0,
    )
    report = EfficiencyReport(rows=(row,), bytes_moved=10**9, peak_bandwidth_gbs=2.0)
    assert report.total_ms == 1000.0
    assert report.achieved_gbs == 1.0
    assert report.bandwidth_utilisation_pct == 50.0


def test_utilisation_is_zero_without_a_known_device_peak() -> None:
    row = KernelEfficiency(
        name="k",
        calls=1,
        total_ms=1000.0,
        blocks=1,
        block_threads=1,
        registers=0,
        waves=1.0,
        tail_waste_pct=0.0,
    )
    report = EfficiencyReport(rows=(row,), bytes_moved=10**9, peak_bandwidth_gbs=0.0)
    assert report.bandwidth_utilisation_pct == 0.0


def test_rich_render_includes_the_bandwidth_summary_when_a_payload_is_known() -> None:
    kernels = [_kernel("relu_kernel", 100, grid="2x1x1", block="128x1x1")]
    report = EfficiencyReport.build(kernels, sm_count=1, peak_bandwidth_gbs=2.0, bytes_moved=10**6)
    text = render(report)
    assert "relu_kernel" in text
    assert "GB/s achieved" in text


def test_rich_render_omits_the_bandwidth_summary_without_a_payload() -> None:
    kernels = [_kernel("relu_kernel", 100, grid="2x1x1", block="128x1x1")]
    report = EfficiencyReport.build(kernels, sm_count=1)
    text = render(report)
    assert "relu_kernel" in text
    assert "GB/s achieved" not in text
