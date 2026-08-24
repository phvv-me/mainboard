# Grid-shape parsing, wave math, and the rendered launch-efficiency report.

import math

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.profile import EfficiencyReport, KernelEfficiency
from mainboard.profile.efficiency import grid_blocks, readable

from ..strategies import WORDS
from .conftest import kernel, render

_ONE_ROW = (
    KernelEfficiency(
        name="k",
        calls=1,
        total_ms=1000.0,
        blocks=1,
        block_threads=1,
        registers=0,
        waves=1.0,
        tail_waste_pct=0.0,
    ),
)


# The pinned examples below carry the branches, so the random budget only needs to add breadth
# and is trimmed from the shared default.
@settings(max_examples=10)
@given(dims=st.lists(st.integers(min_value=1, max_value=999), min_size=1, max_size=3))
@example(dims=[110, 1, 1])  # the CUPTI spelling
def test_grid_blocks_multiplies_out_every_extent_it_can_read(dims: list[int]) -> None:
    """A launch shape is the product of its dimensions, however the backend spells it.

    Parenthesised and comma-separated spellings are accepted too, so a backend that
    formats differently does not silently report zero, and a shape with no digits at all
    reads as zero rather than as the empty product.
    """
    total = math.prod(dims)
    assert grid_blocks("x".join(str(dim) for dim in dims)) == total
    assert grid_blocks(f"({', '.join(str(dim) for dim in dims)})") == total
    assert grid_blocks("") == 0
    assert grid_blocks("garbage") == 0


# The pinned examples below carry the branches, so the random budget only needs to add breadth
# and is trimmed from the shared default.
@settings(max_examples=10)
@given(name=WORDS)
def test_readable_trims_mangled_names_and_passes_plain_ones_through(name: str) -> None:
    """A plain symbol survives untouched, and an Itanium-mangled one keeps its first three parts.

    numba appends a long content hash to every symbol, so the tail of a mangled name is
    noise, while a name that mangles into nothing at all is returned as it came.
    """
    assert readable(name) == name
    # Itanium mangling writes each part as `<length><identifier>`, so `5numba` is `numba`.
    assert readable("_ZN5numba5tests3jitE") == "numba.tests.jit"
    assert readable("_ZN") == "_ZN"


# The pinned examples below carry the branches, so the random budget only needs to add breadth
# and is trimmed from the shared default.
@settings(max_examples=15)
@given(
    launches=st.lists(
        st.tuples(
            st.sampled_from(["gemm", "relu"]),
            st.integers(min_value=0, max_value=5_000),
            st.integers(min_value=1, max_value=8),
        ),
        max_size=6,
    ),
    sm_count=st.integers(min_value=1, max_value=4),
)
@example(launches=[("gemm", 100, 3), ("gemm", 100, 3), ("relu", 500, 4)], sm_count=2)
@example(launches=[("stalled", 0, 1)], sm_count=2)  # a kernel that never ran is not a row
def test_kernels_rank_by_total_time_and_carry_their_wave_quantisation(
    launches: list[tuple[str, int, int]], sm_count: int
) -> None:
    """Rows rank slowest first, and each one's waves and tail follow from its grid.

    A whole number of waves leaves no tail at all, and a fractional one wastes a share of
    the run strictly under a full wave. Kernels with no measured duration are dropped
    rather than ranked, so a launch that never ran cannot lead the table.
    """
    kernels = [
        kernel(name, ns, grid=f"{blocks}x1x1", block="256x1x1") for name, ns, blocks in launches
    ]
    rows = KernelEfficiency.aggregate(kernels, sm_count=sm_count)
    ran = [launch for launch in launches if launch[1] > 0]

    assert [row.total_ms for row in rows] == sorted((row.total_ms for row in rows), reverse=True)
    assert sum(row.calls for row in rows) == len(ran)
    assert {row.name for row in rows} == {name for name, _, _ in ran}
    for row in rows:
        assert row.waves == row.blocks / sm_count
        assert row.block_threads == 256
        assert 0.0 <= row.tail_waste_pct < 100.0
        assert (row.tail_waste_pct == 0.0) is float(row.waves).is_integer()


def test_the_report_keeps_only_the_hottest_rows() -> None:
    """`build` ranks by total device time and truncates to `top`."""
    kernels = [kernel(name, ns, grid="1x1x1") for name, ns in (("a", 100), ("b", 300), ("c", 200))]
    report = EfficiencyReport.build(kernels, sm_count=1, top=2)
    assert [row.name for row in report.rows] == ["b", "c"]
    assert report.sm_count == 1


@pytest.mark.parametrize(
    ("report", "achieved_gbs", "utilisation_pct"),
    [
        (EfficiencyReport(bytes_moved=10**9), 0.0, 0.0),
        (EfficiencyReport(rows=_ONE_ROW, bytes_moved=0, peak_bandwidth_gbs=2.0), 0.0, 0.0),
        (EfficiencyReport(rows=_ONE_ROW, bytes_moved=10**9, peak_bandwidth_gbs=0.0), 1.0, 0.0),
        (EfficiencyReport(rows=_ONE_ROW, bytes_moved=10**9, peak_bandwidth_gbs=2.0), 1.0, 50.0),
    ],
    ids=["no_device_time", "no_payload", "no_known_peak", "a_second_of_a_gigabyte"],
)
def test_achieved_bandwidth_is_the_payload_over_the_device_time(
    report: EfficiencyReport, achieved_gbs: float, utilisation_pct: float
) -> None:
    """Bandwidth needs both a payload and some device time, and a score needs a known peak."""
    assert report.achieved_gbs == achieved_gbs
    assert report.bandwidth_utilisation_pct == utilisation_pct


@pytest.mark.parametrize("bytes_moved", [10**6, 0], ids=["with_a_payload", "without_a_payload"])
def test_the_rendered_report_summarises_bandwidth_only_when_a_payload_is_known(
    bytes_moved: int,
) -> None:
    """The table always lists the kernels, and the title scores bandwidth when it can."""
    kernels = [kernel("relu_kernel", 100, grid="2x1x1", block="128x1x1")]
    report = EfficiencyReport.build(
        kernels, sm_count=1, peak_bandwidth_gbs=2.0, bytes_moved=bytes_moved
    )
    text = render(report)
    assert "relu_kernel" in text
    assert ("GB/s achieved" in text) is bool(bytes_moved)
