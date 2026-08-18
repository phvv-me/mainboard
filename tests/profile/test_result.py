import json
from pathlib import Path

from mainboard.profile import (
    ActivityRecord,
    KernelTrace,
    MemcpyTrace,
    Profile,
    ProfileDiff,
    RegionStat,
    RegionSummary,
    RegionWindow,
    perfetto,
)
from mainboard.profile.result import _region_text


def _traced_profile() -> Profile:
    """A profile with two regions and kernels/memcpys binned across their windows."""
    return Profile(
        device="dev",
        summaries=(
            RegionSummary(name="encode", wall_ms=2.0, avg_util_pct=50.0, avg_power_w=100.0),
            RegionSummary(name="decode", wall_ms=1.0),
        ),
        windows=(
            RegionWindow(name="encode", start_ns=0, end_ns=1000, wall_ns=2_000_000),
            RegionWindow(name="decode", start_ns=1000, end_ns=2000, wall_ns=1_000_000),
        ),
        kernels=(
            KernelTrace(name="gemm", start_ns=0, end_ns=600, grid="8x1x1", block="256x1x1"),
            KernelTrace(name="gemm", start_ns=1000, end_ns=1400),
            KernelTrace(name="relu", start_ns=600, end_ns=700),
        ),
        memcpys=(MemcpyTrace(kind="HtoD", start_ns=0, end_ns=100, bytes_moved=4096),),
        activities=(
            ActivityRecord(kind="runtime", name="cudaLaunchKernel", start_ns=0, end_ns=5),
        ),
    )


def test_profile_efficiency_reads_launch_shape_off_the_kernel_traces() -> None:
    """`Profile.efficiency` is a thin verb over `EfficiencyReport.build`."""
    report = _traced_profile().efficiency(sm_count=2)
    assert {row.name for row in report.rows} == {"gemm", "relu"}
    assert report.sm_count == 2


def test_profile_timeline_reads_busy_and_idle_off_the_kernel_traces() -> None:
    """`Profile.timeline` is a thin verb over `DeviceTimeline.from_traces`."""
    timeline = _traced_profile().timeline()
    assert timeline.busy_ns > 0
    assert timeline.activities == 4  # 3 kernels + 1 memcpy


def test_profile_stats_and_bottlenecks() -> None:
    """Stats collapse per name; bottlenecks are the slowest by total wall time."""
    profile = _traced_profile()
    stats = profile.stats()
    assert {s.name for s in stats} == {"encode", "decode"}
    assert profile.bottlenecks(top=1)[0].name == "encode"


def test_profile_diff_ranks_regressions(tmp_path: Path) -> None:
    """A diff matches regions by name and ranks by absolute change; round-trips on disk."""
    base = Profile(summaries=(RegionSummary(name="r", wall_ms=2.0),))
    cur = Profile(
        summaries=(RegionSummary(name="r", wall_ms=1.0), RegionSummary(name="new", wall_ms=5.0))
    )
    diff = cur.diff(base)
    assert isinstance(diff, ProfileDiff)
    speedup = next(row for row in diff.rows if row.name == "r").speedup
    assert speedup == 2.0  # 2ms -> 1ms
    path = tmp_path / "p.json"
    cur.save(path)
    assert Profile.load(path).stats()[0].name == "new"


def test_profile_diff_carries_host_and_device_labels() -> None:
    base = Profile(host="a", device="cuda:0")
    cur = Profile(host="b", device="cuda:1")
    diff = cur.diff(base)
    assert diff.baseline_host == "a"
    assert diff.current_host == "b"
    assert diff.baseline_device == "cuda:0"
    assert diff.current_device == "cuda:1"


def test_profile_report_and_str_are_plain_text() -> None:
    """`report`/`__str__` give a per-region table; an empty profile says so."""
    assert "encode" in _traced_profile().report()
    assert str(_traced_profile()) == _traced_profile().report()
    assert "No profiling data" in Profile().report()


def test_profile_perfetto_export(tmp_path: Path) -> None:
    """`Profile.perfetto` writes loadable Chrome trace JSON with all four tracks."""
    path = tmp_path / "trace.json"
    _traced_profile().perfetto(path)
    data = json.loads(path.read_text())
    names = {e["name"] for e in data["traceEvents"]}
    assert {"gemm", "relu", "HtoD", "cudaLaunchKernel"} <= names


def test_perfetto_lays_untraced_regions_sequentially(tmp_path: Path) -> None:
    """Without device windows, regions are placed sequentially by wall time."""
    profile = Profile(
        summaries=(RegionSummary(name="a", wall_ms=1.0), RegionSummary(name="b", wall_ms=2.0))
    )
    path = tmp_path / "t.json"
    perfetto.write_trace(profile, path)
    spans = [e for e in json.loads(path.read_text())["traceEvents"] if e["ph"] == "X"]
    assert [s["name"] for s in spans] == ["a", "b"]
    assert spans[1]["ts"] > spans[0]["ts"]  # b starts after a


def test_perfetto_origin_is_zero_with_no_events(tmp_path: Path) -> None:
    """An empty profile still writes a valid (event-less span) trace."""
    path = tmp_path / "e.json"
    perfetto.write_trace(Profile(), path)
    assert "traceEvents" in json.loads(path.read_text())


def test_region_stat_aggregate_collapses_calls() -> None:
    """Repeated occurrences of a name collapse into one stat with the call count."""
    rows = (RegionSummary(name="r", wall_ms=1.0), RegionSummary(name="r", wall_ms=3.0))
    stat = RegionStat.aggregate(rows)[0]
    assert stat.calls == 2
    assert stat.total_ms == 4.0
    assert stat.avg_ms == 2.0


def test_report_lists_gpu_activity_section_only_when_kernels_present() -> None:
    """The plain-text report grows a `GPU activity` section only when kernels were traced."""
    assert "GPU activity" not in Profile().report()
    assert "GPU activity" in _traced_profile().report()


def test_report_lists_dropped_activities_even_without_dropped_spans() -> None:
    """The capture-limit note covers dropped activities independently of dropped spans."""
    report = Profile(dropped_activities=3).report()
    assert "oldest GPU activities dropped" in report
    assert "oldest spans dropped" not in report


def test_region_text_reports_emptiness_for_no_stats() -> None:
    """The plain-text region table says so rather than emitting a blank header."""
    assert _region_text([]) == "No regions recorded."
