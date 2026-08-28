import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mainboard.profile import (
    DeviceEvidence,
    Profile,
    ProfileDiff,
    RegionStat,
    RegionSummary,
    perfetto,
)

from .support import traced_profile


def test_the_profile_verbs_read_their_reports_off_the_evidence_it_holds() -> None:
    """`efficiency`, `timeline`, `stats` and `bottlenecks` are thin verbs over the traces."""
    profile = traced_profile()
    assert {row.name for row in profile.efficiency(sm_count=2).rows} == {"gemm", "relu"}
    assert profile.efficiency(sm_count=2).sm_count == 2
    assert profile.timeline().busy_ns > 0
    assert profile.timeline().activities == 4  # 3 kernels + 1 memcpy
    assert {stat.name for stat in profile.stats()} == {"encode", "decode"}
    assert profile.bottlenecks(top=1)[0].name == "encode"


def test_a_diff_matches_regions_by_name_and_survives_a_round_trip(tmp_path: Path) -> None:
    """Regions are matched by name and ranked by absolute change, and a profile reloads.

    A region present in only one of the two profiles has no speedup to report rather than
    a division by zero, and the host and device labels of both sides travel with the diff.
    """
    base = Profile(host="a", device="cuda:0", summaries=(RegionSummary(name="r", wall_ms=2.0),))
    current = Profile(
        host="b",
        device="cuda:1",
        summaries=(RegionSummary(name="r", wall_ms=1.0), RegionSummary(name="new", wall_ms=5.0)),
    )
    diff = current.diff(base)
    assert isinstance(diff, ProfileDiff)
    assert [row.delta_ms for row in diff.rows] == sorted(
        (row.delta_ms for row in diff.rows), key=abs, reverse=True
    )
    assert next(row for row in diff.rows if row.name == "r").speedup == 2.0  # 2ms -> 1ms
    assert next(row for row in diff.rows if row.name == "new").speedup == 0.0
    assert (diff.baseline_host, diff.current_host) == ("a", "b")
    assert (diff.baseline_device, diff.current_device) == ("cuda:0", "cuda:1")

    path = tmp_path / "p.json"
    current.save(path)
    assert Profile.load(path).stats()[0].name == "new"


@pytest.mark.parametrize(
    ("profile", "present", "absent"),
    [
        (Profile(), ("No profiling data collected.",), ("Spans", "GPU activity")),
        (traced_profile(), ("Spans", "encode", "GPU activity", "3 kernels"), ("Capture limit",)),
        (
            Profile(dropped_activities=3),
            ("Capture limit", "oldest GPU activities dropped"),
            ("oldest spans dropped",),
        ),
        (
            Profile(summaries=(RegionSummary(name="r", wall_ms=1.0),), dropped_spans=2),
            ("Spans", "oldest spans dropped"),
            ("oldest GPU activities dropped",),
        ),
        (
            Profile(
                summaries=(RegionSummary(name="r", wall_ms=1.0),),
                device_evidence=DeviceEvidence.ABSENT,
            ),
            ("Spans", "Device", "no device evidence collected"),
            ("GPU activity",),
        ),
    ],
    ids=[
        "nothing_collected",
        "spans_and_gpu_activity",
        "activities_dropped",
        "spans_dropped",
        "device_evidence_asked_for_and_absent",
    ],
)
def test_the_report_carries_only_the_evidence_sections_it_has(
    profile: Profile, present: Sequence[str], absent: Sequence[str]
) -> None:
    """A section appears only when its evidence does, and an empty run says so plainly."""
    text = profile.report()
    assert str(profile) == text
    assert all(fragment in text for fragment in present)
    assert not any(fragment in text for fragment in absent)
    assert Profile._region_text([]) == "No regions recorded."


# Two region names over a short list is a small space, so a trimmed budget covers it and keeps
# the suite's wall time where it was.
@settings(max_examples=15)
@given(
    regions=st.lists(
        st.tuples(
            st.sampled_from(["encode", "decode"]),
            st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        ),
        max_size=6,
    )
)
def test_per_name_stats_collapse_every_call_of_a_region_into_one_row(
    regions: Sequence[tuple[str, float]],
) -> None:
    """A region called many times is one row with its call count, not one row per call."""
    summaries = [RegionSummary(name=name, wall_ms=wall) for name, wall in regions]
    stats = RegionStat.aggregate(summaries)

    assert {stat.name for stat in stats} == {name for name, _ in regions}
    assert sum(stat.calls for stat in stats) == len(regions)
    assert [stat.total_ms for stat in stats] == sorted(
        (stat.total_ms for stat in stats), reverse=True
    )
    for stat in stats:
        walls = [wall for name, wall in regions if name == stat.name]
        assert stat.calls == len(walls)
        assert stat.total_ms == sum(walls)
        assert stat.avg_ms == sum(walls) / len(walls)


def test_the_perfetto_export_lays_one_track_out_per_activity_class(tmp_path: Path) -> None:
    """`Profile.perfetto` writes loadable Chrome trace JSON with all four tracks."""
    path = tmp_path / "trace.json"
    traced_profile().perfetto(path)
    events = json.loads(path.read_text())["traceEvents"]
    assert {"gemm", "relu", "HtoD", "cudaLaunchKernel", "encode"} <= {e["name"] for e in events}
    assert {1, 2, 3, 4} <= {e["tid"] for e in events}


def test_the_perfetto_export_lays_untraced_regions_out_sequentially(tmp_path: Path) -> None:
    """Without device windows, regions are placed one after another by wall time.

    A profile with nothing in it at all still writes a valid, event-less trace rather than
    failing to find an origin timestamp.
    """
    profile = Profile(
        summaries=(RegionSummary(name="a", wall_ms=1.0), RegionSummary(name="b", wall_ms=2.0))
    )
    path = tmp_path / "t.json"
    perfetto.write_trace(profile, path)
    spans = [e for e in json.loads(path.read_text())["traceEvents"] if e["ph"] == "X"]
    assert [span["name"] for span in spans] == ["a", "b"]
    assert spans[1]["ts"] > spans[0]["ts"]  # b starts after a

    empty = tmp_path / "e.json"
    perfetto.write_trace(Profile(), empty)
    assert json.loads(empty.read_text())["traceEvents"]
