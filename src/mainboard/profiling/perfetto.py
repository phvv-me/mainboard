"""Export a :class:`~mainboard.profiling.result.Profile` as a Perfetto timeline.

Emits the Chrome Trace Event JSON that ui.perfetto.dev loads natively: regions on one
track (nested duration events auto-stack), GPU kernels and memcpys on their own tracks.
When the profile was traced, region windows and kernel records share the device clock, so
they line up on the same timeline; otherwise regions are laid out sequentially by wall
time. This is the cross-vendor view — nsys, rocprofv3, and xctrace all speak Perfetto too.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .protocols import Json, TraceEvent

if TYPE_CHECKING:
    from os import PathLike

    from .result import Profile

_NS_PER_US = 1000.0
_REGIONS, _KERNELS, _MEMCPYS, _ACTIVITIES = 1, 2, 3, 4


def _meta(name: str, tid: int, label: str) -> TraceEvent:
    return {"ph": "M", "name": name, "pid": 0, "tid": tid, "args": {"name": label}}


def _origin_ns(profile: Profile) -> int:
    """Earliest device timestamp, so the exported timeline starts at zero."""
    starts = [w.start_ns for w in profile.windows]
    starts += [k.start_ns for k in profile.kernels] + [m.start_ns for m in profile.memcpys]
    starts += [a.start_ns for a in profile.activities]
    return min(starts) if starts else 0


def _span(
    name: str, tid: int, start_ns: int, dur_ns: int, origin: int, args: dict[str, Json]
) -> TraceEvent:
    return {
        "ph": "X",
        "name": name,
        "pid": 0,
        "tid": tid,
        "ts": (start_ns - origin) / _NS_PER_US,
        "dur": dur_ns / _NS_PER_US,
        "args": args,
    }


def write_trace(profile: Profile, path: str | PathLike[str]) -> None:
    """Write ``profile`` as a Chrome/Perfetto trace JSON to ``path``."""
    events: list[TraceEvent] = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": 0,
            "tid": 0,
            "args": {"name": profile.device or "mainboard"},
        },
        _meta("thread_name", _REGIONS, "regions"),
        _meta("thread_name", _KERNELS, "GPU kernels"),
        _meta("thread_name", _MEMCPYS, "GPU memcpy"),
        _meta("thread_name", _ACTIVITIES, "CUDA API & activity"),
    ]
    origin = _origin_ns(profile)
    for window in profile.windows:
        events.append(
            _span(
                window.name, _REGIONS, window.start_ns, window.end_ns - window.start_ns, origin, {}
            )
        )
    for kernel in profile.kernels:
        events.append(
            _span(
                kernel.name,
                _KERNELS,
                kernel.start_ns,
                kernel.duration_ns,
                origin,
                {"grid": kernel.grid, "block": kernel.block, "registers": kernel.registers},
            )
        )
    for memcpy in profile.memcpys:
        events.append(
            _span(
                memcpy.kind,
                _MEMCPYS,
                memcpy.start_ns,
                memcpy.duration_ns,
                origin,
                {"bytes": memcpy.bytes_moved, "GB/s": round(memcpy.bandwidth_gbps, 1)},
            )
        )
    for activity in profile.activities:
        events.append(
            _span(
                activity.name,
                _ACTIVITIES,
                activity.start_ns,
                activity.duration_ns,
                origin,
                {"kind": activity.kind, "correlation": activity.correlation_id},
            )
        )
    if not profile.windows:  # untraced: lay regions out sequentially by wall time
        clock = 0.0
        for summary in profile.summaries:
            events.append(
                {
                    "ph": "X",
                    "name": summary.name,
                    "pid": 0,
                    "tid": _REGIONS,
                    "ts": clock,
                    "dur": summary.wall_ms * _NS_PER_US,
                    "args": {},
                }
            )
            clock += summary.wall_ms * _NS_PER_US
    Path(path).write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}))
