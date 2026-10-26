# Export a `Profile` as a Perfetto timeline: the Chrome Trace Event JSON ui.perfetto.dev loads.

import json
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from ..formats.protocols import Json, TraceEvent

# `.result` imports this module, so `Profile` stays type-only here.
if TYPE_CHECKING:
    from . import Profile

_NS_PER_US = 1000.0
_REGIONS, _KERNELS, _MEMCPYS, _ACTIVITIES = 1, 2, 3, 4
_TRACKS = {
    _REGIONS: "regions",
    _KERNELS: "GPU kernels",
    _MEMCPYS: "GPU memcpy",
    _ACTIVITIES: "CUDA API & activity",
}


def _meta(name: str, tid: int, label: str) -> TraceEvent:
    return {"ph": "M", "name": name, "pid": 0, "tid": tid, "args": {"name": label}}


def _span(name: str, tid: int, ts: float, *, dur: float, args: dict[str, Json]) -> TraceEvent:
    """One complete event, `ts` and `dur` in microseconds."""
    return {"ph": "X", "name": name, "pid": 0, "tid": tid, "ts": ts, "dur": dur, "args": args}


def _timed(
    name: str, tid: int, *, start_ns: int, dur_ns: int, args: dict[str, Json] | None = None
) -> TraceEvent:
    """One complete event from nanoseconds already relative to the timeline origin."""
    return _span(name, tid, start_ns / _NS_PER_US, dur=dur_ns / _NS_PER_US, args=args or {})


def _origin_ns(profile: Profile) -> int:
    """Earliest device timestamp, so the exported timeline starts at zero."""
    starts = [w.start_ns for w in profile.windows]
    starts += [k.start_ns for k in profile.kernels] + [m.start_ns for m in profile.memcpys]
    starts += [a.start_ns for a in profile.activities]
    return min(starts, default=0)


def write_trace(profile: Profile, path: str | PathLike[str]) -> None:
    """Write `profile` as a Chrome/Perfetto trace JSON to `path`."""
    origin = _origin_ns(profile)
    events = [
        _meta("process_name", 0, profile.device or "mainboard"),
        *(_meta("thread_name", tid, label) for tid, label in _TRACKS.items()),
        *(
            _timed(w.name, _REGIONS, start_ns=w.start_ns - origin, dur_ns=w.end_ns - w.start_ns)
            for w in profile.windows
        ),
        *(
            _timed(
                k.name,
                _KERNELS,
                start_ns=k.start_ns - origin,
                dur_ns=k.duration_ns,
                args={"grid": k.grid, "block": k.block, "registers": k.registers},
            )
            for k in profile.kernels
        ),
        *(
            _timed(
                m.kind,
                _MEMCPYS,
                start_ns=m.start_ns - origin,
                dur_ns=m.duration_ns,
                args={"bytes": m.bytes_moved, "GB/s": round(m.bandwidth_gbps, 1)},
            )
            for m in profile.memcpys
        ),
        *(
            _timed(
                a.name,
                _ACTIVITIES,
                start_ns=a.start_ns - origin,
                dur_ns=a.duration_ns,
                args={"kind": a.kind, "correlation": a.correlation_id},
            )
            for a in profile.activities
        ),
    ]
    if not profile.windows:  # untraced: lay regions out sequentially by wall time
        clock = 0.0
        for summary in profile.summaries:
            events.append(
                _span(summary.name, _REGIONS, clock, dur=summary.wall_ms * _NS_PER_US, args={})
            )
            clock += summary.wall_ms * _NS_PER_US
    Path(path).write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "ns"}),
        encoding="utf-8",
        newline="\n",
    )
