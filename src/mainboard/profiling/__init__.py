"""Cross-platform memory/time profiling for mainboard.

A simple, vendor-agnostic API: annotate regions and a :class:`Profiler` samples the
device while they run. Annotation goes to the native timeline via the matching
:class:`Tracer` (NVTX / ROCTx / `os_signpost`), so the same code is inspectable under
Nsight, `rocprofv3`, or Instruments — and free when nothing is attached.

    from mainboard.profiling import Profiler, region, profile

    with Profiler() as p:
        with region("attention"):
            ...
        print(p.report())

Auto-annotation: ``Profiler.auto(["my.package"])`` (runtime, PEP 669) or
:func:`instrument_source` (static AST rewrite). Importing this package registers the
vendor tracers.
"""

from __future__ import annotations

from .annotate import (
    callbacks,
    disable_auto,
    enable_auto,
    instrument_source,
    profile,
    region,
    tracer,
)
from .benchmark import BenchSample, benchmark, compare
from .bottleneck import gpu_busy, wait_for_idle
from .bottleneck import profile as profile_fn
from .dispatch import arch_config, current_arch_key
from .models import RegionStat, RegionSummary
from .profiler import Profiler
from .report import Bound, KernelStat, ProfileReport
from .result import Profile, ProfileDiff, RegionDelta
from .stages import StageProfile, profile_stages
from .trace import (
    Activity,
    ActivityRecord,
    BottleneckReport,
    CallbackSession,
    HotKernel,
    HotRegion,
    KernelTrace,
    MemcpyTrace,
)
from .tracer import Tracer

__all__ = [
    "Activity",
    "ActivityRecord",
    "BenchSample",
    "BottleneckReport",
    "Bound",
    "CallbackSession",
    "HotKernel",
    "HotRegion",
    "KernelStat",
    "KernelTrace",
    "MemcpyTrace",
    "Profile",
    "ProfileDiff",
    "ProfileReport",
    "Profiler",
    "RegionDelta",
    "RegionStat",
    "RegionSummary",
    "StageProfile",
    "Tracer",
    "arch_config",
    "benchmark",
    "callbacks",
    "compare",
    "current_arch_key",
    "disable_auto",
    "enable_auto",
    "gpu_busy",
    "instrument_source",
    "profile",
    "profile_fn",
    "profile_stages",
    "region",
    "tracer",
    "wait_for_idle",
]
