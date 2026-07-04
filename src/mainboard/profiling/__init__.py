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

For always-on timing in production code rather than a bounded benchmark session, see
:func:`span`: ``with span("extract"):`` or ``@span``, off by default, async-safe, and
collapsed by :class:`Collector` into per-path count/total/mean/p50/p95/max stats.
"""

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
from .collector import Collector, Reservoir, default_collector
from .dispatch import arch_config, current_arch_key
from .health import Diagnosis
from .models import RegionStat, RegionSummary, SpanRecord, SpanStat
from .profiler import Profiler
from .report import Bound, KernelStat, ProfileReport
from .result import Profile, ProfileDiff, RegionDelta
from .spans import Span, disable_spans, enable_spans, span, spans_enabled
from .stages import StageProfile, profile_stages
from .storage import ReadResult, StorageBandwidth, nvme_to_hbm
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
    "Collector",
    "Diagnosis",
    "HotKernel",
    "HotRegion",
    "KernelStat",
    "KernelTrace",
    "MemcpyTrace",
    "Profile",
    "ProfileDiff",
    "ProfileReport",
    "Profiler",
    "ReadResult",
    "RegionDelta",
    "RegionStat",
    "RegionSummary",
    "Reservoir",
    "Span",
    "SpanRecord",
    "SpanStat",
    "StageProfile",
    "StorageBandwidth",
    "Tracer",
    "arch_config",
    "benchmark",
    "callbacks",
    "compare",
    "current_arch_key",
    "default_collector",
    "disable_auto",
    "disable_spans",
    "enable_auto",
    "enable_spans",
    "gpu_busy",
    "instrument_source",
    "nvme_to_hbm",
    "profile",
    "profile_fn",
    "profile_stages",
    "region",
    "span",
    "spans_enabled",
    "tracer",
    "wait_for_idle",
]
