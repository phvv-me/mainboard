# One-call staged profiling: benchmark a set of named steps, optionally trace them.

from collections.abc import Callable, Mapping

from patos import FrozenModel

from .benchmark import BenchSample, benchmark
from .profiler import Profiler
from .protocols import DeviceProbe
from .result import Profile
from .spans import span
from .trace import Activity


class StageProfile(FrozenModel):
    """Result of :func:`profile_stages`: wall-clock per stage and an optional deep trace.

    samples: one :class:`BenchSample` per stage, in the order the cases were given.
    profile: the CUPTI :class:`Profile` from the single trace pass, or ``None`` when
        tracing was off or no GPU was present.
    """

    samples: tuple[BenchSample, ...] = ()
    profile: Profile | None = None

    def __str__(self) -> str:
        if self.profile is None:
            return self.timing_text()
        return f"{self.timing_text()}\n\n{self.profile.report()}\n\n{self.profile.trace_report()}"

    def show(self) -> None:
        """Print the per-stage timing table, and the deep report when traced."""
        print(str(self))

    def timing_text(self) -> str:
        """The plain-text per-stage mean/min table (no deep trace)."""
        if not self.samples:
            return "No stages profiled."
        header = f"{'stage':<24}{'mean':>13}{'min':>16}"
        rows = [
            f"{s.label:<24}{s.mean_us / 1e3:>10.3f} ms{s.min_us / 1e3:>10.3f} ms (min)"
            for s in self.samples
        ]
        return "\n".join([header, *rows])


def profile_stages[T, S](
    cases: Mapping[str, Callable[[], T]],
    *,
    gpu: DeviceProbe | None = None,
    sync: Callable[[], S] | None = None,
    trace: bool | Activity = False,
    iters: int = 5,
    warmup: int = 1,
) -> StageProfile:
    """Benchmark each named stage and, when ``trace`` and ``gpu`` are given, trace them.

    cases: ordered map of stage name to a zero-arg callable (bind args with a lambda).
    gpu: the device to trace on, or ``None`` to skip the deep trace on a CPU-only host
        (`mainboard.probe` is not a dependency of profiling, so the caller resolves it).
    sync: device barrier called after each run so async GPU work is timed (e.g.
        ``torch.cuda.synchronize``); also drains each region in the trace pass.
    trace: open one CUPTI activity pass over the stages when truthy *and* ``gpu`` is
        given — ``True`` for the default kinds, or an :class:`Activity` flag for exactly
        those. Skipped silently on a CPU-only host, so the same call works everywhere.
    iters/warmup: timed and untimed runs per stage for the wall-clock pass.
    """
    samples = tuple(
        benchmark(fn, label=name, iters=iters, warmup=warmup, sync=sync)
        for name, fn in cases.items()
    )
    profile = _trace_stages(cases, trace, sync, gpu) if trace and gpu is not None else None
    return StageProfile(samples=samples, profile=profile)


def _trace_stages[T, S](
    cases: Mapping[str, Callable[[], T]],
    trace: bool | Activity,
    sync: Callable[[], S] | None,
    gpu: DeviceProbe,
) -> Profile:
    """Run one CUPTI pass, each stage bracketed by a `span` and a device ``sync``."""
    kinds = trace if isinstance(trace, Activity) else Activity.ALL
    with Profiler(
        gpus=(gpu,),
        features=Profiler.Feature.SPANS | Profiler.Feature.MARKERS | Profiler.Feature.ACTIVITY,
        activities=kinds,
    ) as profiler:
        for name, fn in cases.items():
            with span(name):
                fn()
            if sync is not None:
                sync()
    return profiler.result()
