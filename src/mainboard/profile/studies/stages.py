from collections.abc import Callable, Mapping

from patos import FrozenModel

from ...log import logger
from ...probe.units.gpu import GPU
from ..benchmark import BenchSample, benchmark
from ..profiler import Profiler
from ..result import Profile
from ..spans import span
from ..trace import Activity

_US_PER_MS = 1e3


class StageProfile(FrozenModel):
    """Result of `profile_stages`: wall-clock per stage and an optional deep trace.

    samples: one per stage, in the order the cases were given.
    profile: the single trace pass, None when tracing was off.
    """

    samples: tuple[BenchSample, ...] = ()
    profile: Profile | None = None

    def __str__(self) -> str:
        if self.profile is None:
            return self.timing_text()
        return f"{self.timing_text()}\n\n{self.profile.report()}\n\n{self.profile.trace_report()}"

    def show(self) -> None:
        logger.result(str(self))

    def timing_text(self) -> str:
        """The plain-text per-stage mean/min table, without the deep trace."""
        if not self.samples:
            return "No stages profiled."
        header = f"{'stage':<24}{'mean':>13}{'min':>16}"
        rows = [
            f"{s.label:<24}{s.mean_us / _US_PER_MS:>10.3f} ms"
            f"{s.min_us / _US_PER_MS:>10.3f} ms (min)"
            for s in self.samples
        ]
        return "\n".join([header, *rows])


def profile_stages[T, S](
    cases: Mapping[str, Callable[[], T]],
    *,
    gpu: GPU | None = None,
    sync: Callable[[], S] | None = None,
    trace: bool | Activity = False,
    iters: int = 5,
    warmup: int = 1,
) -> StageProfile:
    """Benchmark each named zero-arg stage, then optionally trace them in one separate pass.

    gpu: the device to trace on, None for the profiler's host discovery.
    sync: device barrier after each run (e.g. `torch.cuda.synchronize`), so async GPU work is
        timed; in the trace pass it drains each stage before its span closes.
    trace: True for every activity kind the device offers, or exactly the given `Activity`
        kinds. A missing device or collector raises rather than returning no trace.
    """
    samples = tuple(
        benchmark(fn, label=name, iters=iters, warmup=warmup, sync=sync)
        for name, fn in cases.items()
    )
    profile = _trace_stages(cases, trace, sync, gpu) if trace else None
    return StageProfile(samples=samples, profile=profile)


def _trace_stages[T, S](
    cases: Mapping[str, Callable[[], T]],
    trace: bool | Activity,
    sync: Callable[[], S] | None,
    gpu: GPU | None,
) -> Profile:
    with Profiler(
        gpus=(gpu,) if gpu is not None else (),
        features=Profiler.Feature.SPANS | Profiler.Feature.MARKERS | Profiler.Feature.ACTIVITY,
        activities=trace if isinstance(trace, Activity) else Activity.ALL,
    ) as profiler:
        for name, fn in cases.items():
            _trace_stage(name, fn, sync)
    return profiler.result()


def _trace_stage[T, S](name: str, fn: Callable[[], T], sync: Callable[[], S] | None) -> None:
    with span(name):
        fn()
        if sync is not None:
            sync()
