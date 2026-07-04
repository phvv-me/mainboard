"""One-call staged profiling: benchmark a set of named steps, optionally trace them.

The "profile an experiment" pattern is always the same: build a dict of named callables
(pool build, encode, decode, a transform), wall-clock each with a device ``sync``, and on
a GPU host also run *one* CUPTI activity pass that wraps each step in a `region` for the
kernel-level breakdown. :func:`profile_stages` collapses that boilerplate into a single
call returning a :class:`StageProfile` — the per-stage :class:`BenchSample` list plus, when
traced, the deep :class:`Profile`. It reuses :func:`benchmark`, :class:`Profiler`, and
`region`; it does not reimplement timing or tracing.
"""

from typing import TYPE_CHECKING

from ..gpu import GPU
from ..models.base import FrozenModel
from .annotate import region
from .benchmark import BenchSample, benchmark
from .profiler import Profiler
from .result import Profile
from .trace import Activity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class StageProfile(FrozenModel):
    """Result of :func:`profile_stages`: wall-clock per stage and an optional deep trace.

    samples: one :class:`BenchSample` per stage, in the order the cases were given.
    profile: the CUPTI :class:`Profile` from the single trace pass, or ``None`` when
        tracing was off or no GPU was present.
    """

    samples: tuple[BenchSample, ...] = ()
    profile: Profile | None = None

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

    def __str__(self) -> str:
        if self.profile is None:
            return self.timing_text()
        return f"{self.timing_text()}\n\n{self.profile.report()}\n\n{self.profile.trace_report()}"

    def show(self, *, color: bool = True) -> None:
        """Print the per-stage timing table, and the rich deep report when traced."""
        print(self.timing_text())
        if self.profile is not None:
            self.profile.show(color=color)


def profile_stages[T, S](
    cases: Mapping[str, Callable[[], T]],
    *,
    sync: Callable[[], S] | None = None,
    trace: bool | Activity = False,
    iters: int = 5,
    warmup: int = 1,
) -> StageProfile:
    """Benchmark each named stage and, when ``trace`` and a GPU are present, trace them.

    cases: ordered map of stage name to a zero-arg callable (bind args with a lambda).
    sync: device barrier called after each run so async GPU work is timed (e.g.
        ``torch.cuda.synchronize``); also drains each region in the trace pass.
    trace: open one CUPTI activity pass over the stages when truthy *and* a GPU is
        present — ``True`` for the default kinds, or an :class:`Activity` flag for exactly
        those. Skipped silently on a CPU-only host, so the same call works everywhere.
    iters/warmup: timed and untimed runs per stage for the wall-clock pass.
    """
    samples = tuple(
        benchmark(fn, label=name, iters=iters, warmup=warmup, sync=sync)
        for name, fn in cases.items()
    )
    profile = _trace_stages(cases, trace, sync) if trace and GPU.all() else None
    return StageProfile(samples=samples, profile=profile)


def _trace_stages[T, S](
    cases: Mapping[str, Callable[[], T]],
    trace: bool | Activity,
    sync: Callable[[], S] | None,
) -> Profile:
    """Run one CUPTI pass, each stage bracketed by a `region` and a device ``sync``."""
    kinds = trace if isinstance(trace, Activity) else Activity.ALL
    with Profiler(trace=kinds) as profiler:
        for name, fn in cases.items():
            with region(name):
                fn()
            if sync is not None:
                sync()
    return profiler.result()
