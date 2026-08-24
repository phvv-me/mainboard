# A tiny, reusable micro-benchmark: time a callable and compare alternatives.

import time
from collections.abc import Callable, Mapping
from statistics import fmean

from patos import FrozenModel


class BenchSample(FrozenModel):
    """Timing of one callable: every per-iteration time in ``samples``, plus its aggregates.

    samples: microseconds for each timed iteration, in the order they ran, so a caller that
        needs the per-run rows reads them here instead of driving one-iteration benchmarks.
        ``mean_us``, ``min_us`` and ``runs`` are read off this one record.
    """

    label: str
    samples: tuple[float, ...]

    @property
    def mean_us(self) -> float:
        """Mean microseconds per call over the timed iterations."""
        return fmean(self.samples)

    @property
    def min_us(self) -> float:
        """Fastest microsecond time over the timed iterations."""
        return min(self.samples)

    @property
    def runs(self) -> int:
        """How many timed iterations were recorded."""
        return len(self.samples)


def _noop() -> None:
    pass


def benchmark[T, S](
    fn: Callable[[], T],
    *,
    label: str = "fn",
    iters: int = 20,
    warmup: int = 3,
    sync: Callable[[], S] | None = None,
) -> BenchSample:
    """Time ``fn`` over ``iters`` runs after ``warmup`` untimed calls.

    fn: the zero-arg callable to time (bind args with a lambda/partial).
    sync: a barrier called after each run (e.g. ``torch.cuda.synchronize``) so async GPU
        work is included in the timing rather than just the launch.
    """
    barrier: Callable[[], S | None] = sync or _noop
    for _ in range(warmup):
        fn()
    barrier()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        barrier()
        samples.append((time.perf_counter() - start) * 1e6)
    return BenchSample(label=label, samples=tuple(samples))


def compare[T, S](
    cases: Mapping[str, Callable[[], T]],
    *,
    iters: int = 20,
    warmup: int = 3,
    sync: Callable[[], S] | None = None,
) -> list[BenchSample]:
    """Benchmark each named callable and print a mean/min-time table, fastest first."""
    samples = sorted(
        (
            benchmark(fn, label=name, iters=iters, warmup=warmup, sync=sync)
            for name, fn in cases.items()
        ),
        key=lambda s: s.mean_us,
    )
    width = max((len(s.label) for s in samples), default=4)
    for sample in samples:
        print(f"{sample.label:{width}s}  {sample.mean_us:10.3f} us  (min {sample.min_us:.3f})")
    return samples
