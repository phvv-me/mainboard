# A tiny, reusable micro-benchmark: time a callable and compare alternatives.

import time
from collections.abc import Callable, Mapping

from patos import FrozenModel


class BenchSample(FrozenModel):
    """Timing of one callable: ``mean_us``/``min_us`` per call over ``runs`` iterations."""

    label: str
    mean_us: float
    min_us: float
    runs: int


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
    best, total = float("inf"), 0.0
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        barrier()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        total += elapsed
    return BenchSample(label=label, mean_us=total / iters * 1e6, min_us=best * 1e6, runs=iters)


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
