from cyclopts import App

from . import NAME, __version__
from .machine import Machine
from .visual import MachineView

# Pin the version explicitly: left to its default, cyclopts derives `--version`
# from `importlib.metadata` of the calling package, which silently drifts to a
# stale wheel's number under an editable install. The in-tree `__version__` is
# the single source of truth, so the CLI always reports the code it is running.
app = App(
    name=NAME,
    version=__version__,
    help="Inspect CPU, GPU, and NPU hardware topology.",
)


@app.default
def show(color: bool = True) -> None:
    """Render a Rich terminal view of the current machine."""
    MachineView(Machine()).print(color=color)


@app.command
def profile(
    target: str,
    *,
    auto: str = "",
    trace: bool = False,
    perfetto: str = "",
    color: bool = True,
) -> None:
    """Run a module or script under the profiler and show where time goes.

    target: a module name (``pkg.mod``) or a ``.py`` script path, run as ``__main__``.
    auto: comma-separated package prefixes to auto-annotate (every call becomes a region).
    trace: enable deep per-kernel tracing (CUDA). perfetto: also write a timeline JSON to
    this path (open at ui.perfetto.dev).
    """
    import runpy

    from .profiling import Profiler

    modules = tuple(part for part in auto.split(",") if part)
    with Profiler(trace=trace, auto=modules) as profiler:
        if target.endswith(".py"):
            runpy.run_path(target, run_name="__main__")
        else:
            runpy.run_module(target, run_name="__main__")
    result = profiler.result()
    result.show(color=color)
    if perfetto:
        result.perfetto(perfetto)


@app.command
def storage(file_gb: float = 2.0, iters: int = 5, device: str = "cuda") -> None:
    """Probe node-local NVMe-to-HBM read bandwidth, the page-cache bounce versus the GDS DMA.

    Writes a ``file_gb`` file to the probed scratch tier and reads it into HBM both by the mmap
    page-cache copy and, where a real cuFile DMA is reachable, by GPU Direct Storage, printing the
    achieved GB/s and latency for each plus the speedup. Skips a discipline gracefully when there
    is no CUDA device, no node-local scratch, or no live GDS.

    file_gb: probe file size in gigabytes. iters: timed read count. device: accelerator device.
    """
    from .profiling import nvme_to_hbm

    result = nvme_to_hbm(file_gb=file_gb, iters=iters, device=device)
    if not result.available:
        print(f"nvme->hbm probe unavailable: {result.skipped}")
        return
    print(f"nvme->hbm read bandwidth on {result.scratch_path} ({result.file_gb:.1f} GB file)")
    for read in (result.mmap, result.gds):
        if read is not None:
            print(
                f"  {read.label:5s}  {read.gigabytes_per_s:8.2f} GB/s  {read.latency_ms:8.2f} ms"
            )
    if result.speedup is not None:
        print(f"  gds is {result.speedup:.2f}x the mmap bounce")
    elif result.skipped:
        print(f"  gds skipped: {result.skipped}")


def main() -> None:
    """Run the Mainboard command-line interface."""
    app()


if __name__ == "__main__":
    main()
