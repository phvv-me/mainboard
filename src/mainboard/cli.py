import sys

from cyclopts import App

from . import NAME, __version__
from .machine import Machine
from .profiling import Profiler, nvme_to_hbm
from .profiling.python import PythonFormat, PythonMode
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
profiles = App(
    name="profile",
    help="Profile Python, GPU activity, or a live process through one interface.",
)
app.command(profiles)


@app.default
def show(color: bool = True) -> None:
    """Render a Rich terminal view of the current machine."""
    MachineView(Machine()).print(color=color)


@profiles.command(name="run")
def profile_run(
    target: str,
    *,
    python: bool = True,
    spans: bool = True,
    device: bool = True,
    markers: bool = True,
    activity: bool = True,
    mode: PythonMode = PythonMode.WALL,
    format: PythonFormat = PythonFormat.PSTATS,
    output: str = "",
    duration: float | None = None,
    sampling_rate: str = "1khz",
    executable: str = sys.executable,
    timeout: float | None = None,
    color: bool = True,
) -> None:
    """Run a module or script once and show every available profiling lane.

    target: module name or `.py` script path.
    python: collect Tachyon samples when Python 3.15 is available.
    spans: record dormant `span` annotations and the whole program span.
    device: sample target-process GPU telemetry while spans are open.
    markers: emit native NVTX, ROCTx, or signpost ranges for spans.
    activity: collect asynchronous native kernel and memory-copy activities.
    mode: Python wall, CPU, GIL, or exception sampling mode.
    format: Python profile output representation.
    output: optional Python profile artifact path.
    duration: optional Tachyon duration in seconds.
    sampling_rate: Tachyon sample rate such as `1khz` or `20khz`.
    executable: Python executable used by Tachyon and the target.
    timeout: hard deadline for the target process.
    """
    features = Profiler.Feature(0)
    for enabled, feature in (
        (python, Profiler.Feature.PYTHON),
        (spans, Profiler.Feature.SPANS),
        (device, Profiler.Feature.DEVICE),
        (markers, Profiler.Feature.MARKERS),
        (activity, Profiler.Feature.ACTIVITY),
    ):
        if enabled:
            features |= feature
    Profiler.run(
        target,
        features=features,
        mode=mode,
        format=format,
        output=output or None,
        duration=duration,
        sampling_rate=sampling_rate,
        executable=executable,
        timeout=timeout,
    ).show(color=color)


@profiles.command
def attach(
    pid: int,
    *,
    mode: PythonMode = PythonMode.WALL,
    format: PythonFormat = PythonFormat.PSTATS,
    output: str = "",
    duration: float = 30.0,
    sampling_rate: str = "1khz",
    executable: str = sys.executable,
    timeout: float | None = None,
    color: bool = True,
) -> None:
    """Attach Tachyon to a live Python process and show its sampled hotspots."""
    Profiler.attach(
        pid,
        mode=mode,
        format=format,
        output=output or None,
        duration=duration,
        sampling_rate=sampling_rate,
        executable=executable,
        timeout=timeout,
    ).show(color=color)


@profiles.command
def dump(
    pid: int,
    *,
    all_threads: bool = True,
    async_aware: bool = False,
    executable: str = sys.executable,
    timeout: float = 10.0,
    color: bool = True,
) -> None:
    """Print one Python stack snapshot from a live process."""
    Profiler.dump(
        pid,
        all_threads=all_threads,
        async_aware=async_aware,
        executable=executable,
        timeout=timeout,
    ).show(color=color)


@app.command
def storage(file_gb: float = 2.0, iters: int = 5, device: str = "cuda") -> None:
    """Probe node-local NVMe-to-HBM read bandwidth, the page-cache bounce versus the GDS DMA.

    Writes a ``file_gb`` file to the probed scratch tier and reads it into HBM both by the mmap
    page-cache copy and, where a real cuFile DMA is reachable, by GPU Direct Storage, printing the
    achieved GB/s and latency for each plus the speedup. Skips a discipline gracefully when there
    is no CUDA device, no node-local scratch, or no live GDS.

    file_gb: probe file size in gigabytes. iters: timed read count. device: accelerator device.
    """
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
