"""NVMe-to-HBM read-bandwidth probe: how fast node-local scratch feeds the accelerator.

A spill tier that parks tensors on node-local NVMe and streams them into HBM lives or dies by this
bandwidth, and there are two disciplines for the read. The mmap path reads the file into the page
cache and copies the bytes host-to-device, a NVMe -> page cache -> HBM bounce. GPU Direct Storage
(cuFile, through kvikio) DMAs the bytes from NVMe straight into the device tensor, skipping the
host bounce. This probe writes a known-size file to the probed scratch tier and times both reads,
reporting achieved GB/s and per-read latency, so a host can see whether GDS actually beats the
bounce before a serving tier commits to it.

The probe degrades the way the rest of mainboard does. No CUDA device, no writable node-local
scratch, or no torch and it reports unavailable rather than raising. kvikio absent, or present but
stuck in cuFile compat mode (a host-staged POSIX read masquerading as a DMA), and it reports the
mmap number alone with GDS skipped, never a compat bounce dressed up as GDS. Where both run, the
two GB/s figures sit side by side as the evidence that the direct path is live.
"""

import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import cast

from ..models.base import FrozenModel
from ..models.scratch import Scratch
from .benchmark import benchmark
from .protocols import CurrentKvikioDefaults, Kvikio, KvikioCompatibility, Torch


def torch_api() -> Torch:
    """Load the optional Torch module through its structural contract."""
    return cast("Torch", import_module("torch"))


def kvikio_api() -> Kvikio:
    """Load the optional kvikio module through its structural contract."""
    return cast("Kvikio", import_module("kvikio"))


class ReadResult(FrozenModel):
    """One read discipline's measured bandwidth over the probe file.

    label: the discipline, ``mmap`` (page-cache copy) or ``gds`` (direct NVMe -> HBM DMA).
    gigabytes_per_s: achieved read bandwidth, the file size over the mean per-read wall time.
    latency_ms: mean wall time of one full-file read, in milliseconds.
    """

    label: str
    gigabytes_per_s: float
    latency_ms: float


class StorageBandwidth(FrozenModel):
    """The NVMe-to-HBM read-bandwidth probe's result, the mmap and GDS reads side by side.

    available: whether the probe ran (a CUDA device, a writable scratch tier and torch were found).
    scratch_path: the node-local scratch dir the probe wrote its file to, or ``None`` when none.
    file_gb: the probe file's size in gigabytes, the bytes each read moves.
    mmap: the page-cache read result, the baseline bounce, present whenever the probe ran.
    gds: the GPU Direct Storage read result, present only where a real cuFile DMA was reachable.
    skipped: why a discipline did not run (no CUDA, no scratch, kvikio absent or in compat mode).
    """

    available: bool = False
    scratch_path: Path | None = None
    file_gb: float = 0.0
    mmap: ReadResult | None = None
    gds: ReadResult | None = None
    skipped: str = ""

    @property
    def speedup(self) -> float | None:
        """How many times the GDS read beats the mmap bounce, or ``None`` when GDS did not run."""
        if self.gds is None or self.mmap is None or self.mmap.gigabytes_per_s == 0:
            return None
        return self.gds.gigabytes_per_s / self.mmap.gigabytes_per_s


def nvme_to_hbm(
    file_gb: float = 2.0, iters: int = 5, warmup: int = 1, device: str = "cuda"
) -> StorageBandwidth:
    """Measure NVMe-to-HBM read bandwidth on node-local scratch, mmap bounce versus GDS DMA.

    Writes a ``file_gb`` file of random bytes to the probed scratch tier, then times reading it
    into an HBM tensor both ways: the mmap path (read into page cache, copy host-to-device) and,
    where a real cuFile DMA is reachable, the GPU Direct Storage path (DMA straight into the device
    tensor). Each read drops the page cache first so the mmap number is a true cold-file read, not
    a RAM hit. Reports both achieved GB/s and per-read latency. Returns an unavailable result
    rather than raising when there is no CUDA device, no writable scratch, or no torch.

    file_gb: the probe file's size in gigabytes, large enough to amortise per-call overhead.
    iters, warmup: timed and untimed read counts the benchmark averages over.
    device: the accelerator device the file is read into.
    """
    if find_spec("torch") is None:
        return StorageBandwidth(skipped="torch not installed")
    torch = torch_api()

    scratch = Scratch.probe()
    if not torch.cuda.is_available() or scratch.path is None:
        why = "no CUDA device" if not torch.cuda.is_available() else "no node-local scratch"
        return StorageBandwidth(scratch_path=scratch.path, skipped=why)

    path = scratch.path / ".mainboard_nvme_hbm_probe.bin"
    nbytes = int(file_gb * 1024**3)
    try:
        write_probe_file(path, nbytes)
        mmap = read_mmap(path, nbytes, device, iters, warmup)
        gds, skipped = read_gds(path, nbytes, device, iters, warmup)
        return StorageBandwidth(
            available=True,
            scratch_path=scratch.path,
            file_gb=file_gb,
            mmap=mmap,
            gds=gds,
            skipped=skipped,
        )
    finally:
        path.unlink(missing_ok=True)


def write_probe_file(path: Path, nbytes: int) -> None:
    """Write ``nbytes`` of random bytes to ``path`` and flush them to disk.

    Random rather than zeros so a filesystem that elides all-zero blocks cannot shortcut the read,
    and ``fsync`` so the bytes are on the device before any timed read rather than in a dirty
    write-back buffer. Written in chunks to keep the host staging buffer bounded.
    """
    chunk = 256 * 1024**2
    with path.open("wb") as handle:
        written = 0
        while written < nbytes:
            block = os.urandom(min(chunk, nbytes - written))
            handle.write(block)
            written += len(block)
        handle.flush()
        os.fsync(handle.fileno())


def drop_page_cache(path: Path) -> None:
    """Evict ``path`` from the page cache so the next read hits NVMe, not RAM.

    ``posix_fadvise(DONTNEED)`` drops the file's resident pages without root, so the mmap read
    measures a cold-file NVMe stream rather than a page-cache hit and is comparable to the GDS DMA
    that never touches the cache. Best-effort: a platform without the advice leaves the cache warm,
    which only understates the GDS speedup.
    """
    if not hasattr(os, "posix_fadvise"):
        return  # macOS and Windows lack the advice; the mmap read stays a best-effort cold read
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_mmap(path: Path, nbytes: int, device: str, iters: int, warmup: int) -> ReadResult:
    """Time the mmap read: file into a host buffer, copy host-to-device, the page-cache bounce.

    Reads the whole file into a host tensor and copies it to the device, the NVMe -> page cache ->
    HBM path a mmap-tier gather takes, dropping the cache before each read so the NVMe stream is
    cold. The device sync is folded into the timing so the copy's async tail counts.
    """
    torch = torch_api()

    out = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def once() -> None:
        drop_page_cache(path)
        host = torch.from_file(str(path), dtype=torch.uint8, size=nbytes)
        out.copy_(host)

    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    sample = benchmark(once, label="mmap", iters=iters, warmup=warmup, sync=sync)
    return as_result("mmap", nbytes, sample.mean_us)


def read_gds(
    path: Path, nbytes: int, device: str, iters: int, warmup: int
) -> tuple[ReadResult | None, str]:
    """Time the GDS read when a real cuFile DMA is reachable, else report why it was skipped.

    DMAs the whole file straight from NVMe into the device tensor through a ``kvikio.CuFile``
    handle, the direct path with no page-cache bounce, dropping the cache before each read for
    parity with the mmap timing. Returns ``None`` with a reason when kvikio is absent or stuck in
    cuFile compat mode (a host-staged POSIX read that is not a DMA), so a compat bounce is never
    reported as GDS.
    """
    if find_spec("kvikio") is None:
        return None, "kvikio not installed"
    kvikio = kvikio_api()
    torch = torch_api()

    if compat_mode_preferred(kvikio):
        return None, "kvikio in cuFile compat mode (host bounce, not a GDS DMA)"

    out = torch.empty(nbytes, dtype=torch.uint8, device=device)

    def once() -> None:
        drop_page_cache(path)
        with kvikio.CuFile(str(path), "r") as handle:
            handle.read(out)

    try:
        once()  # one probe read: a mount that refuses the DMA (EPERM in a job sandbox) skips here
    except (OSError, RuntimeError) as refused:
        return None, f"cuFile DMA refused on {path} ({refused})"

    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    sample = benchmark(once, label="gds", iters=iters, warmup=warmup, sync=sync)
    return as_result("gds", nbytes, sample.mean_us), ""


def compat_mode_preferred(kvikio: KvikioCompatibility) -> bool:
    """Whether kvikio prefers cuFile compat mode, across the submodule and dual-API differences.

    The modern cu13 wheel does not auto-import ``kvikio.defaults`` on ``import kvikio`` and renamed
    the getter to ``is_compat_mode_preferred``, while older builds exposed ``compat_mode()``
    returning a ``CompatMode`` enum. Import the submodule (aliased, so it attaches to the real
    ``kvikio`` package as a side effect without rebinding this function's own parameter) when the
    attribute is missing, then read whichever getter the build offers, so a stuck-in-compat probe
    is caught on every kvikio.
    """
    if not hasattr(kvikio, "defaults"):
        import_module("kvikio.defaults")  # pragma: no cover
    defaults = kvikio.defaults
    if isinstance(defaults, CurrentKvikioDefaults):
        return bool(defaults.is_compat_mode_preferred())
    return defaults.compat_mode() is not kvikio.CompatMode.OFF


def as_result(label: str, nbytes: int, mean_us: float) -> ReadResult:
    """A :class:`ReadResult` from a read's byte count and mean microsecond wall time."""
    seconds = mean_us / 1e6
    gbps = (nbytes / 1e9) / seconds if seconds > 0 else 0.0
    return ReadResult(label=label, gigabytes_per_s=gbps, latency_ms=mean_us / 1e3)
