"""Tests for the NVMe-to-HBM read-bandwidth probe, the mmap bounce versus the GDS DMA.

The probe writes a known-size file to node-local scratch and times reading it into the accelerator
both ways. These tests drive it on the CPU with a tiny file and a fake kvikio so every branch runs
without a GPU or the GDS driver: the unavailable paths (no torch, no CUDA, no scratch), the mmap
read, the GDS read, and the skips (kvikio absent, kvikio in compat mode). The byte arithmetic in
``as_result`` and the ``speedup`` ratio are checked directly.
"""

import os
import sys
import types
from pathlib import Path

import pytest

import mainboard.profiling.storage as storage_mod
from mainboard.models.scratch import Scratch
from mainboard.profiling.storage import (
    ReadResult,
    StorageBandwidth,
    as_result,
    drop_page_cache,
    nvme_to_hbm,
    write_probe_file,
)


def force_posix_fadvise(monkeypatch: pytest.MonkeyPatch, *, present: bool) -> list[int]:
    """Force `os.posix_fadvise` to exist (or not), regardless of the real host platform.

    `drop_page_cache` is best-effort across a POSIX advice only Linux offers, and the CI
    matrix gates coverage on both Linux and macOS, so each branch needs to be reachable on
    either runner rather than relying on whichever platform happens to run the test.
    Returns the list `posix_fadvise`'s ``advice`` arguments land in, when forced present.
    """
    if not present:
        monkeypatch.delattr(os, "posix_fadvise", raising=False)
        return []
    calls: list[int] = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda fd, offset, length, advice: calls.append(advice), raising=False
    )
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    return calls


def force_scratch(monkeypatch: pytest.MonkeyPatch, path: Path | None) -> None:
    """Pin the probed scratch tier to ``path`` (or none) so a test controls the file location."""
    scratch = Scratch(path=path, free_bytes=10 * 1024**3, source="TMP") if path else Scratch()
    monkeypatch.setattr(Scratch, "probe", classmethod(lambda cls: scratch))


def force_cuda(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    """Pretend the host has (or lacks) a CUDA device so the probe's gate is testable on CPU."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def fake_kvikio(compat: bool) -> types.SimpleNamespace:
    """A CPU stand-in for kvikio whose ``CuFile.read`` fills the device buffer from the file."""

    class CuFile:
        def __init__(self, path: str, flags: str) -> None:
            self.handle = open(path, "rb")  # noqa: SIM115 (closed in __exit__)

        def __enter__(self) -> CuFile:
            return self

        def __exit__(self, *exc: object) -> None:
            self.handle.close()

        def read(self, buf: object) -> int:
            import torch

            raw = self.handle.read()
            flat = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            buf.copy_(flat)  # pyrefly: ignore  # buf is a uint8 tensor in the probe
            return len(raw)

    mode = types.SimpleNamespace(ON=True, OFF=False)
    return types.SimpleNamespace(
        CompatMode=mode,
        CuFile=CuFile,
        defaults=types.SimpleNamespace(compat_mode=lambda: mode.ON if compat else mode.OFF),
    )


def fake_kvikio_refusing_dma() -> types.SimpleNamespace:
    """A stand-in for kvikio whose ``CuFile.read`` refuses the DMA, as a sandboxed mount would."""

    class CuFile:
        def __init__(self, path: str, flags: str) -> None:
            pass

        def __enter__(self) -> CuFile:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def read(self, buf: object) -> int:
            raise OSError("Operation not permitted")

    mode = types.SimpleNamespace(ON=False, OFF=False)
    return types.SimpleNamespace(
        CompatMode=mode,
        CuFile=CuFile,
        defaults=types.SimpleNamespace(compat_mode=lambda: mode.OFF),
    )


def install_kvikio(monkeypatch: pytest.MonkeyPatch, module: object | None) -> None:
    """Make ``find_spec('kvikio')`` resolve to ``module`` (or be absent), leaving torch intact.

    The probe gates both torch and kvikio through ``find_spec``, so the patch must answer for the
    ``kvikio`` name alone and defer every other name (notably ``torch``) to the real resolver.
    """
    real = storage_mod.find_spec

    def resolve(name: str) -> object | None:
        if name == "kvikio":
            return object() if module is not None else None
        return real(name)

    monkeypatch.setattr(storage_mod, "find_spec", resolve)
    if module is not None:
        monkeypatch.setitem(sys.modules, "kvikio", module)


def test_unavailable_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No torch and the probe reports unavailable rather than importing it."""
    monkeypatch.setattr(storage_mod, "find_spec", lambda name: None)
    result = nvme_to_hbm()
    assert result.available is False
    assert "torch" in result.skipped


def test_unavailable_without_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A host with scratch but no CUDA device reports unavailable, naming the missing device."""
    force_cuda(monkeypatch, available=False)
    force_scratch(monkeypatch, tmp_path)
    result = nvme_to_hbm()
    assert result.available is False
    assert "CUDA" in result.skipped


def test_unavailable_without_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with CUDA but no writable scratch reports unavailable, naming the missing tier."""
    force_cuda(monkeypatch, available=True)
    force_scratch(monkeypatch, None)
    result = nvme_to_hbm()
    assert result.available is False
    assert "scratch" in result.skipped


def test_mmap_only_when_kvikio_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no kvikio the probe runs the mmap read alone and skips GDS with a reason."""
    force_cuda(monkeypatch, available=True)
    force_scratch(monkeypatch, tmp_path)
    install_kvikio(monkeypatch, None)
    result = nvme_to_hbm(file_gb=1 / 1024, iters=2, warmup=1, device="cpu")
    assert result.available is True
    assert result.mmap is not None and result.mmap.gigabytes_per_s > 0
    assert result.gds is None
    assert "kvikio" in result.skipped
    assert result.speedup is None
    assert not (tmp_path / ".mainboard_nvme_hbm_probe.bin").exists()  # cleaned up


def test_gds_skipped_in_compat_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """kvikio in cuFile compat mode is a host bounce, so GDS is skipped rather than reported."""
    force_cuda(monkeypatch, available=True)
    force_scratch(monkeypatch, tmp_path)
    install_kvikio(monkeypatch, fake_kvikio(compat=True))
    result = nvme_to_hbm(file_gb=1 / 1024, iters=2, warmup=1, device="cpu")
    assert result.available is True
    assert result.gds is None
    assert "compat" in result.skipped


def test_gds_dma_refused_is_reported_as_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A mount that refuses the DMA (EPERM in a job sandbox) skips GDS, naming the refusal."""
    force_cuda(monkeypatch, available=True)
    force_scratch(monkeypatch, tmp_path)
    install_kvikio(monkeypatch, fake_kvikio_refusing_dma())
    result = nvme_to_hbm(file_gb=1 / 1024, iters=2, warmup=1, device="cpu")
    assert result.available is True
    assert result.gds is None
    assert "refused" in result.skipped


def test_gds_and_mmap_both_run_when_gds_is_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live GDS path produces both reads side by side, with a defined speedup ratio."""
    force_cuda(monkeypatch, available=True)
    force_scratch(monkeypatch, tmp_path)
    install_kvikio(monkeypatch, fake_kvikio(compat=False))
    result = nvme_to_hbm(file_gb=1 / 1024, iters=2, warmup=1, device="cpu")
    assert result.available is True
    assert result.mmap is not None and result.mmap.gigabytes_per_s > 0
    assert result.gds is not None and result.gds.gigabytes_per_s > 0
    assert result.skipped == ""
    assert result.speedup == pytest.approx(
        result.gds.gigabytes_per_s / result.mmap.gigabytes_per_s
    )


def test_compat_mode_preferred_reads_the_modern_getter() -> None:
    """The cu13/26.x kvikio reports its preference through ``is_compat_mode_preferred``."""
    kvikio = types.SimpleNamespace(
        defaults=types.SimpleNamespace(is_compat_mode_preferred=lambda: True)
    )
    assert storage_mod.compat_mode_preferred(kvikio) is True


def test_compat_mode_preferred_falls_back_to_the_legacy_enum() -> None:
    """An older kvikio without the new getter is read via ``compat_mode()`` vs ``CompatMode``."""
    mode = types.SimpleNamespace(ON=object(), OFF=object())
    kvikio = types.SimpleNamespace(
        CompatMode=mode, defaults=types.SimpleNamespace(compat_mode=lambda: mode.OFF)
    )
    assert storage_mod.compat_mode_preferred(kvikio) is False


def test_write_probe_file_lands_the_requested_bytes(tmp_path: Path) -> None:
    """The probe file is exactly the requested size, written in bounded chunks and fsynced."""
    path = tmp_path / "probe.bin"
    write_probe_file(path, 5 * 1024)
    assert path.stat().st_size == 5 * 1024


def test_as_result_turns_bytes_and_time_into_bandwidth() -> None:
    """One gigabyte read in one millisecond is a terabyte per second, the GB/s arithmetic."""
    result = as_result("mmap", nbytes=10**9, mean_us=1000.0)
    assert result.gigabytes_per_s == pytest.approx(1000.0)
    assert result.latency_ms == pytest.approx(1.0)


def test_as_result_is_zero_on_a_zero_time_read() -> None:
    """A degenerate zero-time read reports zero bandwidth rather than dividing by zero."""
    assert as_result("gds", nbytes=10**9, mean_us=0.0).gigabytes_per_s == 0.0


def test_speedup_is_none_without_a_gds_read() -> None:
    """The speedup ratio is undefined when only the mmap read ran."""
    only_mmap = StorageBandwidth(
        available=True, mmap=ReadResult(label="mmap", gigabytes_per_s=5.0, latency_ms=1.0)
    )
    assert only_mmap.speedup is None


def test_drop_page_cache_is_silent_on_a_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cache eviction is best-effort: an absent file is a no-op, never a raised error."""
    force_posix_fadvise(monkeypatch, present=True)
    drop_page_cache(tmp_path / "does_not_exist.bin")  # returns without raising


def test_drop_page_cache_is_a_noop_without_the_posix_advice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """macOS and Windows lack `posix_fadvise`, so the cache is left warm rather than raising."""
    calls = force_posix_fadvise(monkeypatch, present=False)
    path = tmp_path / "probe.bin"
    path.write_bytes(b"x")
    drop_page_cache(path)
    assert calls == []


def test_drop_page_cache_evicts_with_dontneed_when_the_advice_is_offered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Where `posix_fadvise` exists, eviction actually calls it with `POSIX_FADV_DONTNEED`."""
    calls = force_posix_fadvise(monkeypatch, present=True)
    path = tmp_path / "probe.bin"
    path.write_bytes(b"x")
    drop_page_cache(path)
    assert calls == [os.POSIX_FADV_DONTNEED]
