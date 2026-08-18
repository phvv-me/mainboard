from mainboard.probe import GPU, NPU, UnitKind, Vendor


def test_base_npu_neutral_defaults() -> None:
    """The base `NPU` reports the NPU kind with unknown identity."""
    npu = NPU()
    assert npu.kind == UnitKind.NPU
    assert npu.vendor == Vendor.UNKNOWN
    assert npu.label == "unknown"


def test_gpu_all_skips_a_provider_that_raises() -> None:
    """One provider whose `all` throws must not sink the whole machine probe.

    A backend can load and then fail mid-probe (an unexpected NVML error, a
    binding that imports but throws). `GPU.all` fans out best-effort, so a
    raising provider is dropped and the surviving providers still report.
    """

    class GoodGPU(GPU):
        @classmethod
        def all(cls) -> tuple[GPU, ...]:
            return (cls(index=0),)

    class BrokenGPU(GPU):
        @classmethod
        def all(cls) -> tuple[GPU, ...]:
            raise RuntimeError("backend exploded mid-probe")

    gpus = GPU.all()
    assert any(isinstance(gpu, GoodGPU) for gpu in gpus)
    assert not any(isinstance(gpu, BrokenGPU) for gpu in gpus)


def test_npu_all_skips_a_provider_that_raises() -> None:
    """The NPU fan-out degrades per provider exactly like the GPU one."""

    class GoodNPU(NPU):
        @classmethod
        def all(cls) -> tuple[NPU, ...]:
            return (cls(index=0),)

    class BrokenNPU(NPU):
        @classmethod
        def all(cls) -> tuple[NPU, ...]:
            raise OSError("driver handle vanished")

    npus = NPU.all()
    assert any(isinstance(npu, GoodNPU) for npu in npus)
    assert not any(isinstance(npu, BrokenNPU) for npu in npus)
