from __future__ import annotations

from enum import StrEnum


class CompilerKind(StrEnum):
    """Normalized host compiler kind."""

    CLANG_GRACE = "clang-grace"
    GCC = "gcc"
    CLANG = "clang"
    NVCC = "nvcc"
    UNKNOWN = "unknown"


class SystemFamily(StrEnum):
    """Machine-family grouping used to guide build choices."""

    GRACE_HOPPER = "grace-hopper"
    GRACE_BLACKWELL = "grace-blackwell"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"
    GENERIC = "generic"


class CudaPythonVariant(StrEnum):
    """CUDA Python stack family required on the current system."""

    CU12 = "cu12"
    CU13 = "cu13"


class CmakeBuildType(StrEnum):
    """CMake build type string."""

    RELEASE = "Release"


class DiskKind(StrEnum):
    """Drive technology for a mounted disk or partition."""

    NVME = "nvme"
    SSD = "ssd"
    HDD = "hdd"
    RAMDISK = "ramdisk"
    UNKNOWN = "unknown"


class UnitKind(StrEnum):
    """Schedulable execution-resource category."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    DSP = "dsp"
    MEDIA = "media"
    UNKNOWN = "unknown"


class Vendor(StrEnum):
    """Normalized hardware vendor."""

    ARM = "arm"
    APPLE = "apple"
    NVIDIA = "nvidia"
    QUALCOMM = "qualcomm"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"
