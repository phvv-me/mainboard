from contextlib import suppress
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import CoreDeviceType, CoreSystem, CudaRuntime, Nvml


def text(value: bytes | str) -> str:
    """Convert CUDA/NVML byte strings and scalars to text."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


# The bindings ship no type stubs, so each loader returns the Protocol in `protocols.py`
# (the surface mainboard calls) bound to the real module. `import_module` returns an untyped
# `ModuleType` that pyrefly cannot bridge to a Protocol (its `__getattr__` defeats the
# structural check), so each loader carries a `pyrefly: ignore` for that genuine stub gap.
def _load_runtime() -> CudaRuntime:
    return import_module("cuda.bindings.runtime")  # pyrefly: ignore[bad-return]


def _load_nvml() -> Nvml:
    try:
        return import_module("cuda.bindings._nvml")  # pyrefly: ignore[bad-return]
    except ImportError:
        return import_module("cuda.bindings.nvml")  # pyrefly: ignore[bad-return]


def _load_system() -> CoreSystem:
    return import_module("cuda.core.system")  # pyrefly: ignore[bad-return]


class NvidiaApis:
    """CUDA/NVML imports for the NVIDIA provider.

    The bindings ship no type stubs, so the handles are declared as the Protocols in
    `protocols.py` (the surface mainboard actually calls) and bound to the real, untyped
    modules at runtime. `cuda.bindings` (runtime + NVML) is the required, ABI-stable layer
    for device discovery, memory, and live sensors. `cuda.core` is an optional convenience
    layer: its compiled extensions can fail to load when another library (e.g. a torch
    wheel bundling an older `libstdc++`) is imported first, so the provider keeps working
    without it.
    """

    if TYPE_CHECKING:
        runtime: CudaRuntime
        nvml: Nvml
        system: CoreSystem | None
        cuda_device_type: CoreDeviceType | None

    def __init__(self) -> None:
        self.runtime = _load_runtime()
        self.nvml = _load_nvml()
        self.system = None
        self.cuda_device_type = None
        with suppress(ImportError):
            self.system = _load_system()
            self.cuda_device_type = import_module("cuda.core").Device
        self.nvml_errors: tuple[type[Exception], ...] = tuple(
            error
            for name in (
                "NotSupportedError",
                "NoPermissionError",
                "UnknownError",
                "GpuIsLostError",
            )
            if isinstance(error := getattr(self.nvml, name, None), type)
            and issubclass(error, Exception)
        )

    @property
    def has_cuda_core(self) -> bool:
        """Whether the optional `cuda.core` layer loaded successfully."""
        return self.cuda_device_type is not None


@cache
def nvidia_apis() -> NvidiaApis:
    """Return cached CUDA/NVML imports for NVIDIA devices."""
    return NvidiaApis()
