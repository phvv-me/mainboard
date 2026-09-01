import platform
from contextlib import suppress
from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .protocols import CoreDeviceType, CoreSystem, CudaRuntime, Nvml


def text(value: bytes | str) -> str:
    """Convert CUDA/NVML byte strings and scalars to text."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


class NvidiaApis:
    """The public CUDA Python layers available to Mainboard's NVIDIA provider.

    NVML is the discovery and sensor layer on every platform. CUDA Runtime and CUDA Core add
    visible-device mapping, runtime metadata, and coherent-memory detection where native
    extension policy permits them. Windows currently keeps the ordinary control-plane probe on
    NVML alone: that avoids importing unrelated runtime/compiler extensions during a facts read
    while preserving the same provider interface.
    """

    if TYPE_CHECKING:
        runtime: CudaRuntime | None
        nvml: Nvml
        system: CoreSystem | None
        cuda_device_type: CoreDeviceType | None

    def __init__(self) -> None:
        self.nvml = cast("Nvml", import_module("cuda.bindings.nvml"))
        self.runtime = None
        self.system = None
        self.cuda_device_type = None
        if platform.system() != "Windows":
            self._load_optional_cuda_layers()
        self.nvml_errors: tuple[type[Exception], ...] = tuple(
            error
            for name in (
                "NotSupportedError",
                "NoPermissionError",
                "UnknownError",
                "GpuIsLostError",
                "LibraryNotFoundError",
            )
            if isinstance(error := getattr(self.nvml, name, None), type)
            and issubclass(error, Exception)
        )

    def _load_optional_cuda_layers(self) -> None:
        """Load richer CUDA layers without making either one a discovery requirement."""
        with suppress(ImportError, OSError):
            self.runtime = cast("CudaRuntime", import_module("cuda.bindings.runtime"))
        if self.runtime is not None:
            with suppress(ImportError, OSError):
                self.system = cast("CoreSystem", import_module("cuda.core.system"))
                self.cuda_device_type = import_module("cuda.core").Device

    @property
    def has_cuda_core(self) -> bool:
        """Whether the optional `cuda.core` layer loaded successfully."""
        return self.cuda_device_type is not None


@cache
def nvidia_apis() -> NvidiaApis:
    """Return cached CUDA/NVML imports for NVIDIA devices."""
    return NvidiaApis()
