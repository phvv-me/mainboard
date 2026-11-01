# The public CUDA Python layers available to Mainboard's NVIDIA provider.
#
# NVML is the discovery and sensor layer on every platform. CUDA Runtime and CUDA Core add
# visible-device mapping, runtime metadata, and coherent-memory detection where their optional
# native extensions load. A missing or OS-policy-blocked optional layer degrades to NVML rather
# than making platform identity decide which public CUDA APIs are attempted. A layer that did not
# import leaves its name unbound and its `has_` flag False, so each use is gated on the flag.

nvml_errors: tuple[type[Exception], ...]

try:
    from cuda.bindings import nvml
except ImportError, OSError:
    has_nvml = False
    nvml_errors = ()
else:
    has_nvml = True
    nvml_errors = (
        nvml.NotSupportedError,
        nvml.NoPermissionError,
        nvml.UnknownError,
        nvml.GpuIsLostError,
        nvml.LibraryNotFoundError,
    )

try:
    from cuda.bindings import runtime
except ImportError, OSError:
    has_runtime = False
else:
    has_runtime = True

# The system layer serves only beside the runtime that maps visible devices onto it.
try:
    from cuda.core import system
except ImportError, OSError:
    has_cuda_core = False
else:
    has_cuda_core = has_runtime

__all__ = [
    "has_cuda_core",
    "has_nvml",
    "has_runtime",
    "nvml",
    "nvml_errors",
    "runtime",
    "system",
]
