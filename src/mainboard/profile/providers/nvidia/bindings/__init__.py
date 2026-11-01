# The accelerator bindings, each present only where its package is installed. A binding that did
# not import leaves its name unbound and its `has_` flag False, so each use is gated on the flag.

try:
    import nvtx
except ImportError:
    has_nvtx = False
else:
    has_nvtx = True

try:
    from cupti import cupti
except ImportError:
    has_cupti = False
else:
    has_cupti = True

try:
    from cuda.bindings import driver, runtime
except ImportError, OSError:
    has_cuda = False
else:
    has_cuda = True

__all__ = ["cupti", "driver", "has_cuda", "has_cupti", "has_nvtx", "nvtx", "runtime"]
