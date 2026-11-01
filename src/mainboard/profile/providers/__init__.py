# Imported for Tracer.__init_subclass__ registration: Tracer.detect imports this package before
# walking implementations(), so every vendor backend must be reachable from here. A backend whose
# annotation library is not installed fails its import and never registers.
from contextlib import suppress

from .nvidia import tracer as nvidia_tracer

with suppress(ImportError):
    from .amd import tracer as amd_tracer
with suppress(ImportError):
    from .apple import tracer as apple_tracer

__all__ = ["amd_tracer", "apple_tracer", "nvidia_tracer"]
