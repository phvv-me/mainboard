# Where a dispatch path's receipts go beyond this workspace's own files, and the live machine
# series a running job publishes into them. `base` holds the registry and the one composition
# every path shares; each sink module names its own service and nothing else does.

# Imported for Tracker.__init_subclass__ registration: a sink joins the registry the moment its
# module loads, and this initializer is what every dispatch path imports, so a fresh process can
# mint any declared service by name.
from . import wandb as wandb_sink
from .base import Tracker, credential, is_batched, mirrored, sink, streamed
from .sampler import Sampler, attesting_line, host_env, sampling_line

__all__ = [
    "Sampler",
    "Tracker",
    "attesting_line",
    "is_batched",
    "credential",
    "host_env",
    "mirrored",
    "sampling_line",
    "sink",
    "streamed",
    "wandb_sink",
]
