# Non-ssh provider backends: `route` picks between the existing ssh-family scheduler path and a
# registered `ProviderBackend` (Modal Sandboxes, HPC-AI instances, Vast.ai rentals) by
# `HostProfile.kind`. Each REST backend reads its own key through its module's `api_key`, so the
# one exported here stays HPC-AI's and `vast.api_key` is imported from its module by name.

from .base import (
    Account,
    Capability,
    Credentials,
    Delivery,
    LogSource,
    Market,
    ProviderBackend,
    Standing,
    http_transport,
    route,
)
from .hpcai import HpcAiBackend, api_key
from .modal import ModalBackend
from .vast import VastBackend

__all__ = [
    "Account",
    "Capability",
    "Credentials",
    "Delivery",
    "HpcAiBackend",
    "LogSource",
    "Market",
    "ModalBackend",
    "ProviderBackend",
    "Standing",
    "VastBackend",
    "api_key",
    "http_transport",
    "route",
]
