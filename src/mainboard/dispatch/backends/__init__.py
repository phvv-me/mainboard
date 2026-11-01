# Non-ssh provider backends, which `route` picks by `HostProfile.kind` (Modal Sandboxes, HPC-AI
# instances, Vast.ai rentals, and the `CloudBackend` clouds held as ssh hosts: RunPod, Lambda,
# DigitalOcean, Verda, Vultr, Hot Aisle, Hyperstack, Shadeform). Each REST backend reads its key
# through its module's `api_key`; the one exported here is HPC-AI's, so `vast.api_key` is imported
# from its module by name.

from .base import Credentials, ProviderBackend, Rentable, http_transport, route
from .capabilities import (
    Account,
    Capability,
    Delivery,
    Inventory,
    LogSource,
    Market,
    Rented,
    Standing,
)
from .cloud.core import CloudBackend
from .cloud.curated.hotaisle import HotAisleBackend
from .cloud.curated.hyperstack import HyperstackBackend
from .cloud.curated.runpod import RunPodBackend
from .cloud.curated.shadeform import ShadeformBackend
from .cloud.listed.digitalocean import DigitalOceanBackend
from .cloud.listed.lambdacloud import LambdaBackend
from .cloud.listed.verda import VerdaBackend
from .cloud.listed.vultr import VultrBackend
from .hpcai import HpcAiBackend, api_key
from .modal import ModalBackend
from .vast import VastBackend

__all__ = [
    "Account",
    "Capability",
    "CloudBackend",
    "Credentials",
    "Delivery",
    "DigitalOceanBackend",
    "HotAisleBackend",
    "HpcAiBackend",
    "HyperstackBackend",
    "Inventory",
    "LambdaBackend",
    "LogSource",
    "Market",
    "ModalBackend",
    "ProviderBackend",
    "RunPodBackend",
    "Rentable",
    "Rented",
    "ShadeformBackend",
    "Standing",
    "VastBackend",
    "VerdaBackend",
    "VultrBackend",
    "api_key",
    "http_transport",
    "route",
]
