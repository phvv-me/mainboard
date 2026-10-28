# Non-ssh provider backends, which `route` picks by `HostProfile.kind` (Modal Sandboxes, HPC-AI
# instances, Vast.ai rentals, and the `CloudBackend` clouds held as ssh hosts: RunPod, Lambda,
# DigitalOcean, Verda, Vultr, Hot Aisle, Hyperstack, Shadeform). Each REST backend reads its key
# through its module's `api_key`; the one exported here is HPC-AI's, so `vast.api_key` is imported
# from its module by name.

from .base import Credentials, ProviderBackend, Rentable, http_transport, route
from .capabilities import Account, Capability, Delivery, LogSource, Market, Standing
from .cloud.core import CloudBackend
from .cloud.digitalocean import DigitalOceanBackend
from .cloud.hotaisle import HotAisleBackend
from .cloud.hyperstack import HyperstackBackend
from .cloud.lambdacloud import LambdaBackend
from .cloud.runpod import RunPodBackend
from .cloud.shadeform import ShadeformBackend
from .cloud.verda import VerdaBackend
from .cloud.vultr import VultrBackend
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
    "LambdaBackend",
    "LogSource",
    "Market",
    "ModalBackend",
    "ProviderBackend",
    "RunPodBackend",
    "Rentable",
    "ShadeformBackend",
    "Standing",
    "VastBackend",
    "VerdaBackend",
    "VultrBackend",
    "api_key",
    "http_transport",
    "route",
]
