import io
from email.message import Message
from urllib.error import HTTPError

from gpuhunt import AcceleratorVendor, CatalogItem


def refusal(code: int, said: str) -> HTTPError:
    """The error a provider's API raises when it answers `code` with `said`."""
    return HTTPError("https://x", code, said, Message(), io.BytesIO(said.encode()))


def offer(*, instance_name: str = "x", location: str = "r", spot: bool = False) -> CatalogItem:
    """A one-card RTX 4090 catalog row, named and placed as asked."""
    return CatalogItem(
        provider="runpod",
        instance_name=instance_name,
        location=location,
        price=1.0,
        cpu=8,
        memory=32.0,
        gpu_count=1,
        gpu_name="RTX4090",
        gpu_memory=24.0,
        spot=spot,
        disk_size=None,
        gpu_vendor=AcceleratorVendor.NVIDIA,
    )
