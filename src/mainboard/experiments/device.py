"""A short, stable tag for the accelerator an experiment ran on, from the machine probe."""

import functools
import re

from ..probe.machine import Machine

# Product names as the probe reports them, shortened to the slugs rows are partitioned by.
_ALIASES: dict[str, str] = {
    "NVIDIA GeForce RTX 4090": "RTX_4090",
    "NVIDIA GeForce RTX 5080": "RTX_5080",
    "NVIDIA GeForce RTX 5090": "RTX_5090",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA H100 PCIe": "H100_PCIe",
    "NVIDIA H100": "H100",
    "NVIDIA H200": "H200",
    "NVIDIA GH200 120GB": "GH200_120GB",
    "NVIDIA GH200 480GB": "GH200_480GB",
    "NVIDIA GH200 96GB": "GH200_96GB",
    "NVIDIA GH200": "GH200",
    "NVIDIA GB10": "GB10",
    "NVIDIA A100-SXM4-80GB": "A100_80GB",
    "NVIDIA A100-PCIE-40GB": "A100_40GB",
    "NVIDIA A100": "A100",
    "NVIDIA L40S": "L40S",
}


def _shorten(label: str) -> str:
    if label in _ALIASES:
        return _ALIASES[label]
    return re.sub(r"[^A-Za-z0-9]+", "_", label.replace("NVIDIA ", "")).strip("_")


@functools.cache
def device_tag(index: int = 0) -> str:
    """`{short_name}_CC{major}.{minor}` for the GPU at `index`, its bare short name for an
    accelerator with no CUDA architecture (an Apple GPU), or `CPU` without one.

    index: accelerator ordinal as the probe lists it.
    """
    gpus = Machine().gpus
    if index >= len(gpus):
        return "CPU"
    gpu = gpus[index]
    architecture = getattr(gpu, "cuda_architecture", None)
    if architecture is None:
        return _shorten(gpu.label)
    return f"{_shorten(gpu.label)}_CC{architecture.major}.{architecture.minor}"


def device_name(index: int = 0) -> str:
    """Return the probe's full product name for the GPU at `index`, or `cpu` without one."""
    gpus = Machine().gpus
    return gpus[index].label if index < len(gpus) else "cpu"
