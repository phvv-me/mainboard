# Per-architecture config dispatch: pick a value keyed on a given GPU.

from collections.abc import Mapping

from .protocols import DeviceProbe


def arch_config[T](table: Mapping[str, T], *, default: T, gpu: DeviceProbe | None = None) -> T:
    """Select the entry of ``table`` for ``gpu``'s architecture.

    table: maps an arch key (``DeviceProbe.arch_key``, e.g. ``sm_90``) to the config for
        that generation — tile sizes, a Helion config, any per-arch value.
    default: returned when ``gpu`` is ``None`` or its key is absent from ``table``, so
        callers always get a usable config.
    gpu: the device to dispatch on, or ``None`` on a CPU-only host.
    """
    if gpu is None:
        return default
    return table.get(gpu.arch_key, default)
