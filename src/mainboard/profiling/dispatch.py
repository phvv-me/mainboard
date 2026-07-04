"""Per-architecture config dispatch: pick a value keyed on the current GPU.

Kernel configs are not portable. A Helion/Triton tile that fits Hopper's shared
memory (`sm_90`, tile_n=32) overflows Blackwell's smaller per-block SMEM (`sm_121`,
tile_n=16), so the right config has to be *selected* for the device the code is
running on rather than hand-pinned. :func:`arch_config` does exactly that: given an
arch-keyed table it returns the entry for this machine's GPU, falling back to
``default`` for an unlisted (or absent) device. The table is generic — the values are
whatever the caller stores (a Helion config, a tile-size tuple, a dataclass).
"""

from typing import TYPE_CHECKING

from ..gpu import GPU

if TYPE_CHECKING:
    from collections.abc import Mapping


def current_arch_key(*, device_index: int = 0) -> str | None:
    """The dispatch key (`GPU.arch_key`) of this machine's GPU, or ``None`` if none.

    device_index: which GPU from ``GPU.all()`` to read (default the first). The key
    is the stable, dot-free arch id — ``sm_90`` on NVIDIA — that :func:`arch_config`
    looks up in a table.
    """
    gpus = GPU.all()
    if not gpus:
        return None
    gpu = gpus[device_index] if device_index < len(gpus) else gpus[0]
    return gpu.arch_key


def arch_config[T](table: Mapping[str, T], *, default: T, device_index: int = 0) -> T:
    """Select the entry of ``table`` for this machine's GPU architecture.

    table: maps an arch key (``GPU.arch_key``, e.g. ``sm_90``) to the config for that
        generation — tile sizes, a Helion config, any per-arch value.
    default: returned when the current GPU's key is absent from ``table`` or no GPU is
        present, so callers always get a usable config.
    device_index: which GPU from ``GPU.all()`` to dispatch on.
    """
    key = current_arch_key(device_index=device_index)
    return table.get(key, default) if key is not None else default
