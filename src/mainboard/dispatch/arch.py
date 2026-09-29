# A card's architecture as the compute capability CUDA builds for, so a request can name what its
# kernels need (`sm_120`, `hopper`) and the market answers with the cheapest card that runs them.
#
# The capability, not the marketing generation, is what decides whether a cubin loads. "Blackwell"
# is two instruction sets: the datacenter parts (B200, GB200: sm_100; B300, GB300: sm_103) carry
# tcgen05 and tensor memory, while the RTX and RTX PRO parts (RTX 5090, RTX PRO 4500 and 6000:
# sm_120; DGX Spark's GB10: sm_121) do not. Code built `sm_100a` loads on sm_100 alone, `sm_120a`
# on sm_120 alone, so a spelled capability matches exactly and a trailing `+` asks for at least it.
#
# Cards come from gpuhunt's table (the one its capability filter reads) with the few it lists
# wrong or not at all corrected below, since a wrong capability rents a card the kernels refuse.

import re

from patos import FrozenModel

from ..core.errors import MissionError

type Capability = tuple[int, int]

# Families by the capabilities they span, for a request that names a generation.
FAMILIES: dict[str, tuple[Capability, Capability]] = {
    "turing": ((7, 5), (7, 5)),
    "ampere": ((8, 0), (8, 7)),
    "ada": ((8, 9), (8, 9)),
    "hopper": ((9, 0), (9, 0)),
    "blackwell": ((10, 0), (12, 1)),
    "blackwell-dc": ((10, 0), (10, 3)),
    "blackwell-rtx": ((12, 0), (12, 1)),
}
# Where gpuhunt's table is wrong (B300 is sm_103) or silent, keyed as `card` spells a name.
_CORRECTED: dict[str, Capability] = {
    "B300": (10, 3),
    "GB300": (10, 3),
    "GB10": (12, 1),
    "RTXPRO5000": (12, 0),
    "RTXPRO4000": (12, 0),
    "RTXPRO2000": (12, 0),
    "A800": (8, 0),
    "H800": (9, 0),
    "H20": (9, 0),
}
_TOP: Capability = (99, 9)
_SPELLED = re.compile(r"(?:sm_?)?(\d{2,3})[af]?|(\d+)\.(\d)")
# The words a card is spelled with that no catalog keys on.
_NOISE = ("NVIDIA", "GEFORCE", " ", "_", "-")


def card(name: str) -> str:
    """`name` as the catalogs key a card (`RTX 4090`, `NVIDIA_GeForce_RTX_4090` -> `RTX4090`)."""
    spelled = name.upper()
    for noise in _NOISE:
        spelled = spelled.replace(noise, "")
    return spelled


def sm(capability: Capability | None) -> str:
    """`capability` as nvcc names it (`(12, 0)` -> `sm_120`), empty when unknown."""
    return f"sm_{capability[0]}{capability[1]}" if capability else ""


def capability(gpu_name: str) -> Capability | None:
    """The compute capability of the card `gpu_name` names, None for one no table knows."""
    import gpuhunt  # type: ignore[import-untyped]  # noqa: PLC0415  (a dozen catalogs, on use)

    keyed = card(gpu_name)
    if keyed in _CORRECTED:
        return _CORRECTED[keyed]
    return next(
        (
            tuple(info.compute_capability)
            for info in gpuhunt.KNOWN_NVIDIA_GPUS
            if card(info.name) == keyed
        ),
        None,
    )


class Arch(FrozenModel):
    """The compute capabilities a request accepts, `low` to `high` inclusive.

    spelled: what was asked (`sm_120`, `hopper`, `sm_90+`), for messages.
    """

    spelled: str
    low: Capability
    high: Capability

    def holds(self, found: Capability | None) -> bool:
        """Whether a card of capability `found` runs what this asks for; unknown never does."""
        return found is not None and self.low <= found <= self.high

    def vast(self) -> tuple[int, int]:
        """The span as Vast's `compute_cap` field counts it (`sm_120` -> 1200)."""
        return self.low[0] * 100 + self.low[1] * 10, self.high[0] * 100 + self.high[1] * 10


def arch(spelled: str) -> Arch:
    """`spelled` as the span it asks for: a family, `sm_120`, `120`, `12.0`, or any with `+`.

    A capability matches exactly, the way an `sm_XXa` cubin loads; `+` asks for it or newer.
    """
    text = spelled.strip().lower().replace(" ", "")
    at_least = text.endswith("+")
    text = text.removesuffix("+")
    if text in FAMILIES:
        low, high = FAMILIES[text]
    elif found := _SPELLED.fullmatch(text):
        digits, major, minor = found.groups()
        low = (int(digits[:-1]), int(digits[-1])) if digits else (int(major), int(minor))
        high = low
    else:
        raise MissionError(
            f"unknown architecture {spelled!r}; name a capability (`sm_120`, `sm_90+`, `8.9`) "
            f"or a family ({', '.join(FAMILIES)})"
        )
    return Arch(spelled=spelled, low=low, high=_TOP if at_least else high)
