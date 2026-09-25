# What a claim loads once, and exactly how long it may keep it.
#
# pytest's session scope is too wide for a physical acquisition: a checkpoint the first claim
# loaded stayed resident for every claim after it, so a run of eleven claims met the twelfth with
# 20.6 GB of a 24 GB card gone (4.04 GB reserved but unallocated) and lost all 68 of its trials at
# fixture setup, though each claim reproduces alone. `scope="package"` silently degrades to session
# here, since a claim folder has no `__init__.py` and forcing one is not this subsystem's call. So
# the stage is keyed by the claim the universe computes, and a trial of another claim drops what
# the previous one held. A consumer that can read its resident bytes declares that probe, and the
# drop is checked against the floor recorded before the first load.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Stage[Held]:
    """One claim's measure-once holdings, released the moment collection leaves that claim.

    claim: the node these holdings belong to, named in a refusal.
    resident: reads the bytes currently held, so the release can be checked; None skips the check.
    """

    def __init__(self, claim: str, *, resident: Callable[[], int] | None = None) -> None:
        self.claim = claim
        self.resident = resident
        self.floor = resident() if resident else 0
        self.held: dict[str, Held] = {}

    def drop(self) -> None:
        """Release everything this claim held, then refuse if the space did not come back.

        The probe is read once, after the release, so a release that needs a cache flush does it
        inside the probe.
        """
        self.held.clear()
        if self.resident is None:
            return
        left = self.resident()
        if left > self.floor:
            raise RuntimeError(
                f"{self.claim or 'the universe root'} did not release what it held: "
                f"{left} bytes resident against the {self.floor} it opened with, so every claim "
                "collected after this one measures a smaller machine than it asked for"
            )

    def kept(self, key: str, make: Callable[[], Held]) -> Held:
        """The holding `key` names, made at most once per claim.

        key: what distinguishes this holding from the claim's others, a checkpoint id or a shape.
        """
        if key not in self.held:
            self.held[key] = make()
        return self.held[key]
