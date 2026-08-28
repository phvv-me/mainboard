# WHAT A CLAIM LOADS ONCE, AND EXACTLY HOW LONG IT IS ALLOWED TO KEEP IT.
#
# A session fixture is the physical acquisition unit, and pytest's session scope is too wide for
# it. A checkpoint loaded by the first claim stays resident for every claim collected after it, so
# a run of eleven claims met the twelfth with 20.6 GB of a 24 GB card already gone and lost all 68
# of its trials at fixture setup, with 4.04 GB of the missing space reserved but unallocated. The
# arithmetic was untouched and every one of those claims reproduces alone.
#
# `scope="package"` is the obvious answer and is a documented NO-OP here: a claim folder carries no
# `__init__.py`, so pytest builds no Package node and package scope silently degrades to session.
# Giving every claim one turns the folders into packages, which is a layout decision a measurement
# subsystem has no business forcing.
#
# So the stage is keyed by the claim the universe already computes, and a trial belonging to
# another claim drops what the previous one held before it runs. That is the scope the conftest
# wanted, taken off the file tree instead of off pytest's node types.
#
# AND THE DROP IS CHECKED RATHER THAN ASSUMED. A consumer that can read its own resident bytes
# declares that probe, the stage records the floor before the first load and refuses when leaving
# a claim did not get back under it, naming the claim and both figures. A consumer that cannot
# read them declares nothing and the stage still drops on time.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Stage[Held]:
    """One claim's measure-once holdings, released the moment collection leaves that claim.

    Generic in what it holds, because a checkpoint, a warmed cache and a tokenizer are the same
    thing to this class and none of them is its business.

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

        The holdings are dropped before the probe reads, and the probe is read once, so a
        consumer whose release needs a cache flush does it inside its own probe rather than
        being handed a hook here to forget.
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
        """The object `key` names, made once for this claim and handed back after that.

        key: what distinguishes this holding from the claim's others, a checkpoint id or a shape.
        make: builds it, called at most once per claim.
        """
        if key not in self.held:
            self.held[key] = make()
        return self.held[key]
