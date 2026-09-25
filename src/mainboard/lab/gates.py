import abc
import os
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from patos import Registry

from ..experiments.data import Needs
from ..probe.gating import wait_for_idle

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from ..experiments.data import HfDataset, HfModel, RepoFile
    from .run import Run


class GateStatus(StrEnum):
    """A gate's judgement on a trial precondition."""

    PASSED = auto()
    BLOCKED = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """One gate's check outcome, `reason` empty when passed.

    status: `blocked` legitimately withholds the trial without counting as a failure, `failed`
        means the check itself broke.
    """

    status: GateStatus
    reason: str = ""


class Gate(Registry, abc.ABC):
    """Shared contract for a trial precondition: idle GPU, parity, offline mode, or data receipt.

    A new precondition kind is a new registered subclass, never a branch inside `runnable`. Each
    real check (a wait function, a comparison probe) is an injected callable with a hardware-free
    default, so tests never touch a GPU, the network or a cache.
    """

    @abc.abstractmethod
    def check(self, context: Run) -> GateVerdict:
        """Evaluate this gate's precondition right now."""

    def evaluate(self, probe: Callable[[], bool], *, blocked: str) -> GateVerdict:
        """The one rule every gate's `probe` is read by: `True` passes, `False` blocks with the
        `blocked` reason (unmet, never a failure), and a raise fails with its message."""
        try:
            passed = probe()
        except Exception as error:
            return GateVerdict(status=GateStatus.FAILED, reason=str(error))
        if passed:
            return GateVerdict(status=GateStatus.PASSED)
        return GateVerdict(status=GateStatus.BLOCKED, reason=blocked)


def is_parity_assumed(oracle: str, context: Run) -> bool:
    """The hardware-free default: parity holds unless a real comparison `probe` is injected."""
    return True


def is_offline_declared() -> bool:
    """Whether this process already declares itself offline through `HF_HUB_OFFLINE=1`."""
    return os.environ.get("HF_HUB_OFFLINE") == "1"


@dataclass(frozen=True, slots=True)
class Idle(Gate):
    """Blocks a trial until the GPU has been idle, never counting a busy GPU as a failure.

    seconds: how long to wait for an idle window before blocking.
    """

    seconds: float
    wait: Callable[..., bool] = wait_for_idle

    def check(self, context: Run) -> GateVerdict:
        return self.evaluate(
            lambda: self.wait(timeout=self.seconds),
            blocked=f"GPU still busy after {self.seconds}s",
        )


@dataclass(frozen=True, slots=True)
class Parity(Gate):
    """Blocks a trial until its behavior matches a named reference implementation.

    probe: reports whether parity with `oracle` holds, permissive by default.
    """

    oracle: str
    probe: Callable[[str, Run], bool] = is_parity_assumed

    def check(self, context: Run) -> GateVerdict:
        return self.evaluate(
            lambda: self.probe(self.oracle, context),
            blocked=f"parity with {self.oracle!r} not established",
        )


@dataclass(frozen=True, slots=True)
class Offline(Gate):
    """Blocks a trial unless the process declares itself offline."""

    probe: Callable[[], bool] = is_offline_declared

    def check(self, context: Run) -> GateVerdict:
        return self.evaluate(self.probe, blocked="offline mode is not declared")


@dataclass(frozen=True, slots=True)
class Receipt(Gate):
    """Blocks a trial until its declared dataset need is confirmed staged.

    dataset: the staging key (an `HfDataset`/`HfModel`/`RepoFile` `.key`) this trial needs.
    needs: the staging declarations checked against the keys `staged` reports present.
    """

    dataset: str
    needs: tuple[HfDataset | HfModel | RepoFile, ...] = ()
    staged: Callable[[], Collection[str]] = tuple

    def check(self, context: Run) -> GateVerdict:
        def staged_ok() -> bool:
            missing = Needs(self.needs).verify(self.staged())
            return not any(item.key == self.dataset for item in missing)

        return self.evaluate(staged_ok, blocked=f"{self.dataset} is not staged yet")
