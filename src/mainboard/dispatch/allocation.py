"""Durable provider creation, before a handle or an SSH endpoint exists."""

from patos import FrozenModel, Runtime

from ..core.errors import MissionError
from . import vocabulary
from .lease import Lease
from .shared import logger
from .state.cache import Cache, RunRecord


class Allocation(FrozenModel):
    """One reserved request and the mandatory provider creation checkpoints."""

    cache: Runtime[Cache]
    record: RunRecord

    @property
    def label(self) -> str:
        """The unique label retained locally and sent with the provider's create request."""
        return self.record.creation

    def begin(self, *, lease: Lease | None = None) -> None:
        """Persist the uncertain API boundary before sending a create request."""
        self.cache.leave_prepared(self.record, vocabulary.SUBMITTING, lease=lease)

    def created(self, handle: str) -> str:
        """Attach the returned handle atomically, before waiting or running another API call."""
        if not handle or handle == "None":
            raise MissionError(
                f"provider returned no handle for {self.label}; reconcile its label"
            )
        logger.info("provider creation %s -> %s", self.label, handle)
        self.cache.bind(self.record, handle)
        return handle

    def interrupted(self) -> None:
        """Only a request that never crossed the API boundary is known not to have created."""
        current = self.cache.creation(self.label, self.record.target)
        if current.verdict in vocabulary.TERMINAL:
            return
        if current.verdict == vocabulary.PREPARED:
            self.cache.leave_prepared(current, vocabulary.FAILED)
            return
        if current.verdict != vocabulary.SUBMITTING:
            if current.evidence == "not_started":
                self.cache.resolve(current, vocabulary.FAILED, None, vocabulary.FAILED)
            return
        logger.error(
            "creation %s has no confirmed provider handle; reconcile this exact label before "
            "retrying, because the request may have allocated a billable instance",
            self.label,
        )
