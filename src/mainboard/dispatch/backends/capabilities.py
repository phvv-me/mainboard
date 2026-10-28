# The optional halves of the provider backend contract. Every verb beyond the lifecycle every
# provider has (launch, poll, cancel) is a `Capability` a backend opts into by inheriting it, so a
# caller asks `isinstance(backend, LogSource)` before asking for a log instead of discovering
# mid-sweep that the provider never had one.

import abc
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import field_validator

if TYPE_CHECKING:
    from ...costs.catalog import Offer


class Standing(FrozenModel):
    """What a provider cheaply says about itself, the only thing a compute survey asks it.

    keyed: whether this provider's credentials are present here; never the credential itself.
    credit_usd: the balance the provider reports, None when it publishes spend but no balance.
    usd_hr: the cheapest live rate a sample search found, None when no price is a cheap question.
    note: the row's one human line, the variable to set or why there is no credit. Never a secret.
    """

    keyed: bool = False
    credit_usd: float | None = None
    usd_hr: float | None = None
    note: str = ""

    @field_validator("credit_usd", "usd_hr")
    @classmethod
    def rounded(cls, value: float | None) -> float | None:
        """Money to four places, since hourly GPU rates differ by fractions of a cent."""
        return None if value is None else round(value, 4)


class Capability:
    """One optional half of the backend contract, opted into by inheriting it.

    A backend lacking one names it in `ProviderBackend.lacks` with the refusal line, so the gap
    is a typed absence seen before calling. The root declares nothing; it keys `lacks` by
    contract rather than by a bare name, keeping a declared gap and its class in step.
    """


class Account(Capability, abc.ABC):
    """A provider that answers for its own account rather than for one run."""

    @abc.abstractmethod
    def standing(self) -> Standing:
        """Whether this provider is usable from here, priced and credited where that is cheap.

        A new backend joins the compute survey by implementing this. It never reveals a
        credential and reaches the network only once it found one, so an unconfigured provider
        is free to list.
        """


class Delivery(Capability):
    """A provider that can bring a finished run's artifacts back to this machine."""

    @abc.abstractmethod
    def deliver(self, handle: str, *, path: str) -> None:
        """Fetch `handle`'s output at `path` back to the local filesystem."""


class Rented(FrozenModel):
    """One machine a provider says this account is renting right now.

    handle: the provider's id for the rental, which ends it.
    label: the label it was created under, `mainboard-<id>` for one this tool rented.
    gpu: the cards it carries, as the provider names them.
    usd_hr: what it bills per hour, None when the listing carries no rate.
    """

    handle: str
    label: str = ""
    gpu: str = ""
    status: str = ""
    usd_hr: float | None = None


class Inventory(Capability, abc.ABC):
    """A provider that lists every machine the account is renting, however it was rented."""

    @abc.abstractmethod
    def rentals(self) -> list[Rented]:
        """Every live rental on the account, as the provider reports it.

        Only the provider's listing catches a machine rented from another checkout, by hand, or
        by a dispatch whose record was lost, all of which bill the same.
        """


class LogSource(Capability):
    """A provider that keeps a run's captured output and will hand it back."""

    @abc.abstractmethod
    def logs(self, handle: str) -> str:
        """`handle`'s captured log so far."""


class Market(Capability):
    """A provider that quotes a live rentable market, priced offer by offer."""

    @abc.abstractmethod
    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        """Live offers as catalog rows, the authed refresh of an imported price feed.

        gpu_name: the provider's GPU name, empty for the whole market.
        gpus: the GPU count per machine, 0 for any.
        limit: how many offers to bring back, 0 for the backend's own page size.
        """
