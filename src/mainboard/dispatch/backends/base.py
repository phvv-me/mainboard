# The transport-free contract a provider backend implements, plus `route`, the typed
# replacement for an `if kind == ...` chain deciding whether a host stays on the existing
# ssh-family `Scheduler` path or resolves to a `ProviderBackend` registered by kind.

import abc
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from urllib.request import urlopen

from patos import Registry

from ...core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from urllib.request import Request

    from ...context.plan import ExecutionPlan
    from ...manifest.schema.host import HostProfile
    from ..schedulers.base import JobState, Resources

    class HttpResponse(Protocol):
        """The subset of `http.client.HTTPResponse` a transport callable must return."""

        status: int

        def read(self) -> bytes: ...

    type Transport = Callable[[Request], HttpResponse]

# Kinds the existing `Scheduler` path already dispatches; every other kind resolves a
# `ProviderBackend` registered under that same name.
# auto stays ssh-family, matching Scheduler.pick treating an unprobed kind as ssh.
_SSH_FAMILY_KINDS = frozenset({"auto", "local", "pbs", "slurm", "ssh"})


@runtime_checkable
class Backend(Protocol):
    """A provider backend: submit a command, poll it, fetch its output, no transport in sight.

    Unlike `Scheduler`, no method takes a `remote`/`root`: a provider backend owns its own
    transport (an HTTP session, an SDK client) end to end instead of running commands over an
    ssh connection into a synced workspace.
    """

    def cancel(self, handle: str) -> None:
        """Cancel `handle` on the provider."""

    def deliver(self, handle: str, *, path: str) -> None:
        """Fetch `handle`'s output at `path` back to the local filesystem."""

    def logs(self, handle: str) -> str:
        """`handle`'s captured log so far."""

    def state(self, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict."""

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        """Launch `command` under `resources`; return the provider's opaque handle id."""


class ProviderBackend(Registry, abc.ABC):
    """Registry root for non-ssh provider backends, one concrete class per `HostProfile.kind`.

    A concrete backend (`ModalBackend`, `HpcAiBackend`) enrolls here and implements every
    `Backend` method; `route` is the lookup a caller uses instead of hand-rolling an
    `if kind == "modal": ... elif kind == "hpc-ai": ...` chain.
    """

    @abc.abstractmethod
    def cancel(self, handle: str) -> None:
        """Cancel `handle` on the provider."""

    @abc.abstractmethod
    def deliver(self, handle: str, *, path: str) -> None:
        """Fetch `handle`'s output at `path` back to the local filesystem."""

    @abc.abstractmethod
    def logs(self, handle: str) -> str:
        """`handle`'s captured log so far."""

    @abc.abstractmethod
    def state(self, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict."""

    @abc.abstractmethod
    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        """Launch `command` under `resources`; return the provider's opaque handle id."""


def http_transport(request: Request) -> HttpResponse:
    """Send `request` over urllib and return its response, the default every REST backend takes.

    The one audited url open in the package. Each backend builds its own `Request` from a
    constant https root of its own, and a test swaps this callable out wholesale, so no unvetted
    scheme ever reaches urllib through here.
    """
    return urlopen(request)  # ruff:ignore[suspicious-url-open-usage]  reason=the package's single audited seam, every caller builds its Request from a constant https root and tests inject a double since=2026-08-18


def require_budget(resources: Resources) -> None:
    """Refuse an unbounded-cost submission before any network call.

    Every provider backend runs on someone's metered infrastructure, so a caller that forgot
    to set `resources.max_usd` gets a plain refusal here instead of an open-ended bill.

    resources: the resource request a submit call is about to dispatch under.
    """
    if not resources.max_usd:
        raise MissionError("provider dispatch needs an explicit max-usd budget")


def route(profile: HostProfile) -> Literal["ssh-family"] | type[ProviderBackend]:
    """Whether `profile` runs the ssh-family `Scheduler` path or a `ProviderBackend` class.

    `ssh`, `pbs`, `slurm` and `local` stay on the existing scheduler dispatch. Any other kind
    resolves a `ProviderBackend` registered under `profile.kind`, raising a `MissionError`
    naming the known provider kinds when none matches.

    profile: the resolved host profile whose `kind` selects the path.
    """
    if profile.kind in _SSH_FAMILY_KINDS:
        return "ssh-family"
    try:
        return ProviderBackend.find(profile.kind)
    except KeyError:
        raise MissionError(
            f"no provider backend for kind {profile.kind!r}; known kinds are "
            f"{ProviderBackend.names()}"
        ) from None
