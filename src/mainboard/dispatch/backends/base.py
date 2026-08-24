# The contract a provider backend implements, plus `route`, the typed replacement for an
# `if kind == ...` chain deciding whether a host stays on the existing ssh-family `Scheduler`
# path or resolves to a `ProviderBackend` registered by kind.
#
# The contract is split the way the providers themselves are split. `ProviderBackend` carries
# only the job lifecycle all of them truly have, launch a command, poll it, cancel it, and every
# other verb is a `Capability` a backend opts into by inheriting it. A caller therefore asks
# `isinstance(backend, LogSource)` before asking for a log, instead of calling and discovering
# mid-sweep that this provider never had one.

import abc
import os
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol
from urllib.request import urlopen

from patos import FrozenModel, Registry, Singleton
from pydantic import field_validator

from ...core.errors import MissionError
from ...core.project import Project

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from urllib.request import Request

    from ...context.plan import ExecutionPlan
    from ...costs.catalog import Offer
    from ..vocabulary import JobState, Resources

    class HttpResponse(Protocol):
        """The subset of `http.client.HTTPResponse` a transport callable must return."""

        status: int

        def read(self) -> bytes: ...

    type Transport = Callable[[Request], HttpResponse]

# Kinds the existing `Scheduler` path already dispatches; every other kind resolves a
# `ProviderBackend` registered under that same name.
# auto stays ssh-family, matching Scheduler.pick treating an unprobed kind as ssh.
_SSH_FAMILY_KINDS = frozenset({"auto", "local", "pbs", "slurm", "ssh"})

# Every provider call is bounded, so a provider that stops answering costs one slow row rather
# than a wedged command. Generous enough for a cold offer search, short enough that a survey of
# the whole fleet still finishes while someone is looking at it.
_TIMEOUT_S = 10.0

# Where a workspace keeps the provider credentials every refusal here tells someone to set.
_ENV_FILE = ".env"


class Credentials(Singleton):
    """The workspace `.env`, merged into this process's environment once, on first use.

    Every refusal a backend raises names a variable to set in the workspace `.env`, and nothing
    else in the tool ever read that file, so a survey called a provider unkeyed on a machine
    whose keys were sitting at the workspace root. This is the seam every backend crosses before
    it looks a key up, so honoring the promise here honors it for all of them at once.

    What the environment already holds always wins, so an exported key keeps its value and a
    stale line in the file can never shadow a deliberate one. The file is read as data and never
    as shell, meaning a line is `NAME=value`, blanks and `#` comments are skipped, one matching
    pair of surrounding quotes comes off, and nothing is expanded, substituted or executed. No
    value is ever logged or returned, only the names that were defined.

    One shared instance, since merging a file into the environment is a thing that happens once
    per process however many backends ask for it.
    """

    def __init__(self) -> None:
        self.project = Project()
        self.loaded = False
        self.lock = Lock()

    def load(self) -> tuple[str, ...]:
        """Define what the workspace `.env` declares and this environment lacks, by name.

        Empty on every call after the first, on a machine standing outside a workspace, and in a
        workspace that keeps no `.env`, which are three ways of saying this call added nothing. A
        workspace is found the way every other verb finds one, by walking up from the current
        directory to the manifest.

        The whole merge is one critical section rather than a flag flipped up front, because a
        compute survey probes every provider at once. Marking the file read before it has been
        read would let the second provider find nothing while the first is still merging, which
        is a keyed account reported as unkeyed for no reason but timing (seen live 2026-08-19).
        """
        with self.lock:
            if self.loaded:
                return ()
            self.loaded = True
            return self.merged()

    def merged(self) -> tuple[str, ...]:
        """Read the workspace `.env` and define what it declares, returning only the names."""
        try:
            text = (self.project.find_root(Path.cwd()) / _ENV_FILE).read_text(encoding="utf-8")
        except OSError:
            return ()
        defined: list[str] = []
        for line in text.splitlines():
            entry = line.strip().removeprefix("export ").strip()
            if not entry or entry.startswith("#"):
                continue
            name, assigned, value = entry.partition("=")
            name = name.strip()
            if not name or not assigned or name in os.environ:
                continue
            os.environ[name] = self.unquoted(value.strip())
            defined.append(name)
        return tuple(defined)

    @staticmethod
    def unquoted(value: str) -> str:
        """`value` with one matching pair of surrounding quotes off, the way a `.env` writes it."""
        paired = len(value) > 1 and value[0] == value[-1] and value[0] in "\"'"
        return value[1:-1] if paired else value


class Standing(FrozenModel):
    """What a provider cheaply says about itself: whether we may use it, and at what price.

    The account-side answer to the job-side lifecycle, and the only thing a compute survey asks
    a provider for. It never carries a credential, only whether one was found.

    keyed: whether this provider's credentials are present on this machine.
    credit_usd: the balance the provider reports, None when it exposes none, which is the
        honest answer for a provider that publishes spend and never a remaining balance.
    usd_hr: the cheapest live rate a sample search found, None when no price is a cheap
        question here.
    note: the one human line the row carries, the variable to set when there is no key, or why
        there is no credit figure. Never a secret.
    """

    keyed: bool = False
    credit_usd: float | None = None
    usd_hr: float | None = None
    note: str = ""

    @field_validator("credit_usd", "usd_hr")
    @classmethod
    def rounded(cls, value: float | None) -> float | None:
        """Money at the precision money has, so a row reads as a rate and not a float artifact.

        Four places rather than two, since an hourly GPU rate is quoted in fractions of a cent
        and rounding one to cents would make two real offers look identically priced.
        """
        return None if value is None else round(value, 4)


class Capability:
    """One optional half of the backend contract, opted into by inheriting it.

    A backend that can do the thing implements the contract and is found by `isinstance`. A
    backend that cannot simply does not inherit it, and names it in `ProviderBackend.lacks`
    with the line a refusal should carry, so the gap is a typed absence a caller can see before
    it calls rather than a `MissionError` raised from inside a verb that was never real.

    The root itself declares nothing. It exists so `lacks` can be keyed by a contract rather
    than by a bare name, which is what keeps a declared gap and the class it names in step.
    """


class Account(Capability, abc.ABC):
    """A provider that answers for its own account rather than for one run."""

    @abc.abstractmethod
    def standing(self) -> Standing:
        """Whether this provider is usable from here, priced and credited where that is cheap.

        The one question a compute survey asks a provider, so a new backend joins that listing by
        implementing this rather than by being named there. It reads credentials without ever
        revealing them, and only reaches the network once it has found some, which is what keeps
        an unconfigured provider free to list.
        """


class Delivery(Capability):
    """A provider that can bring a finished run's artifacts back to this machine."""

    @abc.abstractmethod
    def deliver(self, handle: str, *, path: str) -> None:
        """Fetch `handle`'s output at `path` back to the local filesystem."""


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

        gpu_name: the provider's GPU name to narrow to, empty for the whole market.
        gpus: the GPU count per machine, 0 for any.
        limit: how many offers to bring back, 0 for the backend's own page size.
        """


class ProviderBackend(Registry, abc.ABC):
    """Registry root for non-ssh provider backends, one concrete class per `HostProfile.kind`.

    Unlike `Scheduler`, no method takes a `remote`/`root`: a provider backend owns its own
    transport (an HTTP session, an SDK client) end to end instead of running commands over an
    ssh connection into a synced workspace.

    What lives here is the lifecycle every provider truly has, launch a command, poll it, cancel
    it. Logs, artifact delivery, account standing and market pricing are `Capability` contracts a
    backend inherits only when it can honor them, so nothing carries a method it would only ever
    refuse. A new backend joins by subclassing this, implementing three methods, and adding
    whichever capabilities it actually has; `route` is the lookup that finds it, so no caller
    hand-rolls an `if kind == "modal": ... elif kind == "vast": ...` chain.
    """

    # What this backend cannot do, and the line to print instead, keyed by the contract it does
    # not inherit. `{handle}` and `{path}` are filled in where the refusal is raised.
    lacks: ClassVar[Mapping[type[Capability], str]] = {}

    @abc.abstractmethod
    def cancel(self, handle: str) -> None:
        """Cancel `handle` on the provider."""

    @abc.abstractmethod
    def state(self, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict."""

    @abc.abstractmethod
    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        """Launch `command` under `resources`; return the provider's opaque handle id."""

    def refusal(self, capability: type[Capability], **facts: str) -> str:
        """Why this backend cannot answer `capability`, and what to do about it instead.

        The line a discovery site raises once `isinstance` has said no. A backend that declared
        the gap in `lacks` supplies its own advice, since only it knows where its output really
        lives, and `facts` fills the `{handle}` and `{path}` the advice names. One that declared
        nothing gets a plain statement of the gap rather than a traceback.

        capability: the contract this backend does not implement.
        facts: the run's details the advice may name, `handle` and `path`.
        """
        advice = self.lacks.get(capability)
        if advice is None:
            return f"the {self.name} backend does not implement {capability.__name__}"
        return advice.format(**facts)


def http_transport(request: Request) -> HttpResponse:
    """Send `request` over urllib and return its response, the default every REST backend takes.

    The one audited url open in the package. Each backend builds its own `Request` from a
    constant https root of its own, and a test swaps this callable out wholesale, so no unvetted
    scheme ever reaches urllib through here. The deadline is the package's own rather than
    urllib's (which has none), so no caller can be left waiting on a provider forever.
    """
    return urlopen(request, timeout=_TIMEOUT_S)  # ruff:ignore[suspicious-url-open-usage]  reason=the package's single audited seam, every caller builds its Request from a constant https root and tests inject a double since=2026-08-18


def require_budget(resources: Resources) -> None:
    """Refuse an unbounded-cost submission before any network call.

    Every provider backend runs on someone's metered infrastructure, so a caller that forgot
    to set `resources.max_usd` gets a plain refusal here instead of an open-ended bill.

    resources: the resource request a submit call is about to dispatch under.
    """
    if not resources.max_usd:
        raise MissionError("provider dispatch needs an explicit max-usd budget")


def route(kind: str) -> Literal["ssh-family"] | type[ProviderBackend]:
    """Whether `kind` runs the ssh-family `Scheduler` path or a `ProviderBackend` class.

    `ssh`, `pbs`, `slurm` and `local` stay on the existing scheduler dispatch. Any other kind
    resolves a `ProviderBackend` registered under it, raising a `MissionError` naming the known
    provider kinds when none matches.

    The kind rather than the profile it came from, since a dispatched run is rebuilt from what
    the dispatch cache recorded at submit time and there is no profile left to pass by then.

    kind: the scheduler or provider kind selecting the path.
    """
    if kind in _SSH_FAMILY_KINDS:
        return "ssh-family"
    try:
        return ProviderBackend.find(kind)
    except KeyError:
        raise MissionError(
            f"no provider backend for kind {kind!r}; known kinds are {ProviderBackend.names()}"
        ) from None
