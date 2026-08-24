# The survey behind `mainboard compute`: every place this workspace can run work, in one list.
# This machine, the hosts the manifest declares, and every registered provider backend, each
# answered by one bounded probe. A host that will not answer and a provider with no key are row
# states here, never failures, so the whole fleet still lists when part of it is down.

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING

from patos import FrozenModel

from .core.errors import MissionError
from .dispatch.backends.base import Account, ProviderBackend, route
from .dispatch.transport import HostUnreachable, SshTransport
from .probe.snapshot import HostFacts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .board import Board
    from .dispatch.onboard import HostSetup
    from .manifest.schema.host import HostProfile

# One bounded ssh round trip per host, under a policy tightened for a survey rather than for a
# job. The two numbers bound different failures: `connect_timeout` is what a host that never
# answers costs, while the derived deadline is the backstop for one that connects and then goes
# quiet, which is the only thing that bounds a stalled handshake. Ten seconds is deliberately
# generous for the connect, since a host behind a ProxyJump measured here answers in under a
# second most times and in six seconds sometimes, and calling that host down would be worse than
# waiting for it. The deadline still lands far under the dispatch default's, so the whole survey
# finishes in the time one stalled host takes rather than in a minute.
_PROBE_SSH = SshTransport(connect_timeout=10.0, server_alive_interval=2.0, server_alive_count=1)

# The `kind` a provider row carries. Providers have no scheduler, so naming the route keeps the
# column meaning one thing (how this path is reached) instead of repeating the provider's name.
_PROVIDER = "provider"

# What a live probe is allowed to fail with before it becomes a row state rather than an error.
# `OSError` is every network fault urllib raises (an `HTTPError` 4xx included), `MissionError` is
# a backend refusing before the network, and `ValueError` is a provider answering something
# unreadable.
_PROBE_FAULTS = (HostUnreachable, MissionError, OSError, ValueError)


class Access(StrEnum):
    """How usable one compute path is right now."""

    HERE = auto()
    READY = auto()
    REACHABLE = auto()
    UNREACHABLE = auto()
    KEYED = auto()
    UNKEYED = auto()


class ComputePath(FrozenModel):
    """One place this workspace can run work, and what reaching it costs right now.

    name: the host alias or the provider's registered name, `local` for this machine.
    kind: how the path is reached, a scheduler kind for a host and `provider` for a backend.
    access: how usable the path is right now.
    detail: the one human line behind `access`, the hardware for a machine, the refusal for a
        host that would not answer, the variable to set for a provider with no key.
    usd_hr: a live cheapest-offer sample, None where no price is a cheap question.
    credit_usd: the balance the provider reports, None where it exposes none.
    """

    name: str
    kind: str
    access: Access
    detail: str = ""
    usd_hr: float | None = None
    credit_usd: float | None = None


def summary(facts: HostFacts) -> str:
    """One line naming what a machine has, its GPUs by model then its memory.

    facts: the machine's probed hardware snapshot.
    """
    counted = Counter(gpu.name for gpu in facts.gpus)
    parts = [f"{count}x {name}" for name, count in counted.items()]
    parts.append(f"{facts.memory_total_bytes / 1e9:.0f} GB RAM")
    return ", ".join(parts)


def reachable(host: str, ssh: SshTransport = _PROBE_SSH) -> str:
    """Why `host` cannot be reached right now, empty when one bounded ssh round trip lands.

    The cheapest question worth asking a host, one `true` over the connection the dispatch
    subsystem would use anyway, so a survey never pays for the workspace probe that `facts` runs.

    host: the ssh alias to try.
    ssh: the bounded transport policy the probe rides, a short-deadline one by default.
    """
    try:
        ssh.warm(host)
    except (HostUnreachable, ConnectionError, RuntimeError) as refusal:
        return str(refusal)
    return ""


class Survey:
    """Every compute path this workspace can reach, probed once, together, and bounded.

    The rows come in the order the question is usually asked: what is under this desk, what has
    already been set up elsewhere, and what can be rented. Each row is one probe, and a probe
    that fails becomes that row's state, so a dead host or an unconfigured provider never costs
    the rest of the answer. The probes run in one small pool, so the survey takes as long as its
    slowest single probe rather than the sum of all of them.

    Every network touch is injected, so a test drives the whole survey without a machine, a host
    or a provider anywhere in reach.
    """

    def __init__(
        self,
        board: Board,
        *,
        facts: Callable[[], HostFacts] = HostFacts.collected,
        reach: Callable[[str], str] = reachable,
        providers: Sequence[ProviderBackend] | None = None,
    ) -> None:
        """board: the workspace whose declared hosts and onboarding records the survey reads.

        facts: probes this machine's hardware.
        reach: answers why a host cannot be reached, empty when it can.
        providers: the provider backends to ask, every registered one when None.
        """
        self.board = board
        self.facts = facts
        self.reach = reach
        self.providers = (
            [backend() for backend in ProviderBackend.implementations()]
            if providers is None
            else list(providers)
        )

    def here(self) -> ComputePath:
        """This machine, from its own probed facts."""
        return ComputePath(
            name="local", kind="local", access=Access.HERE, detail=summary(self.facts())
        )

    def machine(self, alias: str, profile: HostProfile, setup: HostSetup | None) -> ComputePath:
        """One declared host: whether it answers, and what onboarding already recorded of it.

        A host that answers but was never set up is reachable rather than ready, which is the
        difference between a machine that can take a job and one that still needs `setup`. The
        hardware line comes from the onboarding record rather than from the host, so a ready row
        describes real hardware without a second round trip.

        alias: the declared host name.
        profile: that host's resolved profile, whose kind names the scheduler.
        setup: what onboarding recorded for the alias, None when it was never set up.
        """
        refusal = self.reach(alias)
        if refusal:
            return ComputePath(
                name=alias, kind=profile.kind, access=Access.UNREACHABLE, detail=refusal
            )
        if setup is None:
            return ComputePath(
                name=alias, kind=profile.kind, access=Access.REACHABLE, detail="never set up"
            )
        detail = summary(setup.hardware) if setup.hardware else f"{setup.env}, hardware unrecorded"
        return ComputePath(name=alias, kind=profile.kind, access=Access.READY, detail=detail)

    def onboarded(self) -> dict[str, HostSetup]:
        """What onboarding recorded for each alias, read from the dispatch cache, keyed by alias.

        Its own verb because reading it is thread-bound: the cache is one SQLite connection and
        only the thread that opened it may use it. A caller that will hand this survey to a pool
        reads the records first, on the thread that owns the cache, and passes them to `paths`.
        """
        return {setup.host: setup for setup in self.board.dispatcher.cache.hosts()}

    def paths(self, setups: Mapping[str, HostSetup] | None = None) -> list[ComputePath]:
        """Every compute path, probed in parallel, this machine first.

        The onboarding records are read on this thread and handed to each host probe, since the
        dispatch cache is one SQLite connection and the pool below is not its owner. A caller
        that is itself inside a pool has already read them on the owning thread and passes them
        in, which is the same discipline one level up. A declared host whose kind routes to a
        provider is left to that provider's own row, so a rented machine is listed once rather
        than probed as if it were an ssh box.

        setups: the onboarding records by alias, read from the dispatch cache here when None.
        """
        recorded = self.onboarded() if setups is None else setups
        probes: list[Callable[[], ComputePath]] = [self.here]
        probes.extend(
            partial(self.machine, alias, profile, recorded.get(alias))
            for alias, profile in sorted(self.board.manifest.profiles().items())
            if route(profile.kind) == "ssh-family"
        )
        probes.extend(partial(self.provider, backend) for backend in self.providers)
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            return list(pool.map(lambda probe: probe(), probes))

    def provider(self, backend: ProviderBackend) -> ComputePath:
        """One provider backend: whether its credentials are here, and what it says they buy.

        Answering for an account is a capability rather than part of every backend, so one that
        never had the notion is listed with what it lacks instead of being asked and made to
        raise. That is the whole point of discovering the contract here: a survey stays a listing.

        backend: the registered backend to ask, which answers for its own account.
        """
        if not isinstance(backend, Account):
            return ComputePath(
                name=backend.name,
                kind=_PROVIDER,
                access=Access.UNKEYED,
                detail=backend.refusal(Account),
            )
        try:
            standing = backend.standing()
        except _PROBE_FAULTS as fault:
            return ComputePath(
                name=backend.name, kind=_PROVIDER, access=Access.UNREACHABLE, detail=str(fault)
            )
        return ComputePath(
            name=backend.name,
            kind=_PROVIDER,
            access=Access.KEYED if standing.keyed else Access.UNKEYED,
            detail=standing.note,
            usd_hr=standing.usd_hr,
            credit_usd=standing.credit_usd,
        )
