# The survey behind `mainboard compute`: every place this workspace can run work, in one list.
# This machine, the hosts the manifest declares and the machines it is holding, every registered
# provider backend, and every machine a provider says this account is renting right now, each
# answered by one bounded probe. A host that will not answer and a provider with no key are row
# states here, never failures, so the whole fleet still lists when part of it is down.

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import Field

from .core.errors import MissionError
from .dispatch.backends.base import Account, Credentials, Inventory, ProviderBackend, route
from .dispatch.transport import HostUnreachable, SshTransport
from .manifest.held import Holdings
from .probe.snapshot import HostFacts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .board import Board
    from .dispatch.backends.base import Rented
    from .dispatch.onboard import HostSetup
    from .manifest.held import Held
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

# The `kind` a live rental carries: a machine a provider bills for, reached through its provider.
_RENTAL = "rental"

# What a live probe is allowed to fail with before it becomes a row state rather than an error.
# `OSError` is every network fault urllib raises (an `HTTPError` 4xx included), `MissionError` is
# a backend refusing before the network, and `ValueError` is a provider answering something
# unreadable.
_PROBE_FAULTS = (HostUnreachable, MissionError, OSError, ValueError)


class Access(StrEnum):
    """How usable one compute path is right now."""

    HERE = auto()
    PROVISIONED = auto()
    REACHABLE = auto()
    UNREACHABLE = auto()
    KEYED = auto()
    UNKEYED = auto()
    RENTED = auto()


class ComputePath(FrozenModel):
    """One place this workspace can run work, and what reaching it costs right now.

    name: the host alias or the provider's registered name, `local` for this machine.
    kind: how the path is reached, a scheduler kind for a host and `provider` for a backend.
    access: how usable the path is right now.
    detail: the one human line behind `access`, the hardware for a machine, the refusal for a
        host that would not answer, the variable to set for a provider with no key.
    usd_hr: a live cheapest-offer sample, None where no price is a cheap question.
    credit_usd: the balance the provider reports, None where it exposes none.
    observed_at: UTC completion time of this survey observation, not a readiness lease.
    cached_at: onboarding time of retained host facts; empty means their age is unknown.
    """

    name: str
    kind: str
    access: Access
    detail: str = ""
    usd_hr: float | None = None
    credit_usd: float | None = None
    observed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    cached_at: str = ""


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

    An echo marker works in POSIX shells, cmd, and PowerShell without requiring a provisioned
    environment. This proves a remote command answered, not that a GPU job can run.

    host: the ssh alias to try.
    ssh: the bounded transport policy the probe rides, a short-deadline one by default.
    """
    try:
        reply = ssh.run(
            ("ssh", *ssh.options, ssh.destination(host), "echo", "mainboard-reachable"),
            host,
            operation="survey",
        )
    except (HostUnreachable, ConnectionError, RuntimeError) as refusal:
        return str(refusal)
    if "mainboard-reachable" not in (line.strip() for line in reply.splitlines()):
        return "SSH command returned without the expected survey marker; inspect the remote shell"
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
            name="local",
            kind="local",
            access=Access.HERE,
            detail=f"{summary(self.facts())}; live hardware, GPU availability not checked",
        )

    def machine(self, alias: str, profile: HostProfile, setup: HostSetup | None) -> ComputePath:
        """One declared host: whether it answers, and what onboarding already recorded of it.

        Provisioned means an onboarding record exists, not that its environment or scheduler
        still works. Retained hardware is explicitly cached and may be stale. In particular,
        PBS/Slurm login hardware says nothing about a future compute allocation.
        A profile's vars.status-note may replace generic next-step advice, never observed state.

        alias: the declared host name.
        profile: that host's resolved profile, whose kind names the scheduler.
        setup: what onboarding recorded for the alias, None when it was never set up.
        """
        refusal = self.reach(alias)
        note = profile.vars.get("status-note", "")
        cached_at = setup.onboarded_at if setup else ""
        cached = ""
        if setup is not None:
            hardware = summary(setup.hardware) if setup.hardware else "hardware unrecorded"
            cached = f"cached {setup.env}: {hardware} (observed {cached_at or 'time unknown'})"
        if refusal:
            access = Access.UNREACHABLE
            detail = f"{refusal}; {cached}" if cached else refusal
        elif setup is None:
            access = Access.REACHABLE
            detail = (
                "SSH answered; no cached setup or hardware; "
                f"{note or f'run mainboard setup {alias}'}"
            )
        else:
            access = Access.PROVISIONED
            endpoint = "login endpoint only; " if profile.kind in {"pbs", "slurm"} else ""
            action = note or f"inspect mainboard jobs and mainboard facts --on {alias}"
            detail = (
                f"{cached}; {endpoint}job readiness and GPU availability not checked; {action}"
            )
        return ComputePath(
            name=alias,
            kind=profile.kind,
            access=access,
            detail=detail,
            cached_at=cached_at,
        )

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
        held = Holdings(self.board.root).read()
        # Loading credentials mutates the process environment. Finish before SSH launches:
        # concurrent setenv and execve can fail with EFAULT before ssh itself starts.
        Credentials().load()
        machines: list[Callable[[], ComputePath]] = [self.here]
        machines.extend(
            partial(self.machine, alias, profile, recorded.get(alias))
            for alias, profile in sorted(self.board.manifest.profiles().items())
            if route(profile.kind) == "ssh-family"
        )
        probes: list[Callable[[], list[ComputePath]]] = [
            *(partial(_alone, machine) for machine in machines),
            *(partial(self.offered, backend, held) for backend in self.providers),
        ]
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            return [path for listed in pool.map(lambda probe: probe(), probes) for path in listed]

    def offered(self, backend: ProviderBackend, held: Mapping[str, Held]) -> list[ComputePath]:
        """One provider's row, then a row for every machine it says the account is renting.

        The rentals are asked for only of a provider whose key is here, since a provider nobody
        configured has nothing to list and its own row already says why.

        backend: the registered backend to ask.
        held: the machines this workspace is holding, by alias, which name their rental rows.
        """
        standing = self.provider(backend)
        if standing.access is not Access.KEYED or not isinstance(backend, Inventory):
            return [standing]
        try:
            rented = backend.rentals()
        except _PROBE_FAULTS as fault:
            refused = ComputePath(
                name=backend.name, kind=_RENTAL, access=Access.UNREACHABLE, detail=str(fault)
            )
            return [standing, refused]
        holds = {hold.handle: hold for hold in held.values()}
        return [standing, *(self.rental(backend, row, holds.get(row.handle)) for row in rented)]

    @staticmethod
    def rental(backend: ProviderBackend, rented: Rented, held: Held | None) -> ComputePath:
        """One live rental, named by its hold when this workspace holds it.

        backend: the provider that reported it.
        rented: the rental as the provider reported it.
        held: the hold it belongs to, None for a rental this workspace is not holding.
        """
        where = f"{backend.name} {rented.handle}, {rented.gpu or 'unknown card'}, {rented.status}"
        owner = (
            f"held until {held.deadline.isoformat()}; mainboard release {held.alias}"
            if held is not None
            else f"not held here, label {rented.label or 'none'}"
        )
        return ComputePath(
            name=held.alias if held is not None else f"{backend.name}:{rented.handle}",
            kind=_RENTAL,
            access=Access.RENTED,
            detail=f"{where}; {owner}",
            usd_hr=rented.usd_hr,
        )

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


def _alone(probe: Callable[[], ComputePath]) -> list[ComputePath]:
    """`probe`'s one row as a list, the shape a provider's several rows share."""
    return [probe()]
