# The survey behind `mainboard compute`: every place this workspace can run work, in one list.
# This machine, the hosts the manifest declares and the machines it is holding, every registered
# provider backend, and every machine a provider says this account is renting right now, each
# answered by one bounded probe. A host that will not answer and a provider with no key are row
# states here, never failures, so the whole fleet still lists when part of it is down.

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum, auto
from functools import cached_property, partial
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import Field

from .core.errors import MissionError
from .core.section import Verdict
from .dispatch.backends.base import Account, Credentials, Inventory, ProviderBackend, route
from .dispatch.transport import HostUnreachable, SshTransport
from .fitness import Fitness
from .manifest.held import Holdings
from .probe.snapshot import HostFacts
from .probe.system import System

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .board import Board
    from .dispatch.backends.base import Rented
    from .dispatch.onboard import HostSetup
    from .manifest.held import Held
    from .manifest.schema.host import HostProfile

# One bounded ssh round trip per host, under a policy tightened for a survey. `connect_timeout`
# is what a host that never answers costs; the derived deadline is the only bound on one that
# connects and then goes quiet. Ten seconds is generous on purpose: a host behind a ProxyJump
# measured here answers in under a second most times and in six sometimes, and calling it down
# is worse than waiting. The deadline still lands far under the dispatch default's, so the
# survey takes the time one stalled host takes rather than a minute.
_PROBE_SSH = SshTransport(connect_timeout=10.0, server_alive_interval=2.0, server_alive_count=1)

# The `kind` of a provider row and of a live rental. Providers have no scheduler, so naming the
# route keeps the column meaning how the path is reached instead of repeating the provider.
_PROVIDER = "provider"
_RENTAL = "rental"

# What a live probe may fail with and become a row state: `OSError` is every urllib network
# fault (an `HTTPError` 4xx included), `MissionError` a backend refusing before the network, and
# `ValueError` a provider answering something unreadable.
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
    detail: the one line behind `access`: a machine's hardware, a silent host's refusal, the
        variable a provider with no key needs.
    usd_hr: a live cheapest-offer sample, None where no price is a cheap question.
    credit_usd: the balance the provider reports, None where it exposes none.
    observed_at: UTC completion time of this survey observation, not a readiness lease.
    cached_at: onboarding time of retained host facts; empty means their age is unknown.
    issues: every non-pass finding about the machine as `section: detail`, from the judge
        `facts` and `setup` print, empty for a fit machine or one with no census.
    """

    name: str
    kind: str
    access: Access
    detail: str = ""
    usd_hr: float | None = None
    credit_usd: float | None = None
    observed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    cached_at: str = ""
    issues: str = ""


def summary(facts: HostFacts) -> str:
    """One line naming what a machine has, its GPUs by model then its memory."""
    counted = Counter(gpu.name for gpu in facts.gpus)
    memory = f"{facts.memory_total_bytes / 1e9:.0f} GB RAM"
    return ", ".join([*(f"{count}x {name}" for name, count in counted.items()), memory])


def reachable(host: str, ssh: SshTransport = _PROBE_SSH) -> str:
    """Why `host` cannot be reached right now, empty when one bounded ssh round trip lands.

    An echo marker works in POSIX shells, cmd and PowerShell without a provisioned environment.
    This proves a remote command answered, not that a GPU job can run.
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

    The rows come in the order the question is asked: what is under this desk, what is set up
    elsewhere, and what can be rented. A probe that fails becomes its row's state, so a dead host
    or an unconfigured provider never costs the rest, and the probes run in one pool, so the
    survey takes as long as its slowest probe. Every network touch is injected.
    """

    def __init__(
        self,
        board: Board,
        *,
        facts: Callable[[], HostFacts] | None = None,
        reach: Callable[[str], str] = reachable,
        providers: Sequence[ProviderBackend] | None = None,
    ) -> None:
        """facts: probes this machine's hardware and software, over the workspace root when None.
        reach: answers why a host cannot be reached, empty when it can.
        providers: the provider backends to ask, every registered one when None.
        """
        self.board = board
        self.facts = facts or partial(HostFacts.collected, board.root)
        self.reach = reach
        self.providers = (
            [backend() for backend in ProviderBackend.implementations()]
            if providers is None
            else list(providers)
        )

    def here(self) -> ComputePath:
        """This machine, from its own probed facts, with what they mean for this workspace."""
        found = self.facts()
        return ComputePath(
            name="local",
            kind="local",
            access=Access.HERE,
            detail=f"{summary(found)}; live hardware, GPU availability not checked",
            issues=self.issues(found.system, "local"),
        )

    @cached_property
    def fitness(self) -> Fitness:
        """The judge every machine row shares, so the lock is read once per survey."""
        return Fitness(self.board.root, self.board.manifest)

    def issues(self, system: System, host: str) -> str:
        """Every finding about `host` that is not a pass, one `section: detail` each.

        system: the host's census, empty for a host onboarded before censuses were recorded,
            which then has nothing to say rather than a warning per row.
        """
        if not system.surveyed:
            return ""
        judged = self.fitness.judge(system, host=host)
        return "; ".join(
            f"{row.section}: {row.detail}" for row in judged if row.verdict is not Verdict.PASS
        )

    def machine(self, alias: str, profile: HostProfile, setup: HostSetup | None) -> ComputePath:
        """One declared host: whether it answers, and what onboarding already recorded of it.

        Provisioned means an onboarding record exists, not that its environment or scheduler
        still works. Retained hardware is cached and may be stale; PBS/Slurm login hardware says
        nothing about a future compute allocation. A profile's vars.status-note may replace the
        generic next-step advice, never observed state.

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
        census = setup.hardware.system if setup is not None and setup.hardware else System()
        return ComputePath(
            name=alias,
            kind=profile.kind,
            access=access,
            detail=detail,
            cached_at=cached_at,
            issues=self.issues(census, alias),
        )

    def onboarded(self) -> dict[str, HostSetup]:
        """What onboarding recorded for each alias, read from the dispatch cache.

        Its own verb because the cache is one SQLite connection only its opening thread may use:
        a caller that hands this survey to a pool reads the records first, on that thread, and
        passes them to `paths`.
        """
        return {setup.host: setup for setup in self.board.dispatcher.cache.hosts()}

    def paths(self, setups: Mapping[str, HostSetup] | None = None) -> list[ComputePath]:
        """Every compute path, probed in parallel, this machine first.

        A declared host whose kind routes to a provider is left to that provider's own row, so a
        rented machine is listed once rather than probed as if it were an ssh box.

        setups: the onboarding records by alias (see `onboarded`), read here when None.
        """
        recorded = self.onboarded() if setups is None else setups
        held = Holdings(self.board.root).read()
        # Loading credentials mutates the process environment. Finish before SSH launches:
        # concurrent setenv and execve can fail with EFAULT before ssh itself starts.
        Credentials().load()
        machines: list[Callable[[], ComputePath]] = [
            self.here,
            *(
                partial(self.machine, alias, profile, recorded.get(alias))
                for alias, profile in sorted(self.board.manifest.profiles().items())
                if route(profile.kind) == "ssh-family"
            ),
        ]
        with ThreadPoolExecutor(max_workers=len(machines) + len(self.providers)) as pool:
            rows = pool.map(lambda machine: machine(), machines)
            offers = pool.map(partial(self.offered, held=held), self.providers)
            return [*rows, *(path for listed in offers for path in listed)]

    def offered(self, backend: ProviderBackend, held: Mapping[str, Held]) -> list[ComputePath]:
        """One provider's row, then a row for every machine it says the account is renting.

        Only a provider whose key is here is asked for rentals; an unconfigured one has nothing
        to list and its own row already says why.

        held: the machines this workspace is holding, by alias, which name their rental rows.
        """
        standing = self.provider(backend)
        if standing.access is not Access.KEYED or not isinstance(backend, Inventory):
            return [standing]
        try:
            rented = backend.rentals()
        except _PROBE_FAULTS as fault:
            return [
                standing,
                ComputePath(
                    name=backend.name, kind=_RENTAL, access=Access.UNREACHABLE, detail=str(fault)
                ),
            ]
        holds = {hold.handle: hold for hold in held.values()}
        return [standing, *(self.rental(backend, row, holds.get(row.handle)) for row in rented)]

    @staticmethod
    def rental(backend: ProviderBackend, rented: Rented, held: Held | None) -> ComputePath:
        """One live rental `backend` reported, named by its hold when this workspace holds it."""
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

        Answering for an account is a capability, so a backend without it is listed with what it
        lacks rather than asked and made to raise: a survey stays a listing.
        """
        row = partial(ComputePath, name=backend.name, kind=_PROVIDER)
        if not isinstance(backend, Account):
            return row(access=Access.UNKEYED, detail=backend.refusal(Account))
        try:
            standing = backend.standing()
        except _PROBE_FAULTS as fault:
            return row(access=Access.UNREACHABLE, detail=str(fault))
        return row(
            access=Access.KEYED if standing.keyed else Access.UNKEYED,
            detail=standing.note,
            usd_hr=standing.usd_hr,
            credit_usd=standing.credit_usd,
        )
