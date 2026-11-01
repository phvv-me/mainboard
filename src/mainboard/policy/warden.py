"""The checks an unattended act passes: Scope at launch, Grant at push, Budget at rent.

Each asks a native enforcer rather than standing in for one. A launched agent child must carry its
harness's own flags and settings; a rental needs its provider's cap (cloud-cap's
`ProviderBackend.cap`, which refuses a provider with none, a card's limit read from the owner's
grants file); only a push leans on mb's own record, the owner's grants. What a check refuses and
the owner could allow, it files as an ask, naming it in the refusal: a push no grant covers, a
rental past the `[policy]` budget. A child missing its enforcer is refused outright, since no grant
lifts a harness's own controls.

Telemetry decides nothing here (N3): a check reads the grants file and the lake's `costs`, never a
usage row, and an ask it files carries no authority.
"""

from datetime import UTC, datetime
from time import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..core.constants.units import SECONDS_PER_DAY
from ..core.errors import MissionError
from ..core.project import Project
from ..dispatch.backends import route
from ..enums import Act
from .asks import Ask, Asks
from .constants import ENFORCERS
from .enforcers import Enforcement
from .grants import GrantsFile, said

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ..costs.ledger import Ledger
    from ..dispatch.hosts.lease import Cap
    from ..manifest.schema.compute.host import HostProfile
    from ..manifest.schema.services.policy import Policy
    from ..serve.constants import Service


class Warden:
    """The policy checks of one workspace.

    root: the workspace, whose grants file and spool the checks read and write.
    policy: its `[policy]` table.
    """

    def __init__(self, root: Path, policy: Policy) -> None:
        self.root = root
        self.policy = policy
        self.grants = GrantsFile(root)
        self.asks = Asks(root)

    def _allowed(self, act: Act, target: str, *, why: str) -> None:
        """Let `act` on `target` go ahead under a live grant; else file an ask and refuse.

        why: what the refusal and the ask say stood in the way.
        """
        now = datetime.now(UTC)
        try:
            ask = Ask(act=act, target=target, why=why, at=now)
        except ValidationError as fault:
            raise MissionError(f"{act} {target!r}: {said(fault)}") from None
        grants = self.grants.read()
        if grants.covering(act, target, at=now) is not None:
            return
        short = self.asks.file(ask, grants).id[:8]
        name = Project().name
        raise MissionError(
            f"{why}; asked the owner as {short}, which `{name} serve answer {short}` grants "
            f"from a terminal (`{name} serve asks` lists every ask)"
        )

    def launch(self, services: Sequence[Service]) -> None:
        """Refuse to start any running agent child its harness's enforcer does not bound.

        services: the serve children declared; an off one needs nothing.
        """
        enforcement = Enforcement(self.root, services)
        for enforcer in ENFORCERS:
            running = [s for s in services if s.name == enforcer.child and not s.off]
            for service in running:
                if missing := enforcement.missing(enforcer, service.command):
                    raise MissionError(
                        f"serve would start {service.name} without {'; '.join(missing)}, its "
                        "harness's own bound on what it runs; give the child its enforcer first"
                    )

    def push(self, target: str) -> None:
        """Refuse a push no live grant covers, asking the owner for it.

        target: `owner/repo:ref`, the remote as GitHub names it and the ref or bookmark pushed.
        """
        self._allowed(Act.PUSH, target, why=f"no live grant covers push {target}")

    def rent(self, profile: HostProfile, *, max_usd: float, ledger: Ledger) -> Cap:
        """The cap a rental goes ahead under; refused uncapped, or past the `[policy]` budget.

        The cap is cloud-cap's (`ProviderBackend.cap`): a card provider has none until the owner
        declares its card's limit in the grants file (`serve grant card`), never in mb.toml. A
        grant on the provider lifts the budget line only, never the cap, which still bounds the
        bill and is what the rent's own `admit` checks.

        profile: the host the rental goes through.
        max_usd: the most this rental may cost.
        ledger: the lake's `costs`, what rentals were billed.
        """
        backend = route(profile.kind)
        if isinstance(backend, str):
            raise MissionError(f"a {profile.kind!r} host rents nothing, so nothing caps it")
        cap = backend.cap(card_usd=self.grants.read().card(backend.name))
        budget = self.policy.rent
        if budget is None:
            return cap
        since = time() - budget.days * SECONDS_PER_DAY
        spent = sum(row.billed_usd for row in ledger.observations() if row.t_submit >= since)
        if not budget.allows(spent_usd=spent, max_usd=max_usd):
            why = (
                f"rentals billed ${spent:.2f} of the ${budget.usd:.2f} the last {budget.days} "
                f"days allow, and this one may cost ${max_usd:.2f}"
            )
            self._allowed(Act.RENT, cap.provider, why=why)
        return cap
