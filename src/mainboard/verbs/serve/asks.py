"""The `serve` verbs of leave: what waits for the owner's answer, the answer, and leave unasked.

`asks` only reads, so an agent may run it. `answer` and `grant` write the owner's grants file, and
both `main` and the file refuse unless a person at a terminal with no harness marker types them
(cli-terminal's refusal, which `MB_AGENT=0` does not lift). That is a guard, not a boundary:
`env -u <markers> script ...` passes it, and any process of the owner's user can write the file.
The boundary is the harness sandbox (N1(a)) plus the agents' deny rules (`constants.DENIED`).
"""

from datetime import UTC, datetime, timedelta

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, ValidationError

from ...constants import LONGEST_GRANT_HOURS
from ...core.errors import MissionError
from ...core.project import Project
from ...dispatch.backends import ProviderBackend
from ...enums import Act
from ...policy.asks import Asks
from ...policy.grants import Card, Declined, Grant, GrantsFile, said
from ...render import Output
from ..constants import TREE
from ..session import Verbs


@Parameter(name="*")
class Leave(FrozenModel):
    """What one `serve grant` writes beyond its act and scope."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    hours: float = 1.0
    """how long the grant lasts."""
    revoke: bool = False
    """end now every grant in force of exactly this act and scope instead."""
    usd: float = 0.0
    """for `card`, the limit of the card paying the provider."""


_COMPACT = Output()
_LEAVE = Leave()


def _lasting(hours: float) -> timedelta:
    """`hours` as how long a grant lasts, refused unless finite, above 0 and within a year."""
    if 0 < hours <= LONGEST_GRANT_HOURS and (lasting := timedelta(hours=hours)):
        return lasting
    raise MissionError(f"give --hours above 0 and at most {LONGEST_GRANT_HOURS:g}, a year")


class AskVerbs(Verbs):
    """Read the asks, and write the owner's answers, grants and card limits."""

    @staticmethod
    def _card(provider: str, *, leave: Leave, at: datetime) -> Card:
        """The owner's declaration of the limit of the card paying `provider`, a card provider."""
        if leave.revoke:
            raise MissionError("a card's limit is declared anew, never revoked; give --usd")
        if provider not in (billed := ProviderBackend.carded()):
            raise MissionError(f"{provider} bills no card; the providers that do are {billed}")
        try:
            return Card(provider=provider, usd=leave.usd, declared=at)
        except ValidationError as fault:
            raise MissionError(f"a card's limit is --usd above 0: {said(fault)}") from None

    @staticmethod
    def _granted(act: Act, scope: str, *, hours: float, at: datetime) -> Grant:
        """The owner's grant of `act` within `scope` from `at` for `hours`."""
        try:
            return Grant(act=act, scope=scope, granted=at, until=at + _lasting(hours))
        except ValidationError as fault:
            raise MissionError(said(fault)) from None

    def answer(
        self, ask: str, *, hours: float = 1.0, decline: bool = False, output: Output = _COMPACT
    ) -> None:
        """Answer one waiting ask: grant exactly its act and target for some hours, or decline it.

        The grant names the very act and target asked, never a wider scope. A declined ask waits
        again only once something asks it anew. Refuses unless the owner types it at a terminal.

        Args:
            ask: the ask's id, or its first characters as `serve asks` shows them.
            hours: how long the grant lasts.
            decline: say no instead of granting.
        """
        root, now = self.session.workspace_root(), datetime.now(UTC)
        grants_file = GrantsFile(root)
        found = [k for k in Asks(root).pending(grants_file.read(), at=now) if k.id.startswith(ask)]
        if len(found) != 1:
            listing = f"`{Project().name} serve asks` lists them"
            raise MissionError(f"{len(found)} waiting asks start with {ask!r}; {listing}")
        chosen = found[0]
        if decline:
            refusal = Declined(ask=chosen.id, at=now)
            grants_file.changed(f"serve answer {ask}", lambda grants: grants.declining(refusal))
            output.print_record({"ask": chosen.id, "answer": "declined"}, title="answered")
            return
        granted = Grant(
            act=chosen.act,
            scope=chosen.target,
            granted=now,
            until=now + _lasting(hours),
            ask=chosen.id,
        )
        grants_file.changed(f"serve answer {ask}", lambda grants: grants.added(granted))
        output.print_record(_shown(granted), title="answered")

    def asks(self, *, output: Output = _COMPACT) -> None:
        """Show the asks waiting for the owner's answer, oldest first.

        An ask is an act a policy check refused and the owner could allow: a push no grant covers,
        or a rental past the `[policy]` budget. It carries no authority, so anyone may file or
        list one, and only `serve answer` grants it.
        """
        root = self.session.workspace_root()
        pending = Asks(root).pending(GrantsFile(root).read(), at=datetime.now(UTC))
        rows = [
            {
                "ask": ask.id[:8],
                "act": str(ask.act),
                "target": ask.target,
                "asked": ask.at.astimezone().isoformat(timespec="seconds"),
                "why": ask.why,
            }
            for ask in pending
        ]
        output.print_rows(rows, title="asks", columns=("ask", "act", "target", "asked", "why"))

    def grant(
        self,
        act: Act | None = None,
        scope: str = "",
        *,
        leave: Leave = _LEAVE,
        output: Output = _COMPACT,
    ) -> None:
        """Grant one act within one scope for some hours, end such grants, or list those in force.

        A grant covers only its very act, only targets its scope names, and only until it ends.
        `card` is the one act that is no leave: `serve grant card lambda --usd 100` declares the
        limit of the card paying a provider that bills one after the fact, which that provider
        rents nothing without; the latest declaration holds, with no expiry. Refuses unless the
        owner types it at a terminal.

        Args:
            act: what is granted; with none, the grants in force are listed.
            scope: what it covers: one target (`owner/repo:mb/save`, `vast`), or every target
                below a prefix ending in `/*` or `:*` (`owner/repo:mb/*`); for `card`, the
                provider.
        """
        root, now = self.session.workspace_root(), datetime.now(UTC)
        grants_file = GrantsFile(root)
        if act is None:
            live = grants_file.read().live(at=now)
            output.print_rows([_shown(g) for g in live], title="grants", columns=_COLUMNS)
            return
        if not scope:
            raise MissionError(f"name what a {act} grant covers, a target or a `prefix/*` scope")
        command = f"serve grant {act} {scope}"
        if act is Act.CARD:
            card = AskVerbs._card(scope, leave=leave, at=now)
            grants_file.changed(f"{command} --usd {card.usd:g}", lambda g: g.declaring(card))
            declared = card.declared.astimezone().isoformat(timespec="seconds")
            shown = {"card": card.provider, "usd": f"{card.usd:g}", "declared": declared}
            output.print_record(shown, title="declared")
            return
        if leave.revoke:
            ended = grants_file.changed(command, lambda grants: grants.revoked(act, scope, at=now))
            revoked = [_shown(g) for g in ended.grants if g.until == now]
            output.print_rows(revoked, title="revoked", columns=_COLUMNS)
            return
        granted = AskVerbs._granted(act, scope, hours=leave.hours, at=now)
        grants_file.changed(command, lambda grants: grants.added(granted))
        output.print_record(_shown(granted), title="granted")


_COLUMNS = ("act", "scope", "granted", "until", "ask")


def _shown(grant: Grant) -> dict[str, str]:
    """One grant as a printed row, its instants in this machine's zone."""
    return {
        "act": str(grant.act),
        "scope": grant.scope,
        "granted": grant.granted.astimezone().isoformat(timespec="seconds"),
        "until": grant.until.astimezone().isoformat(timespec="seconds"),
        "ask": grant.ask[:8],
    }


_ASKS = AskVerbs()
_BOUND = TREE.node("serve").bind(answer=_ASKS.answer, asks=_ASKS.asks, grant=_ASKS.grant)
answer, asks, grant = _BOUND["answer"], _BOUND["asks"], _BOUND["grant"]
