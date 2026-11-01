"""Grants, card limits, asks, grants files and child settings as the policy tests arrange them.

A grant and an ask built with no time given share one moment, which a test reads back from what
it built (`grant().granted`, `ask().at`).
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from mainboard.enums import Act
from mainboard.policy.asks import Ask
from mainboard.policy.constants import ENFORCERS
from mainboard.policy.grants import Card, Grant, Grants, GrantsFile

_GRANTED = datetime(2026, 10, 29, 6, 0, tzinfo=UTC)
_LASTING = timedelta(hours=1)
_CLAUDE_DENIES = next(enforcer for enforcer in ENFORCERS if enforcer.child == "claude").denies
# How a grants file can be one its reader refuses before reading a grant of it.
type Fault = Literal["no-text", "unreadable", "a-pipe"]


def ask(target: str = "me/repo:main", *, act: Act = Act.PUSH, at: datetime = _GRANTED) -> Ask:
    """An ask for `act` on `target`, filed at `at`."""
    return Ask(act=act, target=target, why=f"no live grant covers {act} {target}", at=at)


def card(provider: str = "lambda", *, usd: float = 100.0, declared: datetime = _GRANTED) -> Card:
    """The owner's declaration that the card paying `provider` stops at `usd`, as of `declared`."""
    return Card(provider=provider, usd=usd, declared=declared)


def claude_settings(deny: tuple[str, ...] | str = _CLAUDE_DENIES, **permissions: list[str]) -> str:
    """The Claude child's settings file text: `deny`, the bypass refused, and `permissions`."""
    held = {"deny": deny, "disableBypassPermissionsMode": "disable", **permissions}
    return json.dumps({"permissions": held})


def grant(
    scope: str = "me/repo:mb/*",
    *,
    act: Act = Act.PUSH,
    granted: datetime = _GRANTED,
    until: datetime = _GRANTED + _LASTING,
    ask: str = "",
) -> Grant:
    """A grant of `act` within `scope` from `granted` until `until`, an hour by default."""
    return Grant(act=act, scope=scope, granted=granted, until=until, ask=ask)


def granted(root: Path, grants: Grants) -> GrantsFile:
    """The grants file of `root` holding `grants`, written as the owner's own 0600 file."""
    grants_file = GrantsFile(root)
    grants_file.path.parent.mkdir(mode=0o700, parents=True)
    grants_file.path.write_text(grants.model_dump_json(), encoding="utf-8")
    os.chmod(grants_file.path, 0o600)
    return grants_file


def live(scope: str = "me/repo:mb/*", *, act: Act = Act.PUSH) -> Grant:
    """A grant of `act` within `scope` in force from a minute ago for an hour."""
    now = datetime.now(UTC)
    return grant(scope, act=act, granted=now - timedelta(minutes=1), until=now + _LASTING)


def spoiled(root: Path, *, fault: Fault) -> GrantsFile:
    """The grants file of `root` as its reader must refuse it, at once, for `fault`.

    fault: a 0600 file holding a byte that is no UTF-8, the same file at mode 000, or a FIFO.
    """
    grants_file = GrantsFile(root)
    grants_file.path.parent.mkdir(mode=0o700, parents=True)
    if fault == "a-pipe":
        os.mkfifo(grants_file.path, 0o600)
        return grants_file
    grants_file.path.write_bytes(b'{"grants": [], "note": "\xff"}')
    os.chmod(grants_file.path, 0o000 if fault == "unreadable" else 0o600)
    return grants_file
