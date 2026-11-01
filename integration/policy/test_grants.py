"""A grant covers one exact act inside its scope while it lasts, in the owner's own 0600 file.

The same file holds the owner's card limits, the latest declaration for a provider holding.
"""

import os
import re
import stat
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import ValidationError

from mainboard import MissionError
from mainboard.enums import Act
from mainboard.policy.grants import Grants, GrantsFile
from mainboard.render import Audience, Terminal

from .builders import Fault, card, grant, granted, spoiled

_GRANTED, _UNTIL = grant().granted, grant().until
_WORD = st.from_regex(r"[A-Za-z0-9_.@+-]{1,8}", fullmatch=True).filter(
    lambda word: word not in {".", ".."}
)
# Words joined by `/` or `:`, as a pushed ref, a provider or a child is spelled.
_TARGET = st.lists(st.tuples(_WORD, st.sampled_from("/:")), min_size=1, max_size=4).map(
    lambda pairs: "".join(word + joint for word, joint in pairs)[:-1]
)
_COVERED = "me/repo:mb/save"


@given(prefix=_TARGET, rest=_TARGET)
@example(prefix="me/repo:mb", rest="save")
def test_a_prefix_scope_covers_the_targets_below_its_separator_and_no_sibling(
    prefix: str, rest: str
) -> None:
    covering = grant(f"{prefix}/*")
    assert covering.covers(Act.PUSH, f"{prefix}/{rest}", at=_GRANTED)
    assert not covering.covers(Act.PUSH, f"{prefix}-{rest}", at=_GRANTED)
    assert not covering.covers(Act.PUSH, prefix, at=_GRANTED)


@given(at=st.datetimes(timezones=st.just(UTC)), act=st.sampled_from(Act))
@example(at=_UNTIL, act=Act.PUSH)
@example(at=_GRANTED, act=Act.PUSH)
def test_a_grant_covers_only_its_own_act_from_its_grant_up_to_its_end(
    at: datetime, act: Act
) -> None:
    expected = act is Act.PUSH and _GRANTED <= at < _UNTIL
    assert grant().covers(act, _COVERED, at=at) == expected


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("me/repo:mb/../main", id="a-target-climbing-out-of-the-scope"),
        pytest.param("me/repo:mb/./save", id="a-target-naming-the-scope-itself-again"),
        pytest.param("me/repo:mb-x/save", id="a-sibling-sharing-the-prefix-text"),
        pytest.param("me/repo:MB/save", id="the-scope-in-another-case"),
        pytest.param("me/repo:mb/save now", id="a-target-with-a-space"),
        pytest.param("other/repo:mb/save", id="the-same-ref-of-another-repository"),
    ],
)
def test_a_grant_covers_no_target_outside_its_scope(target: str) -> None:
    assert not grant().covers(Act.PUSH, target, at=_GRANTED)


@pytest.mark.parametrize(
    "scope",
    [
        pytest.param("*", id="everything"),
        pytest.param("me/repo:mb*", id="a-star-glued-to-a-word"),
        pytest.param("me/**", id="two-stars"),
        pytest.param("me//*", id="an-empty-word-before-the-star"),
        pytest.param("me/*/main", id="a-star-inside"),
        pytest.param("me/", id="a-trailing-separator"),
        pytest.param("me/../*", id="a-word-climbing-out"),
        pytest.param("w" * 201, id="a-target-past-two-hundred-characters"),
    ],
)
def test_a_scope_that_names_no_target_is_refused(scope: str) -> None:
    with pytest.raises(ValidationError, match=r"is no (scope|target)"):
        grant(scope)


def test_a_grant_ending_before_it_begins_is_refused() -> None:
    with pytest.raises(ValidationError, match="ends before it was granted"):
        grant(until=_GRANTED)


def test_an_agent_writes_no_grant_and_leaves_no_file(workspace: Path) -> None:
    grants_file = GrantsFile(workspace)
    agent = Terminal(audience=Audience.AGENT, marked=True)
    with agent.installed(), pytest.raises(MissionError, match="the owner's act"):
        grants_file.changed("serve grant", lambda grants: Grants(grants=(grant(),)))
    assert not grants_file.path.parent.exists()


def test_the_owner_writes_the_grants_whole_into_a_private_file_and_folder(
    workspace: Path, owner: Terminal
) -> None:
    grants_file = GrantsFile(workspace)
    written = grants_file.changed("serve grant", lambda grants: Grants(grants=(grant(),)))
    modes = [
        stat.S_IMODE(path.stat().st_mode) for path in (grants_file.path, grants_file.path.parent)
    ]
    assert (modes, grants_file.read()) == ([0o600, 0o700], written)


def test_writers_racing_for_the_file_each_keep_their_grant_under_a_private_lock(
    workspace: Path, owner: Terminal
) -> None:
    grants_file = GrantsFile(workspace)
    scopes = [f"me/repo:mb/{number}" for number in range(8)]

    def add(scope: str) -> None:
        grants_file.changed(
            "serve grant", lambda held: Grants(grants=(*held.grants, grant(scope)))
        )

    with ThreadPoolExecutor(len(scopes)) as pool:
        for racer in [pool.submit(copy_context().run, add, scope) for scope in scopes]:
            racer.result()
    lock = grants_file.path.with_suffix(".lock")
    assert sorted(held.scope for held in grants_file.read().grants) == scopes
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(0o644, id="readable-by-others"),
        pytest.param(0o620, id="writable-by-the-group"),
        pytest.param(0o604, id="readable-by-anyone-but-the-group"),
    ],
)
def test_a_grants_file_open_to_anyone_else_grants_nothing(workspace: Path, mode: int) -> None:
    grants_file = granted(workspace, Grants(grants=(grant(),)))
    os.chmod(grants_file.path, mode)
    with pytest.raises(MissionError, match="open to others"):
        grants_file.read()


def test_a_grants_file_that_is_a_link_grants_nothing(workspace: Path) -> None:
    grants_file = granted(workspace, Grants(grants=(grant(),)))
    elsewhere = grants_file.path.rename(grants_file.path.with_suffix(".elsewhere"))
    grants_file.path.symlink_to(elsewhere)
    with pytest.raises(MissionError, match="is a link"):
        grants_file.read()


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        pytest.param("no-text", "holds no text; move it aside", id="a-byte-that-is-no-utf8"),
        pytest.param("unreadable", "not readable by its owner; `chmod 600`", id="mode-000"),
        pytest.param("a-pipe", "is not the owner's own file", id="a-fifo-nobody-writes"),
    ],
)
def test_a_grants_file_mb_cannot_read_is_refused_at_once_with_its_fix(
    workspace: Path, fault: Fault, reason: str
) -> None:
    grants_file = spoiled(workspace, fault=fault)
    with pytest.raises(MissionError, match=re.escape(reason)):
        grants_file.read()


@given(
    limits=st.lists(
        st.tuples(st.floats(min_value=0.01, max_value=1e6), st.datetimes(timezones=st.just(UTC))),
        max_size=5,
    )
)
def test_a_cards_limit_is_the_owners_latest_declaration_for_that_provider_alone(
    limits: Sequence[tuple[float, datetime]],
) -> None:
    cards = (*(card(usd=usd, declared=at) for usd, at in limits), card("vultr", usd=7.0))
    latest = max(limits, key=lambda limit: limit[1], default=(0.0, _GRANTED))[0]
    assert Grants(cards=cards).card("lambda") == latest


@pytest.mark.parametrize("usd", [0.0, -1.0, float("nan"), float("inf")])
def test_a_card_limit_that_is_no_positive_finite_figure_is_refused(usd: float) -> None:
    with pytest.raises(ValidationError, match="greater than 0|finite number"):
        card(usd=usd)


def test_the_grants_in_force_are_those_begun_and_not_ended() -> None:
    now = datetime.now(UTC)
    ended = grant(granted=now - timedelta(hours=2), until=now - timedelta(hours=1))
    pending = grant(granted=now + timedelta(hours=1), until=now + timedelta(hours=2))
    current = grant(granted=now - timedelta(hours=1), until=now + timedelta(hours=1))
    assert Grants(grants=(ended, pending, current)).live(at=now) == [current]
