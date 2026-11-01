"""The checks an unattended act passes, and the doctor row of the native enforcers they read."""

import json
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mainboard import MissionError, Project
from mainboard.core import Verdict
from mainboard.costs import Ledger, Observation
from mainboard.enums import Act
from mainboard.manifest import HostProfile
from mainboard.manifest.schema.services.policy import Budget, Policy
from mainboard.policy.constants import DENIED, ENFORCERS
from mainboard.policy.enforcers import Enforcement
from mainboard.policy.grants import Grants
from mainboard.policy.warden import Warden
from mainboard.serve.constants import SERVICES, Service
from mainboard.state import Lake

from .builders import card, claude_settings, grant, granted, live

_TARGET = "me/repo:mb/save"
_BUDGET = Policy(rent=Budget(usd=10.0, days=7))
_VAST = HostProfile(kind="vast")
_LAMBDA = HostProfile(kind="lambda")
_DAY = 86400.0
_CLAUDE = next(enforcer for enforcer in ENFORCERS if enforcer.child == "claude")
_CLAUDE_DENIES = _CLAUDE.denies
_CODEX = next(service for service in SERVICES if service.name == "codex").command
_ALLOWED = {"HOME", "LANG", "LOGNAME", "PATH", "TMPDIR", "USER"}
_SANDBOXED = 'sandbox_mode = "workspace-write"\napproval_policy = "on-request"\n'


@pytest.fixture
def ledger(workspace: Path) -> Ledger:
    """The workspace lake's `costs`, empty."""
    return Ledger(Lake.at(workspace).session())


def _spooled(workspace: Path) -> int:
    """How many lines the workspace's spool holds."""
    spool = workspace / ".mb" / "telemetry" / "spool"
    return sum(len(path.read_text().splitlines()) for path in spool.glob("*.jsonl"))


def _on(name: str, **changed: tuple[str, ...] | tuple[tuple[str, str], ...]) -> list[Service]:
    """The declared children with `name` running, its fields `changed`."""
    return [
        service.model_copy(update={"off": "", **changed}) if service.name == name else service
        for service in SERVICES
    ]


def test_a_push_a_live_grant_covers_goes_ahead_and_asks_nothing(workspace: Path) -> None:
    granted(workspace, Grants(grants=(live(),)))
    Warden(workspace, Policy()).push(_TARGET)
    assert _spooled(workspace) == 0


@pytest.mark.parametrize(
    "grants",
    [
        pytest.param(Grants(), id="no-grant-at-all"),
        pytest.param(Grants(grants=(grant(),)), id="a-grant-that-has-ended"),
        pytest.param(Grants(grants=(live(act=Act.LAUNCH),)), id="a-live-grant-of-another-act"),
        pytest.param(Grants(grants=(live("me/repo:mb-x/*"),)), id="a-live-grant-of-a-sibling"),
        pytest.param(Grants(grants=(live("me/repo:mb/save/*"),)), id="a-live-grant-below-it"),
    ],
)
def test_a_push_no_live_grant_covers_is_refused_and_asked_once(
    workspace: Path, grants: Grants
) -> None:
    granted(workspace, grants)
    with pytest.raises(MissionError, match=f"no live grant covers push {_TARGET}; asked"):
        Warden(workspace, Policy()).push(_TARGET)
    assert _spooled(workspace) == 1


def test_a_push_of_something_that_is_no_target_is_refused_and_asks_nothing(
    workspace: Path,
) -> None:
    granted(workspace, Grants(grants=(live(),)))
    with pytest.raises(MissionError, match="is no target"):
        Warden(workspace, Policy()).push("me/repo:mb/../main")
    assert _spooled(workspace) == 0


def test_a_rent_through_a_card_provider_goes_ahead_only_under_the_owners_latest_limit(
    workspace: Path, ledger: Ledger
) -> None:
    with pytest.raises(
        MissionError, match="the owner declares it with `.* serve grant card lambda"
    ):
        Warden(workspace, _BUDGET).rent(_LAMBDA, max_usd=1.0, ledger=ledger)
    earlier = card(usd=900.0, declared=datetime(2026, 10, 1, tzinfo=UTC))
    granted(workspace, Grants(cards=(card(usd=40.0), earlier, card("vultr", usd=500.0))))
    cap = Warden(workspace, _BUDGET).rent(_LAMBDA, max_usd=1.0, ledger=ledger)
    assert (cap.provider, cap.limit_usd, _spooled(workspace)) == ("lambda", 40.0, 0)


@pytest.mark.parametrize(
    ("policy", "billed", "days_ago", "grants"),
    [
        pytest.param(_BUDGET, 0.0, 1.0, Grants(), id="nothing-billed-yet"),
        pytest.param(_BUDGET, 5.0, 1.0, Grants(), id="billed-up-to-the-budget-exactly"),
        pytest.param(_BUDGET, 8.0, 8.0, Grants(), id="billed-before-the-window"),
        pytest.param(
            _BUDGET,
            8.0,
            1.0,
            Grants(grants=(live("vast", act=Act.RENT),)),
            id="the-owner-granted-it",
        ),
        pytest.param(Policy(), 1000.0, 1.0, Grants(), id="no-budget-declared"),
    ],
)
def test_a_rent_within_the_budget_or_granted_goes_ahead_and_asks_nothing(
    workspace: Path, ledger: Ledger, policy: Policy, billed: float, days_ago: float, grants: Grants
) -> None:
    granted(workspace, grants)
    ledger.record(
        Observation(provider="vast", t_submit=time.time() - days_ago * _DAY, billed_usd=billed)
    )
    Warden(workspace, policy).rent(_VAST, max_usd=5.0, ledger=ledger)
    assert _spooled(workspace) == 0


@pytest.mark.parametrize(
    "grants",
    [
        pytest.param(Grants(), id="no-grant"),
        pytest.param(
            Grants(grants=(live("runpod", act=Act.RENT),)), id="a-grant-on-another-provider"
        ),
        pytest.param(Grants(grants=(live("vast", act=Act.PUSH),)), id="a-grant-of-another-act"),
    ],
)
def test_a_rent_past_the_budget_is_refused_and_asked_unless_the_owner_granted_it(
    workspace: Path, ledger: Ledger, grants: Grants
) -> None:
    granted(workspace, grants)
    ledger.record(Observation(provider="runpod", t_submit=time.time() - _DAY, billed_usd=8.0))
    with pytest.raises(MissionError, match=r"billed \$8\.00 of the \$10\.00 .* may cost \$5\.00"):
        Warden(workspace, _BUDGET).rent(_VAST, max_usd=5.0, ledger=ledger)
    assert _spooled(workspace) == 1


@pytest.mark.parametrize(
    "services",
    [
        pytest.param(SERVICES, id="the-registry-as-declared"),
        pytest.param(_on("codex"), id="codex-running-with-its-own-flags"),
        pytest.param(_on("sink"), id="a-child-no-enforcer-bounds"),
    ],
)
def test_a_launch_whose_agent_children_carry_their_enforcers_goes_ahead_unasked(
    workspace: Path, services: Sequence[Service]
) -> None:
    Warden(workspace, Policy()).launch(services)
    assert _spooled(workspace) == 0


@pytest.mark.parametrize(
    ("services", "lacking"),
    [
        pytest.param(
            _on("claude"), r"--settings \S+/serve/\w+/claude\.json", id="claude-without-settings"
        ),
        pytest.param(
            _on("codex", command=(*_CODEX, "--yolo")),
            "the bound `--yolo` lifts",
            id="codex-told-to-bypass-everything",
        ),
        pytest.param(
            _on("codex", command=(*_CODEX, "-c", "sandbox_mode=danger-full-access")),
            "the bound `-c sandbox_mode=danger-full-access` lifts",
            id="codex-unsandboxed-again-later",
        ),
        pytest.param(
            _on("codex", command=(*_CODEX, "--config=approval_policy='never'")),
            "the bound `--config approval_policy='never'` lifts",
            id="codex-never-asking-in-the-long-spelling",
        ),
        pytest.param(
            _on("codex", command=(*_CODEX, "-sdanger-full-access")),
            "the bound `-sdanger-full-access` lifts",
            id="codex-sandbox-flag-glued-to-its-value",
        ),
        pytest.param(
            _on("claude", command=("claude", "remote-control", "--dangerously-skip-permissions")),
            r"--settings \S+; .*; the bound `--dangerously-skip-permissions` lifts",
            id="claude-told-to-skip-permissions",
        ),
        pytest.param(
            _on("codex", command=("codex", "remote-control", "-c", "approval_policy=on-request")),
            "-c sandbox_mode=workspace-write",
            id="codex-without-its-sandbox",
        ),
        pytest.param(
            _on(
                "codex",
                command=(
                    "codex",
                    "remote-control",
                    "-c",
                    "approval_policy=on-request",
                    "sandbox_mode=workspace-write",
                ),
            ),
            "-c sandbox_mode=workspace-write",
            id="codex-with-the-sandbox-value-loose",
        ),
    ],
)
def test_a_launch_of_an_agent_child_missing_its_enforcer_is_refused(
    workspace: Path, services: Sequence[Service], lacking: str
) -> None:
    with pytest.raises(MissionError, match=f"without {lacking}"):
        Warden(workspace, Policy()).launch(services)


def test_the_doctor_row_says_both_agent_enforcers_wait_while_their_children_are_off(
    workspace: Path,
) -> None:
    row = Enforcement(workspace, SERVICES).section({}, allowed=_ALLOWED, at=datetime.now(UTC))
    off = "codex: not configured (children off); claude: not configured (children off)"
    assert row.detail.startswith(off)
    assert (row.section, row.verdict) == ("policy", Verdict.WARN)


@pytest.mark.parametrize(
    ("settings", "mode", "said"),
    [
        pytest.param(
            claude_settings(), 0o600, "claude: in place", id="every-denial-and-bypass-refused"
        ),
        pytest.param(
            claude_settings(_CLAUDE_DENIES[1:]),
            0o600,
            f"claude: lacks {_CLAUDE_DENIES[0]}",
            id="one-denial-missing",
        ),
        pytest.param(
            json.dumps({"permissions": {"deny": list(_CLAUDE_DENIES)}}),
            0o600,
            "(permissions.disableBypassPermissionsMode: Field required)",
            id="bypass-left-open",
        ),
        pytest.param(
            json.dumps({"permissions": {"deny": list(_CLAUDE_DENIES), "bypass": "disable"}}),
            0o600,
            "(permissions.disableBypassPermissionsMode: Field required)",
            id="bypass-refused-in-a-spelling-claude-never-reads",
        ),
        pytest.param(
            claude_settings(" ".join(_CLAUDE_DENIES)),
            0o600,
            "(permissions.deny: Input should be a valid array)",
            id="every-denial-in-one-string",
        ),
        pytest.param(
            claude_settings(allow=["Bash(git status)", "Bash(*)"]),
            0o600,
            "it allows Bash(*), which runs past every denial",
            id="an-allow-of-every-command",
        ),
        pytest.param(
            claude_settings(allow=["Bash(codex:*)", "Bash(scr*)"]),
            0o600,
            "it allows Bash(codex:*), Bash(scr*), which runs past every denial",
            id="an-allow-of-another-agent-or-a-pty",
        ),
        pytest.param(claude_settings(), 0o644, "is open to others", id="a-file-open-to-others"),
    ],
)
def test_a_running_claude_child_is_judged_by_the_owners_settings_file_it_names(
    workspace: Path, settings: str, mode: int, said: str
) -> None:
    path = Enforcement(workspace, SERVICES).settings_file(_CLAUDE).path
    path.parent.mkdir(parents=True)
    path.write_text(settings, encoding="utf-8")
    path.chmod(mode)
    services = _on("claude", command=("claude", "remote-control", "--settings", str(path)))
    row = Enforcement(workspace, services).section({}, allowed=_ALLOWED, at=datetime.now(UTC))
    assert said in row.detail


def test_a_running_child_declaring_a_dotenv_name_fails_the_allowlist(workspace: Path) -> None:
    (workspace / ".env").write_text("VAST_API_KEY=never-read\n", encoding="utf-8")
    services = _on("sink", environment=(("VAST_API_KEY", "x"),))
    state = Enforcement(workspace, services).allowlist(_ALLOWED)
    assert (state.verdict, state.said) == (
        Verdict.FAIL,
        "a child declares VAST_API_KEY, which the .env defines",
    )


@pytest.mark.parametrize(
    ("profiles", "grants", "verdict", "said"),
    [
        pytest.param(
            {"lambda": _LAMBDA},
            Grants(cards=(card("vultr", usd=50.0),)),
            Verdict.WARN,
            "lambda refused (lambda bills a card after the fact, and that card's limit is the "
            f"only cap on the bill: the owner declares it with `{Project().name} serve grant card "
            "lambda --usd N` from a terminal, or rent from a prepaid or threshold-capped "
            "provider)",
            id="a-card-with-no-limit",
        ),
        pytest.param(
            {"lambda": _LAMBDA, "vast": _VAST},
            Grants(cards=(card(usd=100.0),)),
            Verdict.PASS,
            "lambda card $100, vast prepaid",
            id="every-provider-capped",
        ),
        pytest.param(
            {"gold": HostProfile(kind="ssh")},
            Grants(),
            Verdict.PASS,
            "no host rents from a provider",
            id="only-owned-hosts",
        ),
    ],
)
def test_the_caps_name_each_providers_bound_and_warn_of_a_card_with_no_limit(
    workspace: Path,
    profiles: dict[str, HostProfile],
    grants: Grants,
    verdict: Verdict,
    said: str,
) -> None:
    granted(workspace, grants)
    state = Enforcement(workspace, SERVICES).caps(profiles)
    assert (state.verdict, state.said) == (verdict, said)


@pytest.mark.parametrize(
    ("deny", "verdict", "said"),
    [
        pytest.param(
            list(DENIED),
            Verdict.PASS,
            f"{len(DENIED)} of N2's {len(DENIED)} rules in .agents/settings.json",
            id="every-rule-held",
        ),
        pytest.param(
            list(DENIED[1:]),
            Verdict.WARN,
            f"{len(DENIED) - 1} of N2's {len(DENIED)} rules in .agents/settings.json, the owner "
            "installs N2",
            id="one-rule-missing",
        ),
        pytest.param(
            " ".join(DENIED),
            Verdict.WARN,
            ".agents/settings.json: Input should be a valid array",
            id="every-rule-in-one-string",
        ),
    ],
)
def test_the_deny_rules_of_n2_pass_only_once_the_agents_settings_list_them_all(
    workspace: Path, deny: list[str] | str, verdict: Verdict, said: str
) -> None:
    agents = workspace / ".agents" / "settings.json"
    agents.parent.mkdir()
    agents.write_text(json.dumps({"permissions": {"deny": deny}}), encoding="utf-8")
    state = Enforcement(workspace, SERVICES).denied()
    assert (state.verdict, state.said) == (verdict, said)


@pytest.mark.parametrize(
    ("config", "verdict", "said"),
    [
        pytest.param(
            'sandbox_mode = "danger-full-access"\napproval_policy = "on-request"\n',
            Verdict.FAIL,
            "codex user: sandbox_mode = 'danger-full-access' in ",
            id="codex-unsandboxed",
        ),
        pytest.param(
            'sandbox_mode = "workspace-write"\napproval_policy = "never"\n',
            Verdict.FAIL,
            "codex user: approval_policy = 'never' in ",
            id="codex-never-asking",
        ),
        pytest.param(
            _SANDBOXED, Verdict.WARN, "codex user: sandboxed and asking", id="codex-sandboxed"
        ),
    ],
)
def test_a_codex_user_config_without_sandbox_or_asking_fails_the_row(
    workspace: Path, home: Path, config: str, verdict: Verdict, said: str
) -> None:
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "config.toml").write_text(config, encoding="utf-8")
    row = Enforcement(workspace, SERVICES).section({}, allowed=_ALLOWED, at=datetime.now(UTC))
    assert (row.verdict, said in row.detail) == (verdict, True)


def test_a_grants_file_that_holds_no_grants_fails_the_row_and_refuses_every_push(
    workspace: Path,
) -> None:
    grants_file = granted(workspace, Grants(grants=(live(),)))
    grants_file.path.write_text('{"grants": [{"act": "push"}]', encoding="utf-8")
    row = Enforcement(workspace, SERVICES).grants(at=datetime.now(UTC))
    with pytest.raises(MissionError, match="is no grants file .*; move it aside"):
        Warden(workspace, Policy()).push(_TARGET)
    said = row.said.startswith(f"{grants_file.path} is no grants file")
    moved = f"mv {grants_file.path} {grants_file.path}.aside"
    assert (row.verdict, said, row.fix, _spooled(workspace)) == (Verdict.FAIL, True, moved, 0)


def test_a_grants_file_open_to_others_fails_the_row_with_the_command_that_closes_it(
    workspace: Path,
) -> None:
    grants_file = granted(workspace, Grants(grants=(live(),)))
    os.chmod(grants_file.path, 0o640)
    row = Enforcement(workspace, SERVICES).section({}, allowed=_ALLOWED, at=datetime.now(UTC))
    assert (row.verdict, row.fix) == (Verdict.FAIL, f"chmod 600 {grants_file.path}")


def test_the_live_grants_are_listed_in_the_row_with_when_each_ends(workspace: Path) -> None:
    ended = grant(
        granted=datetime.now(UTC) - timedelta(hours=2),
        until=datetime.now(UTC) - timedelta(hours=1),
    )
    granted(workspace, Grants(grants=(live(), ended)))
    state = Enforcement(workspace, SERVICES).grants(at=datetime.now(UTC))
    assert state.said.startswith("1 live (push me/repo:mb/* until ") and state.said.endswith(
        ") of 2"
    )
