import sys
from decimal import Decimal
from typing import NoReturn

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import Delivery, ModalBackend
from mainboard.dispatch.backends.modal import declared_credit
from mainboard.dispatch.schedulers import JobState, Resources
from mainboard.manifest import Container

from .conftest import FakeModal, ModalFault, environment, plan


@pytest.fixture(autouse=True)
def _credit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_CREDIT_USD", raising=False)


# --- the lazy `modal` import seam ---


def test_a_missing_modal_extra_raises_a_mission_error_naming_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)

    def broken_import(name: str) -> NoReturn:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mainboard.dispatch.backends.modal.import_module", broken_import)
    with pytest.raises(MissionError, match="uv add modal"):
        ModalBackend().cancel("sb-0")


# --- submit ---


def test_submit_refuses_before_any_network_call_when_budget_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    with pytest.raises(MissionError, match="max-usd"):
        ModalBackend().submit(plan(), "echo hi", Resources())
    assert "modal" not in sys.modules


def test_submit_uses_the_default_debian_image_for_a_bare_plan(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    sandbox = fake_modal.sandboxes[handle]
    assert sandbox.kwargs["image"].kind == "debian_slim"
    assert sandbox.entrypoint == ("bash", "-c", "echo hi")


def test_submit_uses_the_container_image_for_a_containerized_plan(fake_modal: FakeModal) -> None:
    containerized = plan(container=Container(image="ubuntu:24.04"))
    handle = ModalBackend().submit(containerized, "echo hi", Resources(max_usd=1.0))
    assert fake_modal.sandboxes[handle].kwargs["image"].ref == "ubuntu:24.04"


@pytest.mark.parametrize(
    ("gpus", "gpu_name", "expected"),
    [
        (0, "", None),
        (0, "H100", None),
        (2, "", "2"),
        (1, "A100", "A100"),
        (3, "A100", "A100:3"),
    ],
)
def test_submit_maps_gpus_and_gpu_name_onto_the_sandboxs_gpu_kwarg(
    fake_modal: FakeModal, gpus: int, gpu_name: str, expected: str | None
) -> None:
    handle = ModalBackend().submit(
        plan(), "echo hi", Resources(max_usd=1.0, gpus=gpus, gpu_name=gpu_name)
    )
    assert fake_modal.sandboxes[handle].kwargs["gpu"] == expected


def test_submit_converts_walltime_to_a_timeout_in_seconds_when_set(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0, walltime="00:10:00"))
    assert fake_modal.sandboxes[handle].kwargs["timeout"] == 600


def test_submit_omits_the_timeout_when_walltime_is_unset(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    assert "timeout" not in fake_modal.sandboxes[handle].kwargs


def test_submit_returns_the_sandbox_object_id(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    assert handle == next(iter(fake_modal.sandboxes))


# --- state ---


@pytest.mark.parametrize(
    ("poll_result", "verdict"),
    [(None, "running"), (0, "ok"), (137, "failed")],
)
def test_state_maps_the_sandboxs_poll_result_onto_a_verdict(
    fake_modal: FakeModal, poll_result: int | None, verdict: str
) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    fake_modal.sandboxes[handle].poll_result = poll_result
    state = ModalBackend().state(handle)
    assert isinstance(state, JobState)
    assert state.exit_code == poll_result
    assert state.verdict == verdict


# --- logs / cancel / deliver ---


def test_logs_reads_the_sandboxs_captured_stdout(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    assert ModalBackend().logs(handle) == "sandbox output"


def test_cancel_terminates_the_sandbox(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    ModalBackend().cancel(handle)
    assert fake_modal.sandboxes[handle].terminated is True


def test_the_declared_delivery_gap_names_volumes_and_the_path_asked_for() -> None:
    advice = ModalBackend().refusal(Delivery, handle="sb-0", path="out/results.json")
    assert advice == (
        "modal backend cannot deliver 'out/results.json' yet; mount a modal Volume at submit "
        "time and pull it by hand until that path lands"
    )


# --- standing, first preference: the cycle budget Modal itself keeps ---


def test_standing_prefers_the_cycle_budget_less_what_the_cycle_has_used(
    fake_modal: FakeModal,
) -> None:
    fake_modal.environments.items = [environment("main", default=True, budget=50.0, used=12.5)]
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd == pytest.approx(37.5)
    assert standing.note == "budget, $50.00 for main less $12.50 used this cycle"


def test_standing_reads_the_default_environments_budget_ahead_of_any_other(
    fake_modal: FakeModal,
) -> None:
    fake_modal.environments.items = [
        environment("staging", budget=10.0, used=1.0),
        environment("main", default=True, budget=50.0, used=12.5),
    ]
    assert ModalBackend().standing().note.endswith("for main less $12.50 used this cycle")


def test_a_workspace_that_never_set_a_budget_falls_through_to_the_derivation(
    fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero budget is what an unbudgeted workspace really answers, not a zero balance."""
    monkeypatch.setenv("MODAL_CREDIT_USD", "30")
    assert ModalBackend().standing().note.startswith("derived,")


def test_a_workspace_without_the_budget_feature_degrades_quietly_to_the_derivation(
    fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODAL_CREDIT_USD", "30")
    fake_modal.environments.refusal = ModalFault(
        "Environment budgets are not enabled for this workspace"
    )
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.note.startswith("derived,")
    assert "budget" not in standing.note


# --- standing, second preference: the declared credit less the metered spend ---


def test_declared_credit_reads_the_env_and_defaults_to_nothing_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert declared_credit() == pytest.approx(0.0)
    monkeypatch.setenv("MODAL_CREDIT_USD", "30.50")
    assert declared_credit() == pytest.approx(30.5)


def test_declared_credit_refuses_a_value_that_is_not_a_dollar_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_CREDIT_USD", "thirty bucks")
    with pytest.raises(MissionError, match="MODAL_CREDIT_USD"):
        declared_credit()


def test_standing_derives_the_balance_from_the_declaration_less_this_cycles_spend(
    fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODAL_CREDIT_USD", "30")
    fake_modal.billing.reply.metered_cost = Decimal("4.75")
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd == pytest.approx(25.25)
    assert standing.note == "derived, $30.00 declared less $4.75 metered in 2026-08"
    assert fake_modal.sandboxes == {}


def test_standing_without_a_declaration_reports_the_spend_and_names_the_variable(
    fake_modal: FakeModal,
) -> None:
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd is None
    assert standing.note == "$1.25 metered in 2026-08, set MODAL_CREDIT_USD to derive a balance"


def test_standing_keeps_the_row_keyed_when_modal_refuses_the_billing_summary(
    fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODAL_CREDIT_USD", "30")
    fake_modal.billing.refusal = ModalFault("Rate limit exceeded")
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd is None
    assert "Rate limit exceeded" in standing.note


def test_standing_without_a_configured_token_names_the_command_that_mints_one(
    fake_modal: FakeModal,
) -> None:
    fake_modal.config.config["token_id"] = ""
    standing = ModalBackend().standing()
    assert standing.keyed is False
    assert "modal token new" in standing.note


def test_standing_without_the_optional_extra_installed_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)

    def broken_import(name: str) -> NoReturn:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("mainboard.dispatch.backends.modal.import_module", broken_import)
    standing = ModalBackend().standing()
    assert standing.keyed is False
    assert "uv add modal" in standing.note
