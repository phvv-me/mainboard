import sys
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import NoReturn

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import Delivery, ModalBackend
from mainboard.dispatch.backends.modal import cycle_month, declared_credit
from mainboard.dispatch.evidence import framing, staging
from mainboard.dispatch.vocabulary import Resources
from mainboard.manifest import Container

from .support import FakeModal, ModalFault, environment, plan

# What a sandbox's `poll()` says, and the verdict a post-mortem reads it as. A sandbox that has
# not exited polls None, so a live run is the absence of an exit code rather than a state string.
_VERDICTS = {None: "running", 0: "ok", 137: "failed"}

# A resource request as `gpu=` spells it: no count is no GPU whatever card was named, a count
# with no name is a bare number, one named card is its own name, and more than one is `name:count`.
_GPU_SPECS = {
    (0, ""): None,
    (0, "H100"): None,
    (2, ""): "2",
    (1, "A100"): "A100",
    (3, "A100"): "A100:3",
}


def broken_import(name: str) -> NoReturn:
    """Stand in for `import_module` on a machine where the optional extra was never installed."""
    raise ModuleNotFoundError(name)


def gpu_kwarg(fake: FakeModal, *, gpus: int, gpu_name: str) -> str | None:
    """The `gpu=` value a submit under `gpus` and `gpu_name` hands the sandbox it creates."""
    handle = ModalBackend().submit(
        plan(), "echo hi", Resources(max_usd=1.0, gpus=gpus, gpu_name=gpu_name)
    )
    return fake.sandboxes[handle].kwargs["gpu"]


@pytest.fixture(autouse=True)
def _credit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODAL_CREDIT_USD", raising=False)


def test_a_missing_modal_extra_refuses_every_verb_with_the_command_that_installs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing SDK never breaks a survey row.

    `modal` is an optional extra, so every call goes through the lazy accessor, and a survey
    listing the row must not raise where a dispatched verb rightly does.
    """
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    monkeypatch.setattr("mainboard.dispatch.backends.modal.import_module", broken_import)
    with pytest.raises(MissionError, match="uv add modal"):
        ModalBackend().cancel("sb-0")
    standing = ModalBackend().standing()
    assert standing.keyed is False
    assert "uv add modal" in standing.note


def test_submit_refuses_before_the_sdk_is_even_imported_when_the_budget_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    with pytest.raises(MissionError, match="max-usd"):
        ModalBackend().submit(plan(), "echo hi", Resources())
    assert "modal" not in sys.modules


@pytest.mark.parametrize(
    ("overrides", "image", "walltime", "timeout"),
    [
        pytest.param(
            {},
            {"kind": "debian_slim"},
            None,
            None,
            id="a-bare-plan-on-the-default-image-and-no-deadline",
        ),
        pytest.param(
            {"container": Container(image="ubuntu:24.04")},
            {"kind": "registry", "ref": "ubuntu:24.04"},
            "00:10:00",
            600,
            id="a-containerized-plan-under-a-walltime",
        ),
    ],
)
def test_submit_creates_one_sandbox_whose_entrypoint_image_and_timeout_are_the_plans(
    fake_modal: FakeModal,
    overrides: dict[str, Container],
    image: dict[str, str],
    walltime: str | None,
    timeout: int | None,
) -> None:
    """The command IS the sandbox entrypoint, so its lifetime, exit code and stdout are the job's.

    A detached exec would leave the sandbox idling as running forever with empty logs, which is
    what the first live submit found.
    """
    handle = ModalBackend().submit(
        plan(**overrides), "echo hi", Resources(max_usd=1.0, walltime=walltime)
    )
    (created,) = fake_modal.sandboxes.values()
    assert handle == created.object_id
    script = f"{staging()}\necho hi\nstatus=$?\n{framing()}\nexit $status"
    assert created.entrypoint == ("bash", "-c", script)
    assert vars(created.kwargs["image"]) == image
    assert created.kwargs["app"].name == "mainboard"
    assert created.kwargs.get("timeout") == timeout
    assert ("timeout" in created.kwargs) is (timeout is not None)


def test_submit_maps_gpus_and_gpu_name_onto_the_sandboxs_gpu_kwarg(fake_modal: FakeModal) -> None:
    spelled = {
        request: gpu_kwarg(fake_modal, gpus=request[0], gpu_name=request[1])
        for request in _GPU_SPECS
    }
    assert spelled == _GPU_SPECS


def test_state_maps_the_sandboxs_poll_result_onto_a_verdict(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    read = {}
    for code in _VERDICTS:
        fake_modal.sandboxes[handle].poll_result = code
        state = ModalBackend().state(handle)
        read[code] = (state.handle, state.exit_code, state.verdict)
    assert read == {code: (handle, code, verdict) for code, verdict in _VERDICTS.items()}


def test_logs_reads_the_sandboxs_captured_stdout(fake_modal: FakeModal) -> None:
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    assert ModalBackend().logs(handle) == "sandbox output"


def test_cancel_terminates_the_sandbox_and_tolerates_one_modal_already_forgot(
    fake_modal: FakeModal,
) -> None:
    """A second cancel of the same sandbox is part of the design.

    A sweep cancels every run it settles, so the same sandbox is cancelled more than once and
    the second call walks straight into the `NotFoundError` a gone id raises.
    """
    handle = ModalBackend().submit(plan(), "echo hi", Resources(max_usd=1.0))
    ModalBackend().cancel(handle)
    assert fake_modal.sandboxes[handle].terminated is True
    ModalBackend().cancel("sb-gone")
    assert "sb-gone" not in fake_modal.sandboxes


def test_the_declared_delivery_gap_names_volumes_and_the_path_asked_for() -> None:
    advice = ModalBackend().refusal(Delivery, handle="sb-0", path="out/results.json")
    assert advice == (
        "modal backend cannot deliver 'out/results.json' yet; mount a modal Volume at submit "
        "time and pull it by hand until that path lands"
    )


def test_standing_prefers_the_default_environments_cycle_budget(fake_modal: FakeModal) -> None:
    """The default environment's budget is read ahead of any other.

    It is the closest thing Modal keeps to a balance, and the default environment is where a
    sandbox lands when nothing names another.
    """
    fake_modal.environments.items = [
        environment("staging", budget=10.0, used=1.0),
        environment("main", default=True, budget=50.0, used=12.5),
    ]
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd == pytest.approx(37.5)
    assert standing.note == "budget, $50.00 for main less $12.50 used this cycle"


@pytest.mark.parametrize(
    "refusal",
    [None, "Environment budgets are not enabled for this workspace"],
    ids=["a-workspace-that-never-set-a-budget", "a-workspace-without-the-budget-feature"],
)
def test_standing_falls_through_to_the_derivation_when_no_budget_answers(
    fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch, refusal: str | None
) -> None:
    """An unbudgeted or unlicensed workspace prints no budget at all.

    A zero budget is what an unbudgeted workspace really answers, not a zero balance, and a
    workspace without the team feature refuses the read outright. Neither is worth printing.
    """
    monkeypatch.setenv("MODAL_CREDIT_USD", "30")
    if refusal:
        fake_modal.environments.refusal = ModalFault(refusal)
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.note.startswith("derived,")
    assert "budget" not in standing.note


@pytest.mark.parametrize(
    ("start", "month"),
    [
        (datetime(2026, 8, 1, 0, 0), "2026-08"),
        (datetime(2026, 9, 1, 8, 59, tzinfo=timezone(timedelta(hours=9))), "2026-08"),
        (datetime(2026, 8, 31, 23, 59, tzinfo=UTC), "2026-08"),
    ],
    ids=["a naive stamp read as UTC", "an aware stamp converted across the boundary", "utc"],
)
def test_the_billing_cycle_month_is_pinned_to_utc_wherever_the_command_was_typed(
    start: datetime, month: str
) -> None:
    """A cycle boundary is a UTC fact, so a JST morning still bills into the UTC month."""
    assert cycle_month(start) == month


def test_declared_credit_reads_the_env_and_refuses_what_is_not_a_dollar_amount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The starting credit comes from the person who bought it.

    Modal reports what a cycle has cost and never what is left.
    """
    assert declared_credit() == pytest.approx(0.0)
    monkeypatch.setenv("MODAL_CREDIT_USD", "30.50")
    assert declared_credit() == pytest.approx(30.5)
    monkeypatch.setenv("MODAL_CREDIT_USD", "thirty bucks")
    with pytest.raises(MissionError, match="MODAL_CREDIT_USD"):
        declared_credit()


@pytest.mark.parametrize(
    ("declared", "metered", "refusal", "credit", "note"),
    [
        pytest.param(
            "30",
            "4.75",
            None,
            25.25,
            "derived, $30.00 declared less $4.75 metered in 2026-08",
            id="a-declaration-and-a-summary-modal-answered",
        ),
        pytest.param(
            "",
            "1.25",
            None,
            None,
            "$1.25 metered in 2026-08, set MODAL_CREDIT_USD to derive a balance",
            id="a-workspace-that-declared-no-credit",
        ),
        pytest.param(
            "30",
            "4.75",
            "Rate limit exceeded",
            None,
            "modal refused the billing summary, Rate limit exceeded",
            id="a-billing-summary-modal-throttled",
        ),
    ],
)
def test_standing_derives_the_balance_from_the_declaration_less_this_cycles_spend(
    fake_modal: FakeModal,
    monkeypatch: pytest.MonkeyPatch,
    declared: str,
    metered: str,
    refusal: str | None,
    credit: float | None,
    note: str,
) -> None:
    """A derived balance names its own arithmetic.

    Subtracting the metered cost from a declared credit is arithmetic we do rather than a
    balance Modal blessed, so the note carries both figures and the cycle they belong to. A
    throttled read says nothing about whether the provider is usable, so the row stays keyed.
    """
    if declared:
        monkeypatch.setenv("MODAL_CREDIT_USD", declared)
    fake_modal.billing.reply.metered_cost = Decimal(metered)
    if refusal:
        fake_modal.billing.refusal = ModalFault(refusal)
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd == (None if credit is None else pytest.approx(credit))
    assert standing.note == note
    assert fake_modal.sandboxes == {}


def test_standing_without_a_configured_token_names_the_command_that_mints_one(
    fake_modal: FakeModal,
) -> None:
    """An unauthenticated machine is caught before any round trip.

    The token pair is what `modal.Client` itself checks first.
    """
    fake_modal.config.config["token_id"] = ""
    standing = ModalBackend().standing()
    assert standing.keyed is False
    assert "modal token new" in standing.note
