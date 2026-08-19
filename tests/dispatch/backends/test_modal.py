import sys
from typing import NoReturn

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import ModalBackend
from mainboard.dispatch.schedulers import JobState, Resources
from mainboard.manifest import Container

from .conftest import FakeModal, plan

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


def test_deliver_raises_a_not_yet_mission_error_naming_volumes() -> None:
    with pytest.raises(MissionError, match="Volume"):
        ModalBackend().deliver("sb-0", path="out/results.json")


# --- standing ---


def test_standing_reads_the_configured_token_pair_without_calling_out(
    fake_modal: FakeModal,
) -> None:
    standing = ModalBackend().standing()
    assert standing.keyed is True
    assert standing.credit_usd is None
    assert "credit unavailable" in standing.note
    assert fake_modal.sandboxes == {}


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
