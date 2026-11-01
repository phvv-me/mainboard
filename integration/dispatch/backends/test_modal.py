"""A Modal sandbox is sent the timeout its budget buys, so Modal ends it at the cap."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import ExecutionPlan, MissionError
from mainboard.core.constants.units import SECONDS_PER_HOUR
from mainboard.dispatch import Resources
from mainboard.dispatch.backends import ModalBackend, modal
from mainboard.manifest import HostProfile

_DEAREST_USD_HR = 0.001972 * SECONDS_PER_HOUR
_DAY = 24 * SECONDS_PER_HOUR


def _submitted(resources: Resources) -> tuple[int, Mock]:
    """The timeout `submit` sent Modal's `Sandbox.create`, and the allocation it began first."""
    sent: dict[str, str | int | None] = {}

    def create(*command: str, **kwargs: str | int | None) -> SimpleNamespace:
        sent.update(kwargs)
        return SimpleNamespace(object_id="sb-1")

    double = SimpleNamespace(
        Image=SimpleNamespace(debian_slim=lambda: "image"),
        App=SimpleNamespace(lookup=lambda name, create_if_missing: name),
        Sandbox=SimpleNamespace(create=create),
    )
    allocation = Mock()
    allocation.label = "mb-test"
    plan = ExecutionPlan(host="modal", profile=HostProfile(kind="modal"), env="default")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modal, "_modal", lambda: double)
        ModalBackend().submit(
            plan, "true", resources, allocation=allocation, cap=ModalBackend.cap(card_usd=0.0)
        )
    timeout = sent["timeout"]
    assert isinstance(timeout, int)
    return timeout, allocation


@given(
    max_usd=st.floats(0.5, 500.0),
    walltime=st.sampled_from([None, "00:05:00", "01:00:00", "30:00:00"]),
    gpus=st.integers(0, 8),
)
def test_the_timeout_never_lets_the_dearest_card_bill_past_max_usd(
    max_usd: float, walltime: str | None, gpus: int
) -> None:
    resources = Resources(max_usd=max_usd, walltime=walltime, gpus=gpus)
    try:
        timeout, allocation = _submitted(resources)
    except MissionError as refused:
        assert "exceeds budget" in str(refused)
        return
    billed = timeout * _DEAREST_USD_HR * max(gpus, 1) / SECONDS_PER_HOUR
    assert (0 < timeout <= _DAY) and billed <= max_usd
    lease = allocation.begin.call_args.kwargs["lease"]
    assert (lease.max_usd, lease.cap) == (max_usd, ModalBackend.cap(card_usd=0.0))
    assert not walltime or timeout == min(resources.walltime_s, _DAY)


def test_a_sandbox_without_a_walltime_ends_when_the_budget_does() -> None:
    timeout, _ = _submitted(Resources(max_usd=1.0))
    assert 0 < timeout < SECONDS_PER_HOUR / _DEAREST_USD_HR
