from types import SimpleNamespace
from urllib.request import Request

import pytest

from mainboard import MissionError
from mainboard.dispatch.backends import (
    Backend,
    HpcAiBackend,
    ModalBackend,
    ProviderBackend,
    VastBackend,
    base,
    http_transport,
    require_budget,
    route,
)
from mainboard.dispatch.schedulers import Resources
from mainboard.manifest import HostProfile

from ..conftest import profile
from .conftest import FakeTransport, hpc_ai_backend, vast_backend

# --- route ---


@pytest.mark.parametrize("kind", ["ssh", "pbs", "slurm", "local"])
def test_route_keeps_ssh_family_kinds_on_the_scheduler_path(kind: str) -> None:
    assert route(profile(kind=kind)) == "ssh-family"


def test_route_resolves_a_registered_provider_backend_by_kind() -> None:
    assert route(profile(kind="modal")) is ModalBackend
    assert route(profile(kind="hpc-ai")) is HpcAiBackend
    assert route(profile(kind="vast")) is VastBackend


def test_route_raises_a_mission_error_naming_known_kinds_for_an_unregistered_kind() -> None:
    with pytest.raises(MissionError) as excinfo:
        route(profile(kind="ec2"))
    assert "ec2" in str(excinfo.value)
    assert "modal" in str(excinfo.value)
    assert "hpc-ai" in str(excinfo.value)


# --- Backend protocol ---


def test_modal_backend_satisfies_the_backend_protocol_structurally() -> None:
    assert isinstance(ModalBackend(), Backend)


def test_hpc_ai_backend_satisfies_the_backend_protocol_structurally() -> None:
    assert isinstance(hpc_ai_backend(transport=FakeTransport()), Backend)


def test_vast_backend_satisfies_the_backend_protocol_structurally() -> None:
    assert isinstance(vast_backend(), Backend)


def test_an_unrelated_object_does_not_satisfy_the_backend_protocol() -> None:
    assert not isinstance(object(), Backend)


def test_provider_backend_root_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ProviderBackend()  # type: ignore[abstract]  reason=proving the abstract contract is enforced since=2026-08-17


# --- require_budget ---


def test_http_transport_hands_the_prepared_request_to_urllib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Request] = []

    def fake_urlopen(request: Request) -> SimpleNamespace:
        seen.append(request)
        return SimpleNamespace(status=200, read=lambda: b"{}")

    monkeypatch.setattr(base, "urlopen", fake_urlopen)
    request = Request("https://example.test/probe")
    assert http_transport(request).read() == b"{}"
    assert seen == [request]


# --- require_budget ---


def test_require_budget_raises_when_max_usd_is_unset() -> None:
    with pytest.raises(MissionError, match="max-usd"):
        require_budget(Resources())


def test_require_budget_passes_when_max_usd_is_set() -> None:
    require_budget(Resources(max_usd=1.0))


def test_route_keeps_auto_on_the_ssh_family() -> None:
    assert route(HostProfile()) == "ssh-family"
