import json
import sys
from email.message import Message
from types import SimpleNamespace
from typing import TYPE_CHECKING
from urllib.error import HTTPError

import pytest

from mainboard import ExecutionPlan
from mainboard.dispatch.backends import HpcAiBackend, VastBackend
from mainboard.manifest import Container, HostProfile

if TYPE_CHECKING:
    from collections.abc import Iterator
    from urllib.request import Request

    from mainboard.dispatch.backends.base import Transport

type PlanField = str | HostProfile | Container | dict[str, str] | None
type Reply = dict | str | HTTPError


def hpc_ai_backend(*, transport: Transport, spot: bool = False) -> HpcAiBackend:
    """An `HpcAiBackend` for tests, with fixed non-secret credentials and an injected transport."""
    return HpcAiBackend(
        spot=spot,
        transport=transport,
    )


def plan(**overrides: PlanField) -> ExecutionPlan:
    """An `ExecutionPlan` for provider-backend tests, defaulting to a bare, uncontainerized host.

    overrides: `ExecutionPlan` fields to override (`profile`, `container`, ...).
    """
    fields: dict[str, PlanField] = {
        "host": "provider-host",
        "profile": HostProfile(kind="modal", root="/repo", sync={"include": ["src"]}),
        "env": "default",
    }
    fields.update(overrides)
    return ExecutionPlan.model_validate(fields)


class FakeTransport:
    """A `Transport` double: records every `Request` and replays queued replies in order.

    A queued dict answers as a JSON body, a queued string answers as that raw text (the shape an
    uploaded log has), and a queued `HTTPError` is raised rather than returned, which is how
    urllib reports a 404 to its caller.
    """

    def __init__(self, *responses: Reply) -> None:
        self.calls: list[Request] = []
        self.responses: list[Reply] = list(responses)

    def __call__(self, request: Request) -> SimpleNamespace:
        self.calls.append(request)
        reply = self.responses.pop(0) if self.responses else {}
        if isinstance(reply, HTTPError):
            raise reply
        body = reply.encode() if isinstance(reply, str) else json.dumps(reply).encode()
        return SimpleNamespace(status=200, read=lambda: body)


def not_found(url: str = "https://console.vast.ai/api/v0/instances/7/") -> HTTPError:
    """A 404 the fake transport raises, the way urllib reports a gone instance or log."""
    return HTTPError(url, 404, "Not Found", Message(), None)


def vast_backend(*responses: Reply, spot: bool = False) -> VastBackend:
    """A `VastBackend` over a queued-response transport, its log poll never really sleeping."""
    return VastBackend(spot=spot, transport=FakeTransport(*responses), sleeper=lambda _: None)


class FakeSandbox:
    """A `modal.Sandbox` double: tracks its own lifecycle and answers `from_id` by object id."""

    registry: dict[str, FakeSandbox]

    def __init__(
        self,
        registry: dict[str, FakeSandbox],
        *entrypoint: str,
        **kwargs: SimpleNamespace | str | int | None,
    ) -> None:
        self.registry = registry
        self.entrypoint = entrypoint
        self.kwargs = kwargs
        self.object_id = f"sb-{len(registry)}"
        self.terminated = False
        self.exec_calls: list[tuple[str, ...]] = []
        self.poll_result: int | None = None
        self.stdout = SimpleNamespace(read=lambda: "sandbox output")
        registry[self.object_id] = self

    def exec(self, *argv: str) -> SimpleNamespace:
        self.exec_calls.append(argv)
        return SimpleNamespace()

    def poll(self) -> int | None:
        return self.poll_result

    def terminate(self) -> None:
        self.terminated = True


class FakeModal(SimpleNamespace):
    """A fully faked `modal` module: only the surface `ModalBackend` actually calls.

    `config.config` mirrors the real module's settings mapping, which is where the SDK itself
    looks for the token pair before its first call; a test blanks an entry to stand for a
    machine nobody ran `modal token new` on.
    """

    def __init__(self) -> None:
        self.sandboxes: dict[str, FakeSandbox] = {}
        super().__init__(
            config=SimpleNamespace(config={"token_id": "ak-1", "token_secret": "as-1"}),
            Image=SimpleNamespace(
                from_registry=lambda ref: SimpleNamespace(kind="registry", ref=ref),
                debian_slim=lambda: SimpleNamespace(kind="debian_slim"),
            ),
            App=SimpleNamespace(
                lookup=lambda name, create_if_missing=False: SimpleNamespace(name=name)
            ),
            Sandbox=SimpleNamespace(
                create=lambda *entrypoint, **kwargs: FakeSandbox(
                    self.sandboxes, *entrypoint, **kwargs
                ),
                from_id=lambda handle: self.sandboxes[handle],
            ),
        )


@pytest.fixture
def fake_modal() -> Iterator[FakeModal]:
    """A `FakeModal` injected into `sys.modules["modal"]`, restored after the test."""
    fake = FakeModal()
    sys.modules["modal"] = fake  # type: ignore[assignment]  reason=a hermetic double stands in for the real optional package since=2026-08-17
    yield fake
    del sys.modules["modal"]
