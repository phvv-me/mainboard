import json
import sys
from datetime import UTC, datetime
from decimal import Decimal
from email.message import Message
from types import SimpleNamespace
from typing import TYPE_CHECKING
from urllib.error import HTTPError

import pytest

from mainboard import ExecutionPlan
from mainboard.dispatch.backends import HpcAiBackend, LogSource, ProviderBackend, VastBackend
from mainboard.dispatch.schedulers import JobState, Resources
from mainboard.manifest import Container, HostProfile

if TYPE_CHECKING:
    from collections.abc import Iterator
    from urllib.request import Request

    from mainboard.dispatch.backends.base import Transport

type PlanField = str | HostProfile | Container | dict[str, str] | None
type Reply = dict | str | HTTPError


class BareBackend(ProviderBackend):
    """A backend with the job lifecycle and not one capability beyond it.

    The shape every discovery site has to handle. It declares where its logs would live and says
    nothing at all about delivery, so this one double covers both halves of `refusal`, the advice
    a backend wrote for itself and the plain statement of a gap it never described.
    """

    name = "bare"

    lacks = {LogSource: "bare backend keeps no logs; read {handle}.log on the box instead"}

    def cancel(self, handle: str) -> None:
        self.cancelled = handle

    def state(self, handle: str) -> JobState:
        return JobState(handle=handle, state="finished", exit_code=0, verdict="ok")

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        del plan, command, resources
        return "bare-1"


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


class ModalFault(Exception):
    """A `modal.exception.Error` stand-in, the root every fault the real SDK raises inherits."""


class FakeBilling:
    """A `Workspace.billing` double: one summary a test can reshape, or one fault it can queue.

    The summary mirrors only what `standing` reads of the real dataclass, a `Decimal` metered cost
    and the cycle start it belongs to, since those two are what the derived balance is built from.
    """

    def __init__(self) -> None:
        self.refusal: Exception | None = None
        self.reply = SimpleNamespace(
            metered_cost=Decimal("1.25"), start=datetime(2026, 8, 1, tzinfo=UTC)
        )

    def summary(self) -> SimpleNamespace:
        if self.refusal:
            raise self.refusal
        return self.reply


class FakeEnvironments:
    """A `modal.environments` double: one environment list a test reshapes, or a fault it queues.

    Each item mirrors only the budget fields `standing` reads off a real `EnvironmentListItem`.
    The default answers a zero budget, which is what a workspace that never set one really says.
    """

    def __init__(self) -> None:
        self.refusal: Exception | None = None
        self.items = [environment("main", default=True)]

    def list_environments(self) -> list[SimpleNamespace]:
        if self.refusal:
            raise self.refusal
        return self.items


def environment(
    name: str, *, default: bool = False, budget: float = 0.0, used: float = 0.0
) -> SimpleNamespace:
    """One `EnvironmentListItem` double, named, budgeted, and used to whatever a test needs."""
    return SimpleNamespace(
        name=name, default=default, cycle_budget_dollars=budget, current_cycle_usage=used
    )


class FakeModal(SimpleNamespace):
    """A fully faked `modal` module: only the surface `ModalBackend` actually calls.

    `config.config` mirrors the real module's settings mapping, which is where the SDK itself
    looks for the token pair before its first call; a test blanks an entry to stand for a
    machine nobody ran `modal token new` on. `environments.list_environments` and
    `Workspace.from_context().billing` are the two account reads the SDK offers, reachable here
    through the same `environments` and `billing` the test holds.
    """

    def __init__(self) -> None:
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.billing = FakeBilling()
        self.environments = FakeEnvironments()
        super().__init__(
            config=SimpleNamespace(config={"token_id": "ak-1", "token_secret": "as-1"}),
            environments=self.environments,
            exception=SimpleNamespace(Error=ModalFault),
            Workspace=SimpleNamespace(from_context=lambda: SimpleNamespace(billing=self.billing)),
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
