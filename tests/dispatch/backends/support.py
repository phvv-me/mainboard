import json
from datetime import UTC, datetime
from decimal import Decimal
from email.message import Message
from types import SimpleNamespace
from typing import TYPE_CHECKING
from urllib.error import HTTPError

from mainboard import ExecutionPlan
from mainboard.dispatch.allocation import Allocation
from mainboard.dispatch.backends import (
    HpcAiBackend,
    LogSource,
    ProviderBackend,
    VastBackend,
)
from mainboard.dispatch.vocabulary import JobState, Resources
from mainboard.manifest import Container, HostProfile

from ..support import Naps

if TYPE_CHECKING:
    from urllib.request import Request

    from mainboard.dispatch.backends.base import Transport

type PlanField = str | HostProfile | Container | dict[str, str] | None
type Reply = dict | str | HTTPError


class BareBackend(ProviderBackend):
    """A backend with the job lifecycle and not one capability beyond it.

    It advises where its logs would live and says nothing about delivery, covering both halves
    of `refusal`.
    """

    name = "bare"

    lacks = {LogSource: "bare backend keeps no logs; read {handle}.log on the box instead"}

    def cancel(self, handle: str) -> None:
        self.cancelled = handle

    def state(self, handle: str) -> JobState:
        return JobState(handle=handle, state="finished", exit_code=0, verdict="ok")

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        del plan, command, resources
        allocation.begin()
        return allocation.created("bare-1")


def hpc_ai_backend(
    *, transport: Transport, spot: bool = False, naps: Naps | None = None
) -> HpcAiBackend:
    """An `HpcAiBackend` over an injected transport, its polls never really sleeping."""
    return HpcAiBackend(spot=spot, transport=transport, sleeper=naps or Naps())


def plan(**overrides: PlanField) -> ExecutionPlan:
    """An `ExecutionPlan` for a bare, uncontainerized provider host, with fields overridden."""
    fields: dict[str, PlanField] = {
        "host": "provider-host",
        "profile": HostProfile(kind="modal", root="/repo", sync={"include": ["src"]}),
        "env": "default",
    }
    fields.update(overrides)
    return ExecutionPlan.model_validate(fields)


class FakeTransport:
    """A `Transport` double: records every `Request` and replays queued replies in order.

    A dict answers as a JSON body, a string as raw text (an uploaded log), and an `HTTPError` is
    raised, as urllib reports a 404. Once the queue is empty a DELETE is confirmed and anything
    else answers `{}`.
    """

    def __init__(self, *responses: Reply) -> None:
        self.calls: list[Request] = []
        self.responses: list[Reply] = list(responses)

    def __call__(self, request: Request) -> SimpleNamespace:
        self.calls.append(request)
        default = {"success": True} if request.get_method() == "DELETE" else {}
        reply = self.responses.pop(0) if self.responses else default
        if isinstance(reply, HTTPError):
            raise reply
        body = reply.encode() if isinstance(reply, str) else json.dumps(reply).encode()
        return SimpleNamespace(status=200, read=lambda: body)

    @property
    def bodies(self) -> list[dict]:
        """The JSON body of every request that carried one; a bodiless storage fetch is skipped."""
        return [json.loads(call.data) for call in self.calls if call.data]

    @property
    def urls(self) -> list[str]:
        """The full url of every recorded request, in order."""
        return [call.full_url for call in self.calls]


def refused(status: int, url: str = "https://console.vast.ai/api/v0/instances/7/") -> HTTPError:
    """The fault urllib raises for `status`, queued as a provider refusal."""
    return HTTPError(url, status, "Refused", Message(), None)


def not_found(url: str = "https://console.vast.ai/api/v0/instances/7/") -> HTTPError:
    """A 404 the fake transport raises, the way urllib reports a gone instance or log."""
    return refused(404, url)


def vast_backend(*responses: Reply, spot: bool = False, naps: Naps | None = None) -> VastBackend:
    """A `VastBackend` replaying `responses` one per call, its polls never really sleeping."""
    return VastBackend(spot=spot, transport=FakeTransport(*responses), sleeper=naps or Naps())


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
        self.poll_result: int | None = None
        self.stdout = SimpleNamespace(read=lambda: "sandbox output")
        registry[self.object_id] = self

    def poll(self) -> int | None:
        return self.poll_result

    def terminate(self) -> None:
        self.terminated = True


class ModalFault(Exception):
    """A `modal.exception.Error` stand-in, the root of every real SDK fault."""


class ModalMissing(ModalFault):
    """A `modal.exception.NotFoundError` stand-in, raised by `Sandbox.from_id` for a gone id."""


class FakeBilling:
    """A `Workspace.billing` double: a reshapeable summary (a `Decimal` metered cost and its
    cycle start, all `standing` reads), or a queued fault.
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
    """A `modal.environments` double: a reshapeable list, or a queued fault.

    The default answers a zero budget, as a workspace that never set one really does.
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
    """One `EnvironmentListItem` double carrying the budget fields `standing` reads."""
    return SimpleNamespace(
        name=name, default=default, cycle_budget_dollars=budget, current_cycle_usage=used
    )


class FakeModal(SimpleNamespace):
    """A fake `modal` module, only the surface `ModalBackend` calls.

    `config.config` holds the token pair the SDK checks first; a test blanks one to stand for a
    machine nobody ran `modal token new` on. `environments` and `billing` are the two account
    reads, held by the test.
    """

    def __init__(self) -> None:
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.billing = FakeBilling()
        self.environments = FakeEnvironments()
        super().__init__(
            config=SimpleNamespace(config={"token_id": "ak-1", "token_secret": "as-1"}),
            environments=self.environments,
            exception=SimpleNamespace(Error=ModalFault, NotFoundError=ModalMissing),
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
                from_id=self.sandbox,
            ),
        )

    def sandbox(self, handle: str) -> FakeSandbox:
        """The sandbox `handle` names, raising `NotFoundError` like the real one for a gone id."""
        try:
            return self.sandboxes[handle]
        except KeyError:
            raise ModalMissing(f"no sandbox {handle}") from None
