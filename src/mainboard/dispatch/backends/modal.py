# `ModalBackend` runs a command inside a Modal Sandbox. `modal` is an optional extra, so every
# call goes through the lazy `_modal` accessor instead of a module-level import.
#
# Modal exposes no account balance, established from its wire contract rather than its docs
# (surveyed 2026-08-19, modal 1.5.4): the shipped `modal_proto` descriptor has one service of 239
# methods over 620 messages, and no field anywhere names a credit, balance, remaining amount or
# prepaid pot. The only account reads are `WorkspaceBillingSummary` (per-cycle `metered_cost`,
# `billed_cost`, and an `adjustments` map whose `Credits` entry is credit *applied* that cycle,
# never credit left) and `EnvironmentList`/`EnvironmentGetBudget`, a spend *cap*. There is no REST
# fallback: every path under api.modal.com answers `application/grpc`, and modal.com/api/* is 404.
#
# So the row asks in order of how much Modal vouches for the figure. First a cycle budget, the
# closest thing to a balance: `modal.environments.list_environments` carries the four budget
# fields of `EnvironmentGetBudget` (`cycle_budget_dollars`, `effective_cycle_spend_limit`,
# `current_cycle_usage`, `spend_limit_reached`) in one call needing no environment id. A workspace
# without the team feature refuses it with `PermissionDeniedError` and one that never set a budget
# answers zero (verified live 2026-08-19); both fall through. Then a balance derived from the
# credit declared once in `MODAL_CREDIT_USD` less this cycle's metered cost, the note saying it is
# derived rather than reported.

import os
from contextlib import suppress
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING

from ...core.errors import MissionError
from ...runtime.job import walltime_seconds
from ..evidence import framing, staging
from ..vocabulary import JobState
from .base import (
    Account,
    Credentials,
    Delivery,
    LogSource,
    ProviderBackend,
    Standing,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ...context.plan import ExecutionPlan
    from ..allocation import Allocation
    from ..vocabulary import Resources

# One shared app for every one-shot sandbox (Modal scopes billing and the dashboard by app).
_APP_NAME = "mainboard"
# Where the workspace declares the prepaid credit Modal will not report.
_CREDIT_VAR = "MODAL_CREDIT_USD"


def _modal() -> ModuleType:
    """The imported `modal` module, raising a clear fix when the optional extra is missing."""
    try:
        return import_module("modal")
    except ModuleNotFoundError:
        raise MissionError(
            "the installed Mainboard tool needs its `modal` extra; reinstall Mainboard "
            "with that extra alongside the extras already in use, then authenticate "
            "with `mainboard run modal token new`"
        ) from None


def cycle_month(start: datetime) -> str:
    """`start` as its billing-cycle month in UTC, a naive stamp read as UTC.

    Modal's cycle boundary is a UTC fact, so the month never depends on the local clock.
    """
    pinned = start if start.tzinfo else start.replace(tzinfo=UTC)
    return pinned.astimezone(UTC).strftime("%Y-%m")


def declared_credit() -> float:
    """The credit `MODAL_CREDIT_USD` declares, 0.0 when none; Modal never says what is left.

    Not a secret, but it lives beside the provider keys in the workspace `.env`, the one file
    every provider's account settings share.
    """
    Credentials().load()
    declared = os.environ.get(_CREDIT_VAR, "")
    if not declared:
        return 0.0
    try:
        return float(declared)
    except ValueError:
        raise MissionError(
            f"{_CREDIT_VAR} must be a dollar amount like `30`, not {declared!r}"
        ) from None


class ModalBackend(ProviderBackend, Account, LogSource):
    """Run a command in a fresh Modal Sandbox, the sandbox's lifetime being the job's.

    Stateless: every call reconnects by id (`modal.Sandbox.from_id`). A sandbox keeps its
    stdout, so logs are real, but its disk dies with it unless a Volume was mounted at create
    time, hence `Delivery` in `lacks`.
    """

    name = "modal"

    lacks = {
        Delivery: "modal backend cannot deliver {path!r} yet; mount a modal Volume at submit "
        "time and pull it by hand until that path lands",
    }

    @staticmethod
    def budgeted(modal: ModuleType) -> Standing | None:
        """The cycle budget less this cycle's usage, None when no budget answers.

        The default environment comes first, where a sandbox lands when nothing names another.

        modal: the imported `modal` module, already known to carry credentials.
        """
        try:
            environments = modal.environments.list_environments()
        except modal.exception.Error:
            return None
        ordered = sorted(environments, key=lambda item: not item.default)
        budget = next((item for item in ordered if item.cycle_budget_dollars), None)
        if budget is None:
            return None
        return Standing(
            keyed=True,
            credit_usd=budget.cycle_budget_dollars - budget.current_cycle_usage,
            note=f"budget, ${budget.cycle_budget_dollars:.2f} for {budget.name} less "
            f"${budget.current_cycle_usage:.2f} used this cycle",
        )

    @staticmethod
    def derived(modal: ModuleType) -> Standing:
        """The declared credit less this cycle's metered spend, or whatever step is missing.

        `Workspace.billing.summary().metered_cost` is the calendar-month cycle's cost before any
        credit or discount. The note carries both figures and the cycle, and a workspace that
        declares nothing still gets the spend. A Modal fault (commonly a rate-limited summary)
        stays a note on a keyed row, since a throttled read says nothing about usability.

        modal: the imported `modal` module, already known to carry credentials.
        """
        declared = declared_credit()
        try:
            summary = modal.Workspace.from_context().billing.summary()
        except modal.exception.Error as refused:
            return Standing(keyed=True, note=f"modal refused the billing summary, {refused}")
        spent, cycle = float(summary.metered_cost), cycle_month(summary.start)
        if not declared:
            return Standing(
                keyed=True,
                note=f"${spent:.2f} metered in {cycle}, set {_CREDIT_VAR} to derive a balance",
            )
        return Standing(
            keyed=True,
            credit_usd=declared - spent,
            note=f"derived, ${declared:.2f} declared less ${spent:.2f} metered in {cycle}",
        )

    def cancel(self, handle: str) -> None:
        """Terminate the sandbox, tolerating one Modal has already forgotten."""
        modal = _modal()
        with suppress(modal.exception.NotFoundError):
            modal.Sandbox.from_id(handle).terminate()

    def logs(self, handle: str) -> str:
        return str(_modal().Sandbox.from_id(handle).stdout.read())

    def standing(self) -> Standing:
        """What the workspace can still spend: its cycle budget, else the derived balance.

        The token pair is what `modal.Client` checks before its first call (from
        `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` or the active `~/.modal.toml` profile), so an
        unauthenticated machine costs no round trip. The `.env` is merged before the SDK is
        imported, since Modal reads that pair from the environment as its module loads.
        """
        Credentials().load()
        try:
            modal = _modal()
        except MissionError as absent:
            return Standing(note=str(absent))
        config = modal.config.config
        if not (config["token_id"] and config["token_secret"]):
            return Standing(note="run `modal token new`, or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET")
        return self.budgeted(modal) or self.derived(modal)

    def state(self, handle: str) -> JobState:
        exit_code = _modal().Sandbox.from_id(handle).poll()
        verdict = "running" if exit_code is None else ("ok" if exit_code == 0 else "failed")
        return JobState(handle=handle, exit_code=exit_code, verdict=verdict)

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        self.admit(plan, resources)
        modal = _modal()
        container = plan.container
        image = (
            modal.Image.from_registry(container.image)
            if container is not None
            else modal.Image.debian_slim()
        )
        kwargs = {
            "app": modal.App.lookup(_APP_NAME, create_if_missing=True),
            "name": allocation.label,
            "image": image,
            "gpu": self._gpu_spec(resources),
        }
        if resources.walltime:
            kwargs["timeout"] = walltime_seconds(resources.walltime)
        # The command IS the sandbox entrypoint, so its lifetime, exit code and stdout are the
        # job's; a detached exec left the sandbox running forever with empty logs (the first live
        # submit). Its status is re-raised after the receipts are framed back.
        script = f"{staging()}\n{command}\nstatus=$?\n{framing()}\nexit $status"
        allocation.begin()
        sandbox = modal.Sandbox.create("bash", "-c", script, **kwargs)
        return allocation.created(str(sandbox.object_id))

    @staticmethod
    def _gpu_spec(resources: Resources) -> str | None:
        """`resources` as a Modal `gpu=` value, or None to request no GPU."""
        if not resources.gpus:
            return None
        if not resources.gpu_name:
            return str(resources.gpus)
        return (
            resources.gpu_name if resources.gpus == 1 else f"{resources.gpu_name}:{resources.gpus}"
        )
