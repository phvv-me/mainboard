# `ModalBackend` runs a command inside a Modal Sandbox. `modal` is an optional extra, so every
# call goes through the lazy `_modal` accessor instead of a module-level import.
#
# Modal exposes no account balance, and that was established by reading its wire contract rather
# than its docs (surveyed 2026-08-19 against modal 1.5.4). The shipped `modal_proto` descriptor
# carries one service of 239 methods over 620 messages, and not one message anywhere in it has a
# field named for a credit, a balance, a remaining amount or a prepaid pot. The only account-side
# reads are `WorkspaceBillingSummary` (per-cycle `metered_cost`, `billed_cost` and an
# `adjustments` map whose `Credits` entry is credit *applied* in that cycle, never credit left)
# and `EnvironmentList`/`EnvironmentGetBudget`, which carry a spend *cap* rather than a balance.
# There is no REST surface to fall back on either, since every path under api.modal.com answers
# `application/grpc` whatever it is sent, and modal.com/api/* is a flat 404.
#
# So the row asks in order of how much Modal itself vouches for the figure. A workspace that set
# a cycle budget has the closest thing Modal keeps to a balance, and `modal.environments.
# list_environments` carries the same four budget fields `EnvironmentGetBudget` answers
# (`cycle_budget_dollars`, `effective_cycle_spend_limit`, `current_cycle_usage`,
# `spend_limit_reached`) for one cheap call that needs no environment id, so that is the read
# taken. A workspace without the team feature refuses it with `PermissionDeniedError`, and one
# that simply never set a budget answers zero (verified live 2026-08-19), so both fall through
# rather than costing a row. Failing that, the row derives a balance from a credit the workspace
# declares once in `MODAL_CREDIT_USD` minus what Modal says the current cycle has metered, and
# the note says out loud that the figure is derived rather than reported.

import os
from contextlib import suppress
from importlib import import_module
from typing import TYPE_CHECKING

from ...core.errors import MissionError
from ..jobs.spec import walltime_seconds
from ..schedulers.base import JobState
from .base import (
    Account,
    Credentials,
    Delivery,
    LogSource,
    ProviderBackend,
    Standing,
    require_budget,
)

if TYPE_CHECKING:
    from types import ModuleType

    from ...context.plan import ExecutionPlan
    from ..schedulers.base import Resources

# The Modal app every sandbox is created under; sandboxes are one-shot jobs, so a single shared
# app is enough (Modal itself scopes billing and the dashboard view by app, not by sandbox).
_APP_NAME = "mainboard"
# Where the workspace declares the prepaid credit Modal itself will not report.
_CREDIT_VAR = "MODAL_CREDIT_USD"


def _modal() -> ModuleType:
    """The imported `modal` module, raising a clear fix when the optional extra is missing."""
    try:
        return import_module("modal")
    except ModuleNotFoundError:
        raise MissionError(
            "the modal backend needs the `modal` package; run `uv add modal` then "
            "`modal token new` to authenticate"
        ) from None


def declared_credit() -> float:
    """The credit `MODAL_CREDIT_USD` declares for this workspace, 0.0 when it declares none.

    Modal reports what a cycle has cost and never what the account has left, so the starting
    figure has to come from the person who bought the credit. It is a plain number rather than a
    secret, and it lives beside the provider keys in the workspace `.env` because that is the one
    file every provider's account-side settings already share, which is also why the file is
    merged into the environment here before the lookup.
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
    """Run a command in a fresh Modal Sandbox and treat the sandbox's own lifetime as the job's.

    Stateless: every call reconnects to the sandbox by id (`modal.Sandbox.from_id`), so one
    instance serves every handle with no session to carry between calls.

    A sandbox keeps its stdout, so logs are real here, but nothing it writes to disk outlives it
    unless a Volume was mounted at create time, which is why `Delivery` is declared in `lacks`
    rather than implemented.
    """

    name = "modal"

    lacks = {
        Delivery: "modal backend cannot deliver {path!r} yet; mount a modal Volume at submit "
        "time and pull it by hand until that path lands",
    }

    def cancel(self, handle: str) -> None:
        """Terminate the sandbox, tolerating one Modal has already forgotten.

        Every run the durable sweep settles is cancelled, and the same run can be settled twice,
        so a sandbox that is no longer addressable is the state this asks for rather than a fault.
        """
        modal = _modal()
        with suppress(modal.exception.NotFoundError):
            modal.Sandbox.from_id(handle).terminate()

    def logs(self, handle: str) -> str:
        return str(_modal().Sandbox.from_id(handle).stdout.read())

    def state(self, handle: str) -> JobState:
        sandbox = _modal().Sandbox.from_id(handle)
        exit_code = sandbox.poll()
        verdict = "running" if exit_code is None else ("ok" if exit_code == 0 else "failed")
        return JobState(handle=handle, exit_code=exit_code, verdict=verdict)

    def standing(self) -> Standing:
        """What the workspace can still spend, by whichever of two routes can answer for it.

        The token pair is what `modal.Client` itself checks before its first call, resolved from
        `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` or the active profile in `~/.modal.toml`, so an
        unauthenticated machine costs no round trip. Past that the preference is for the figure
        Modal itself keeps, a cycle budget, falling back on the one this workspace declares and
        we do the arithmetic for.

        The workspace `.env` is merged in before the SDK is imported at all, since Modal reads
        that pair out of the environment as its own module loads and would never see a token
        this workspace declared but nothing had exported yet.
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

    @staticmethod
    def budgeted(modal: ModuleType) -> Standing | None:
        """The cycle budget less what this cycle has used, None when the workspace keeps none.

        The closest thing Modal holds to a balance, so it is asked first. The default environment
        is preferred since that is where a sandbox lands when nothing names another. A workspace
        that never set a budget answers zero and one without the team feature refuses the read
        outright, and neither is a fault worth printing, so both come back None for the caller to
        fall through on.

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

        The one account read Modal offers is `Workspace.billing.summary`, whose `metered_cost` is
        the cost this workspace has run up in the calendar-month cycle it names, before any credit
        or discount is applied against it. Subtracting that from the declared credit is arithmetic
        we do, not a balance Modal blessed, so the note carries both figures and the cycle they
        belong to, and a workspace that declares nothing still gets the spend rather than a blank.
        Every Modal fault (a rate-limited summary is the common one) stays a note on a keyed row,
        since a throttled read says nothing about whether the provider is usable.

        modal: the imported `modal` module, already known to carry credentials.
        """
        declared = declared_credit()
        try:
            summary = modal.Workspace.from_context().billing.summary()
        except modal.exception.Error as refused:
            return Standing(keyed=True, note=f"modal refused the billing summary, {refused}")
        spent, cycle = float(summary.metered_cost), summary.start.strftime("%Y-%m")
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

    def submit(self, plan: ExecutionPlan, command: str, resources: Resources) -> str:
        require_budget(resources)
        modal = _modal()
        image = (
            modal.Image.from_registry(plan.container.image)
            if plan.containerized
            else modal.Image.debian_slim()
        )
        kwargs = {
            "app": modal.App.lookup(_APP_NAME, create_if_missing=True),
            "image": image,
            "gpu": self._gpu_spec(resources),
        }
        if resources.walltime:
            kwargs["timeout"] = walltime_seconds(resources.walltime)
        # The command IS the sandbox entrypoint, so the sandbox lifetime, exit code,
        # and stdout are the job's own; a detached exec would leave the sandbox idling
        # as running forever with empty logs (found by the first live submit).
        sandbox = modal.Sandbox.create("bash", "-c", command, **kwargs)
        return str(sandbox.object_id)

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
