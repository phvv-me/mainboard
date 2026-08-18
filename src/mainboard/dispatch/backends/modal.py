# `ModalBackend` runs a command inside a Modal Sandbox. `modal` is an optional extra, so every
# call goes through the lazy `_modal` accessor instead of a module-level import.

from importlib import import_module
from typing import TYPE_CHECKING

from ...core.errors import MissionError
from ..jobs.spec import walltime_seconds
from ..schedulers.base import JobState
from .base import ProviderBackend, require_budget

if TYPE_CHECKING:
    from types import ModuleType

    from ...context.plan import ExecutionPlan
    from ..schedulers.base import Resources

# The Modal app every sandbox is created under; sandboxes are one-shot jobs, so a single shared
# app is enough (Modal itself scopes billing and the dashboard view by app, not by sandbox).
_APP_NAME = "mainboard"


def _modal() -> ModuleType:
    """The imported `modal` module, raising a clear fix when the optional extra is missing."""
    try:
        return import_module("modal")
    except ModuleNotFoundError:
        raise MissionError(
            "the modal backend needs the `modal` package; run `uv add modal` then "
            "`modal token new` to authenticate"
        ) from None


class ModalBackend(ProviderBackend):
    """Run a command in a fresh Modal Sandbox and treat the sandbox's own lifetime as the job's.

    Stateless: every call reconnects to the sandbox by id (`modal.Sandbox.from_id`), so one
    instance serves every handle with no session to carry between calls.
    """

    name = "modal"

    def cancel(self, handle: str) -> None:
        _modal().Sandbox.from_id(handle).terminate()

    def deliver(self, handle: str, *, path: str) -> None:
        raise MissionError(
            f"modal backend cannot deliver {path!r} yet; mount a modal Volume at submit time "
            "and pull it by hand until that path lands"
        )

    def logs(self, handle: str) -> str:
        return str(_modal().Sandbox.from_id(handle).stdout.read())

    def state(self, handle: str) -> JobState:
        sandbox = _modal().Sandbox.from_id(handle)
        exit_code = sandbox.poll()
        verdict = "running" if exit_code is None else ("ok" if exit_code == 0 else "failed")
        return JobState(handle=handle, exit_code=exit_code, verdict=verdict)

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
