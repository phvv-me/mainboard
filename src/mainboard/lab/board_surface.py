import inspect
import json
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast

from pydantic import JsonValue

from ..core.project import Project
from .experiment import Experiment
from .gates import GateStatus
from .run import Run

if TYPE_CHECKING:
    from collections.abc import Callable

    from .gates import Gate, GateVerdict
    from .lane import Lane

# The whole published contract between a trial and anything reading its output: one JSON line
# under this key, printed by whatever drives the trial (nothing here prints it). The key names
# the shape, not the harness, so any framework printing the same line reads the same, and a
# reader (a proof-bookkeeping tool turning a claim's run into evidence naming its trial and
# gates) parses it instead of importing this package.
RECEIPT = "trial_receipt"


class Declarations(TypedDict, total=False):
    """The `Experiment` class attributes `experiment()` sets, as a hand-written subclass would."""

    lanes: tuple[Lane, ...]
    gates: tuple[Gate, ...]
    models: tuple[str, ...]
    trials: int
    seed: int


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Shared identity every trial result carries, whatever its outcome.

    gate_evidence: evaluated gate verdicts, ending at the first unmet precondition.
    node: the ledger slug this trial serves, absent from the printed line when empty.
    """

    verdict: ClassVar[GateStatus]

    run_id: str
    gate_evidence: tuple[GateVerdict, ...]
    node: str = field(default="", kw_only=True)

    def receipt(self) -> str:
        """This trial as its one `RECEIPT` JSON line.

        A new outcome kind declares a `verdict` and its fields, and its receipt follows without
        editing a renderer. `producer` is provenance, never something a reader branches on.
        """
        shared = {"run_id", "gate_evidence", "node"}
        payload: dict[str, JsonValue] = {
            "run_id": self.run_id,
            "outcome": str(self.verdict),
            "producer": Project().name,
            **({"node": self.node} if self.node else {}),
            "gates": [
                {"status": str(verdict.status), "reason": verdict.reason}
                for verdict in self.gate_evidence
            ],
            **{name: value for name, value in asdict(self).items() if name not in shared},
        }
        return json.dumps({RECEIPT: payload})


@dataclass(frozen=True, slots=True)
class TrialResult(TrialOutcome):
    """A trial that cleared every gate and ran to completion, with what `measure` returned."""

    verdict: ClassVar[GateStatus] = GateStatus.PASSED

    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class BlockedTrial(TrialOutcome):
    """A trial withheld by a gate that legitimately isn't ready yet, never a failure."""

    verdict: ClassVar[GateStatus] = GateStatus.BLOCKED

    reason: str


@dataclass(frozen=True, slots=True)
class FailedTrial(TrialOutcome):
    """A trial whose gate check itself broke."""

    verdict: ClassVar[GateStatus] = GateStatus.FAILED

    reason: str


_STOPPED: dict[GateStatus, type[BlockedTrial | FailedTrial]] = {
    GateStatus.BLOCKED: BlockedTrial,
    GateStatus.FAILED: FailedTrial,
}


def experiment(
    **declarations: typing.Unpack[Declarations],
) -> Callable[[Callable[..., dict[str, float]]], type[Experiment]]:
    """Turn a plain measuring function into a registered `Experiment` subclass.

    The function's first parameter is `run`; every other keyword-only parameter becomes a pydantic
    config field, its `Annotated` domain kept for `space_of`. The function becomes `measure`,
    called with `run` and every config field as keywords, plus `lane` when it declares one.
    """

    def decorate(fn: Callable[..., dict[str, float]]) -> type[Experiment]:
        hints = typing.get_type_hints(fn, include_extras=True)
        parameters = inspect.signature(fn).parameters
        accepts_lane = "lane" in parameters
        config_names = [name for name in parameters if name not in {"run", "lane"}]

        def measure(self: Experiment, run: Run, lane: Lane | None = None) -> dict[str, float]:
            config = self.model_dump()
            if accepts_lane:
                config["lane"] = lane
            return fn(run, **config)

        def body(namespace: dict[str, object]) -> None:
            namespace.update(declarations)
            namespace["measure"] = measure
            namespace["__module__"] = fn.__module__
            namespace["__qualname__"] = fn.__qualname__
            # A function's snake_case would leak into the registry key, where every
            # other implementation is kebab, so the generated class declares it.
            namespace["name"] = fn.__name__.replace("_", "-")
            namespace["__annotations__"] = {name: hints[name] for name in config_names}
            for name in config_names:
                default = parameters[name].default
                if default is not inspect.Parameter.empty:
                    namespace[name] = default

        return cast(
            "type[Experiment]", types.new_class(fn.__name__, (Experiment,), exec_body=body)
        )

    return decorate


def runnable(
    experiment_cls: type[Experiment],
    model: str,
    config: Experiment,
    *,
    lane: Lane | None = None,
) -> TrialResult | BlockedTrial | FailedTrial:
    """Check `experiment_cls.gates` in order, then set up and measure only when all pass.

    Stop at the first blocked or failed gate, since later checks can need resources earlier gates
    admit, and keep only the evaluated verdicts. A blocked precondition stays distinct from a
    failed check.
    """
    trial_id = config.run_id(model=model, lane=lane)
    run = Run(
        model_id=model, config=config, artifact_dir=Path(Project().out_dir) / "runs" / trial_id
    )
    evidence: list[GateVerdict] = []
    for gate in experiment_cls.gates:
        verdict = gate.check(run)
        evidence.append(verdict)
        if stopped := _STOPPED.get(verdict.status):
            return stopped(run_id=trial_id, gate_evidence=tuple(evidence), reason=verdict.reason)
    config.setup(run)
    metrics = config.measure(run, lane)
    return TrialResult(run_id=trial_id, metrics=metrics, gate_evidence=tuple(evidence))
