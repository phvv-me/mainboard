import inspect
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from ..core.project import Project
from .experiment import Experiment
from .gates import GateStatus
from .run import Run

if TYPE_CHECKING:
    from collections.abc import Callable

    from .gates import Gate, GateVerdict
    from .lane import Lane


class Declarations(TypedDict, total=False):
    """The `Experiment` class attributes `experiment()` may set from its keyword declarations.

    Exactly the class attributes a hand-written `Experiment` subclass would set: `lanes`,
    `gates`, `models`, `trials`, `seed`.
    """

    lanes: tuple[Lane, ...]
    gates: tuple[Gate, ...]
    models: tuple[str, ...]
    trials: int
    seed: int


@dataclass(frozen=True, slots=True)
class TrialOutcome:
    """Shared identity every trial result carries, whatever its outcome.

    run_id: this trial's dedup identity.
    gate_evidence: every declared gate's verdict, in declaration order.
    """

    run_id: str
    gate_evidence: tuple[GateVerdict, ...]


@dataclass(frozen=True, slots=True)
class TrialResult(TrialOutcome):
    """A trial that cleared every gate and ran to completion.

    metrics: the named metrics `measure` returned.
    """

    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class BlockedTrial(TrialOutcome):
    """A trial withheld by a gate that legitimately isn't ready yet, never a failure.

    reason: the blocking gate's own reason.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class FailedTrial(TrialOutcome):
    """A trial whose gate check itself broke.

    reason: the failing gate's own reason.
    """

    reason: str


def experiment(
    **declarations: typing.Unpack[Declarations],
) -> Callable[[Callable[..., dict[str, float]]], type[Experiment]]:
    """Turn a plain measuring function into a registered `Experiment` subclass.

    The decorated function's first parameter is `run`; every other, keyword-only parameter
    becomes a pydantic config field on the generated class, its `Annotated` domain carried
    through unchanged for `space_of` to read later. The function itself becomes the generated
    class's `measure`, called with `run` and every config field as keywords, plus `lane` when
    the function declares that parameter itself.

    declarations: `lanes`, `gates`, `models`, `trials`, `seed`, exactly the class attributes a
        hand-written `Experiment` subclass would set.
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
    """Execute one trial: every declared gate, then `setup` and `measure` once they all clear.

    Every gate is checked, not just the first blocker, so a trial's evidence always reflects
    the full precondition sweep. A blocked gate outranks a failed one in the returned outcome,
    since BLOCKED is an expected, recoverable wait and FAILED is a real break; `runnable` never
    conflates the two.

    experiment_cls: the registered `Experiment` subclass whose declared `gates` this trial
        must clear.
    model: the model id this trial runs against.
    config: the validated experiment instance `setup`/`measure` run with.
    lane: the counterbalanced condition this trial measures, when the experiment declares any.
    """
    trial_id = config.run_id(model=model, lane=lane)
    run = Run(
        model_id=model, config=config, artifact_dir=Path(Project().out_dir) / "runs" / trial_id
    )
    evidence = tuple(gate.check(run) for gate in experiment_cls.gates)

    blocked = next((verdict for verdict in evidence if verdict.status == GateStatus.BLOCKED), None)
    if blocked is not None:
        return BlockedTrial(run_id=trial_id, gate_evidence=evidence, reason=blocked.reason)

    failed = next((verdict for verdict in evidence if verdict.status == GateStatus.FAILED), None)
    if failed is not None:
        return FailedTrial(run_id=trial_id, gate_evidence=evidence, reason=failed.reason)

    config.setup(run)
    metrics = config.measure(run, lane)
    return TrialResult(run_id=trial_id, metrics=metrics, gate_evidence=evidence)
