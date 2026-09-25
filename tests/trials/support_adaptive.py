# Stand-ins for hypothesis and optuna: what is under test is the seam (`Hunt` spends its budget
# and keeps the last shrunk witness; `Study` asks, evaluates, tells and writes one row per
# iteration), not the libraries. They answer exactly the calls `adversarial` and `search` make and
# are installed by name into `sys.modules`, the door `adaptive.driver` opens, so the import path
# under test is the real one.

from types import ModuleType, SimpleNamespace
from typing import Any


class Draws:
    """A fake `@given` drawing from plain sequences and shrinking toward each one's first value.

    Budget and seed arrive from the fake `settings` and `seed`, as the real decorators supply them.
    """

    def __init__(self, function: Any, strategies: dict[str, list[Any]]) -> None:
        self.function = function
        self.strategies = strategies
        self.budget = 0
        self.seed = 0
        self.drawn: list[dict[str, Any]] = []

    def draw(self, index: int) -> dict[str, Any]:
        return {
            name: values[(self.seed + index) % len(values)]
            for name, values in self.strategies.items()
        }

    def __call__(self) -> None:
        for index in range(self.budget):
            example = self.draw(index)
            self.drawn.append(example)
            try:
                self.function(**example)
            except Exception:
                raise self.shrink() from None

    def shrink(self) -> Exception:
        smallest: Exception | None = None
        for index in range(self.budget):
            try:
                self.function(**self.draw(index))
            except Exception as raised:  # noqa: PERF203  the raise IS the signal being minimised
                smallest = raised
                break
        assert smallest is not None
        return smallest


def hypothesis(*, health: tuple[str, ...] = ("too_slow",)) -> ModuleType:
    """A module answering the four names `adversarial.Hunt.against` reaches for."""

    def given(**strategies: list[Any]) -> Any:
        return lambda function: Draws(function, strategies)

    def settings(*, max_examples: int, **rest: Any) -> Any:
        def applied(draws: Draws) -> Draws:
            draws.budget = max_examples
            draws.settings = rest
            return draws

        return applied

    def seed(value: int) -> Any:
        def applied(draws: Draws) -> Draws:
            draws.seed = value
            return draws

        return applied

    module = ModuleType("hypothesis")
    module.given, module.settings, module.seed, module.HealthCheck = (
        given,
        settings,
        seed,
        health,
    )
    return module


class Cycle:
    """A fake optuna study, suggesting each axis in order and recording what it was told."""

    def __init__(self, sampler: Any, direction: str) -> None:
        self.sampler = sampler
        self.direction = direction
        self.asked = 0
        self.told: list[tuple[Any, float]] = []

    def ask(self) -> Any:
        index = self.asked
        self.asked += 1
        return SimpleNamespace(
            number=index,
            suggest_categorical=lambda name, values: values[index % len(values)],
        )

    def tell(self, trial: Any, loss: float) -> None:
        self.told.append((trial.number, loss))


def optuna() -> ModuleType:
    """A module answering the names `search.Optuna` reaches for, and no others."""
    module = ModuleType("optuna")
    module.logging = SimpleNamespace(set_verbosity=lambda level: None, WARNING=30)
    module.samplers = SimpleNamespace(TPESampler=lambda seed: SimpleNamespace(seed=seed))
    module.create_study = lambda direction, sampler: Cycle(sampler, direction)
    return module
