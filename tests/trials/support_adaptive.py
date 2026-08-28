# THE DRIVER, STOOD IN FOR, because what is under test here is the lane and not the library.
#
# hypothesis's shrinker is somebody else's tested code, and a suite that drove the real one would
# be asserting its behaviour while paying for its install. What this package owes a reader is that
# `Hunt` spends the budget it was given, counts what it drew, keeps the LAST witness a shrink
# produced and settles the consumer's own words with it. Every one of those is a statement about
# the SEAM, so the seam is what is doubled: a module answering the exact calls `adversarial` makes,
# and nothing else.

from types import ModuleType
from typing import Any


class Draws:
    """A fake `@given`, drawing from plain sequences and shrinking toward each one's first value.

    The budget arrives from the fake `settings` and the seed from the fake `seed`, exactly as the
    real decorators supply them, so the wrapper reads what the code under test actually set.
    """

    def __init__(self, function: Any, strategies: dict[str, list[Any]]) -> None:
        self.function = function
        self.strategies = strategies
        self.budget = 0
        self.seed = 0
        self.drawn: list[dict[str, Any]] = []

    def draw(self, index: int) -> dict[str, Any]:
        """One example, cycling each strategy's own values so a run is a function of the seed."""
        return {
            name: values[(self.seed + index) % len(values)]
            for name, values in self.strategies.items()
        }

    def __call__(self) -> None:
        """Spend the budget, then shrink a failure toward the front of every strategy."""
        for index in range(self.budget):
            example = self.draw(index)
            self.drawn.append(example)
            try:
                self.function(**example)
            except Exception:
                raise self.shrink() from None

    def shrink(self) -> Exception:
        """Re-run from the front and hand back the smallest example that still raised."""
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
