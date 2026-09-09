from patos import FrozenModel

from ...models import ProcessReading


class SpanMeasurement(FrozenModel):
    """Completed span data summarized when a result is requested."""

    name: str
    wall_ms: float
    samples: tuple[ProcessReading, ...]
