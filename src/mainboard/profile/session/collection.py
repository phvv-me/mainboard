from patos import FrozenModel
from pydantic import Field

from ..trace import Activity
from .feature import Feature


class Collection(FrozenModel):
    """Collection policy retained beside profiling evidence.

    features: independently enabled collectors.
    activities: CUPTI kinds retained when ACTIVITY is enabled.
    device_index: selected entry in the profiler's GPU sequence.
    sample_interval_ms: device telemetry polling interval.
    max_spans: retained span and window limit; excess records are counted.
    auto: modules whose functions receive automatic annotations.
    """

    features: Feature = Feature.DEFAULT
    activities: Activity = Activity.DEFAULT
    device_index: int = Field(default=0, ge=0)
    sample_interval_ms: int = Field(default=50, gt=0)
    max_spans: int = Field(default=100_000, gt=0)
    auto: tuple[str, ...] = ()
