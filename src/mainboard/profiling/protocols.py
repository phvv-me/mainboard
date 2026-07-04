"""Structural contracts for the untyped CUPTI activity records.

`cupti-python` ships no type stubs, so the records its buffer-completed callback yields
are statically opaque. These Protocols pin down exactly the snake_case fields mainboard
reads off each record kind (CUPTI rule: attribute names are snake_case), so the parsing
in `trace.py` stays fully typed without leaning on `Any`. The few fields CUPTI omits on
some record kinds are read defensively with `getattr`, so they stay off the Protocols.
"""

from typing import Protocol

# JSON values accepted by the Chrome/Perfetto trace-event writer.
type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
type TraceEvent = dict[str, Json]


class TimedActivity(Protocol):
    """Any timed CUPTI record: its kind and device-clock window.

    CUPTI buffers yield one opaque record family discriminated at runtime by `kind`; this
    is the field set every record carries. The kind-specific fields below extend it, and
    the collector dispatches on `kind` before reading them. `name`/`cbid`/`correlation_id`
    are absent on some kinds, so the collector still reads those defensively with `getattr`.
    """

    kind: int
    start: int
    end: int


class KernelActivity(TimedActivity, Protocol):
    """A CUPTI CONCURRENT_KERNEL record: launch shape plus the device-clock window."""

    name: str
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    block_y: int
    block_z: int
    static_shared_memory: int
    dynamic_shared_memory: int
    registers_per_thread: int


class MemcpyActivity(TimedActivity, Protocol):
    """A CUPTI MEMCPY record: direction code and device-clock window (`bytes` via getattr)."""

    copy_kind: int


class RawActivity(KernelActivity, MemcpyActivity, Protocol):
    """The opaque CUPTI record as the buffer hands it over, before kind dispatch.

    CUPTI yields one C struct family, so a single record statically exposes every field;
    only the subset valid for its runtime `kind` is meaningful. Typing the buffer as this
    superset lets the collector pass a record to the kind-specific reader without a cast,
    and the reader takes only the fields its kind defines.
    """
