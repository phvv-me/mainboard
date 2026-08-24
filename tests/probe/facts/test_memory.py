import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.probe import Memory

_GIB = 1024**3
# Byte counts a real reading holds, bounded so the gibibyte views stay exact in float rather
# than testing the arithmetic of an integer no machine reports.
_BYTES = st.integers(min_value=0, max_value=2**53)


@pytest.mark.usefixtures("fake_psutil_memory")
def test_a_system_reading_records_the_pool_it_sampled() -> None:
    """A system reading names psutil as its source.

    `Memory.system` takes total, used and free straight off psutil, so a caller can tell a
    host RAM reading from a device one it was handed alongside.
    """
    memory = Memory.system(scope="unified", unified=True)
    assert (memory.total_bytes, memory.used_bytes, memory.free_bytes) == (
        48 * _GIB,
        16 * _GIB,
        32 * _GIB,
    )
    assert memory.scope == "unified"
    assert memory.unified is True
    assert memory.source == "psutil"
    assert Memory.system().scope == "system"


# The arithmetic is a handful of divisions and the round trip either holds or does not, so a
# small budget covers it, with the empty reading pinned below rather than searched for.
@settings(max_examples=15)
@given(
    memory=st.builds(Memory, total_bytes=_BYTES, used_bytes=_BYTES, free_bytes=_BYTES),
    wire=st.from_type(Memory),
)
@example(memory=Memory(), wire=Memory())
def test_a_reading_converts_to_gibibytes_survives_json_and_never_divides_by_zero(
    memory: Memory, wire: Memory
) -> None:
    """The derived views hold for any reading.

    Every byte field has a gibibyte view scaled by the same 1024**3, an empty reading
    reports zero percent used rather than raising, and any reading round trips the wire.
    """
    assert memory.total_gb == memory.total_bytes / _GIB
    assert memory.used_gb == memory.used_bytes / _GIB
    assert memory.free_gb == memory.free_bytes / _GIB
    used_share = memory.used_bytes / memory.total_bytes * 100 if memory.total_bytes else 0.0
    assert memory.percent_used == used_share
    assert Memory.model_validate_json(wire.model_dump_json()) == wire
