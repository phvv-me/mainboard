"""A queue admits a walltime up to its declared ceiling and refuses any longer one."""

from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.manifest import QueuePolicy
from mainboard.runtime.job import walltime

_SHORT_G = 8 * 3600 - 1


@given(asked=st.integers(0, 100 * 3600), ceiling=st.integers(1, 100 * 3600))
@example(asked=_SHORT_G + 1, ceiling=_SHORT_G)
@example(asked=_SHORT_G, ceiling=_SHORT_G)
def test_a_queue_admits_exactly_the_walltimes_up_to_its_ceiling(asked: int, ceiling: int) -> None:
    policy = QueuePolicy(max_walltime=walltime(ceiling))
    assert policy.admits_walltime(walltime(asked)) == (asked <= ceiling)


def test_a_queue_declaring_no_ceiling_admits_any_walltime() -> None:
    assert QueuePolicy().admits_walltime("999:59:59")
