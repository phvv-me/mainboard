"""What dispatch admits by the free disk a job host keeps at its workspace."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.context.admission import admit_disk
from mainboard.core.constants.units import JOB_HOST_DISK


@given(free=st.integers(min_value=0, max_value=JOB_HOST_DISK - 1))
def test_a_job_is_refused_on_a_host_below_the_free_disk_a_job_host_keeps(free: int) -> None:
    with pytest.raises(MissionError, match=r"sweep-7 needs 20 GB free on gold at /w, which has"):
        admit_disk("sweep-7", host="gold", root="/w", free=free)


@given(free=st.integers(min_value=JOB_HOST_DISK, max_value=2**50))
def test_a_job_is_admitted_on_a_host_keeping_the_free_disk_a_job_host_needs(free: int) -> None:
    admit_disk("sweep-7", host="gold", root="/w", free=free)
