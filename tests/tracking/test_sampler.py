import os
from typing import TYPE_CHECKING

import pytest

from mainboard.batch import Topic
from mainboard.tracking import Sampler, host_env, sampling_line

from ..batch.conftest import Recorder

if TYPE_CHECKING:
    from collections.abc import Sequence

_STREAM = "smoke-1"
_JOB = "trial-a"


class FakeMemory:
    """A memory reading, which is one number a sample asks for."""

    def __init__(self, used_gb: float) -> None:
        self.used_gb = used_gb


class FakeCap:
    """The enforced ceiling a host's jobs really run under."""

    def __init__(self, limit_gb: float, capped: bool = True) -> None:
        self.limit_gb = limit_gb
        self.capped = capped


class FakeBusyness:
    """One unit's utilization percentages."""

    def __init__(self, gpu_pct: int, memory_pct: int) -> None:
        self.gpu_pct = gpu_pct
        self.memory_pct = memory_pct


class FakeGPU:
    """One accelerator as a sample reads it."""

    def __init__(self, used_gb: float, gpu_pct: int, memory_pct: int) -> None:
        self.memory = FakeMemory(used_gb)
        self.utilization = FakeBusyness(gpu_pct, memory_pct)


class FakeHost:
    """The host as a sample reads it, with the cap that makes these readings worth shipping."""

    def __init__(self, used_gb: float, limit_gb: float, capped: bool = True) -> None:
        self.memory = FakeMemory(used_gb)
        self.cgroup_memory = FakeCap(limit_gb, capped)


class FakeMachine:
    """A stand-in machine, so a reading is asserted rather than whatever this laptop is doing."""

    def __init__(self, host: FakeHost, gpus: Sequence[FakeGPU] = ()) -> None:
        self.host = host
        self.gpus = list(gpus)


def sampler(bus: Recorder, machine: FakeMachine, **options: float | int) -> Sampler:
    """A sampler over `machine`, publishing into `bus`."""
    return Sampler(
        bus,
        stream=_STREAM,
        job=_JOB,
        interval=options.pop("interval", 0.01),
        machine=machine,
        **options,  # pyrefly: ignore  reason=the two bounds are the only remaining options since=2026-08-22
    )


def test_a_reading_carries_the_cap_a_hosted_dashboard_never_had() -> None:
    """Used memory against the enforced ceiling is the series that predicts an OOM kill."""
    machine = FakeMachine(
        FakeHost(used_gb=50.0, limit_gb=100.0),
        [FakeGPU(4.0, 90, 40), FakeGPU(2.0, 10, 70)],
    )
    assert sampler(Recorder(), machine).reading() == {
        "gpu_used_gb": 6.0,
        "gpu_pct": 90,
        "gpu_memory_pct": 70,
        "host_used_gb": 50.0,
        "host_cap_gb": 100.0,
        "host_capped": True,
        "host_frac": 0.5,
    }


def test_a_machine_with_no_accelerator_and_no_cap_still_reads_as_something() -> None:
    """Every probe behind this is best effort, so a bare host answers zeros rather than raising."""
    bare = FakeMachine(FakeHost(used_gb=3.0, limit_gb=0.0, capped=False))
    reading = sampler(Recorder(), bare).reading()
    assert (reading["gpu_used_gb"], reading["gpu_pct"], reading["host_frac"]) == (0, 0, 0.0)
    assert reading["host_capped"] is False


def test_entering_samples_at_once_so_a_job_that_dies_early_still_left_a_series() -> None:
    bus = Recorder()
    machine = FakeMachine(FakeHost(used_gb=1.0, limit_gb=8.0))
    with sampler(bus, machine, seconds=0.05):
        pass
    published = [line for line in bus.replay() if line.topic is Topic.SAMPLE]
    assert published and published[0].job == _JOB and published[0].batch == _STREAM


def test_an_interval_of_zero_starts_no_thread_at_all() -> None:
    """How a workspace turns the lane off without any caller branching on it."""
    bus = Recorder()
    with sampler(bus, FakeMachine(FakeHost(1.0, 8.0)), interval=0.0) as quiet:
        assert not quiet.thread.is_alive()
    assert bus.replay() == []


def test_a_sampler_ends_with_its_budget_or_with_the_process_it_was_told_to_follow() -> None:
    """A sampler beside a dispatched command must never outlive the command."""
    machine = FakeMachine(FakeHost(1.0, 8.0))
    bounded = sampler(Recorder(), machine, seconds=-1.0)
    assert bounded.expired is True
    orphaned = sampler(Recorder(), machine, parent=os.getpid())
    assert orphaned.expired is False
    assert sampler(Recorder(), machine, parent=2**31 - 1).expired is True


def test_the_loop_stops_the_moment_it_is_told_to() -> None:
    bus = Recorder()
    running = sampler(bus, FakeMachine(FakeHost(1.0, 8.0)), interval=30.0)
    running.stop()
    running.loop()
    assert len(bus.replay()) == 1


@pytest.mark.parametrize(
    ("interval", "carries"),
    [(0.0, False), (10.0, True)],
    ids=["a job that samples nothing", "a job that watches itself"],
)
def test_a_dispatched_job_starts_the_sampler_itself_or_is_left_alone(
    interval: float, carries: bool
) -> None:
    """The seam that carries the live lane onto a machine that is not this one."""
    line = sampling_line(root="/work/p", stream=_STREAM, job=_JOB, interval=interval, seconds=1800)
    assert bool(line) is carries
    if not carries:
        return
    assert host_env("/work/p") in line
    assert "mainboard sample smoke-1 --job trial-a --interval 10 --seconds 1800" in line
    assert line.endswith("&") and '--parent "$$"' in line


def test_a_job_with_no_wall_budget_still_ends_with_its_own_shell() -> None:
    line = sampling_line(root="/work/p", stream=_STREAM, job=_JOB, interval=5)
    assert "--seconds" not in line and '--parent "$$"' in line
    assert host_env("/work/p") == "/work/p/.mainboard/tracking.env"
