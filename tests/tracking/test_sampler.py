import os
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from mainboard.batch import Topic
from mainboard.runtime.job import ToolCall
from mainboard.tracking import Sampler, attesting, host_env, sampling

from ..batch.support import Recorder

_STREAM = "smoke-1"
_JOB = "trial-a"


@dataclass
class FakeMemory:
    used_gb: float


@dataclass
class FakeCap:
    limit_gb: float
    capped: bool = True


@dataclass
class FakeBusyness:
    gpu_pct: int
    memory_pct: int


class FakeGPU:
    def __init__(self, used_gb: float, gpu_pct: int, memory_pct: int) -> None:
        self.memory = FakeMemory(used_gb)
        self.utilization = FakeBusyness(gpu_pct, memory_pct)


class FakeHost:
    def __init__(self, used_gb: float, limit_gb: float, capped: bool = True) -> None:
        self.memory = FakeMemory(used_gb)
        self.cgroup_memory = FakeCap(limit_gb, capped)


@dataclass
class FakeMachine:
    """A stand-in machine, so a reading is asserted rather than whatever this laptop is doing."""

    host: FakeHost
    gpus: Sequence[FakeGPU] = ()


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


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        pytest.param(
            FakeMachine(FakeHost(50.0, 100.0), [FakeGPU(4.0, 90, 40), FakeGPU(2.0, 10, 70)]),
            (6.0, 90, 70, 50.0, 100.0, True, 0.5),
            id="the-cap-a-hosted-dashboard-never-had",
        ),
        pytest.param(
            FakeMachine(FakeHost(3.0, 0.0, capped=False)),
            (0, 0, 0, 3.0, 0.0, False, 0.0),
            id="a-bare-host-answers-zeros-rather-than-raising",
        ),
    ],
)
def test_a_reading_carries_used_memory_against_the_enforced_cap(
    machine: FakeMachine, expected: tuple[float | bool, ...]
) -> None:
    """Used memory against the enforced ceiling is the series that predicts an OOM kill."""
    keys = ("gpu_used_gb", "gpu_pct", "gpu_memory_pct", "host_used_gb", "host_cap_gb")
    keys += ("host_capped", "host_frac")
    assert sampler(Recorder(), machine).reading() == dict(zip(keys, expected, strict=True))


@pytest.mark.parametrize(
    ("gpus", "idle"),
    [
        pytest.param([FakeGPU(4.0, 0, 0)], True, id="a-node-doing-nothing"),
        pytest.param([FakeGPU(4.0, 10, 0)], True, id="a-node-at-exactly-the-threshold"),
        pytest.param([FakeGPU(4.0, 47, 0)], False, id="a-node-another-job-is-already-holding"),
        pytest.param([], True, id="a-machine-with-no-accelerator-at-all"),
    ],
)
def test_an_attestation_says_what_the_machine_was_doing_before_the_work_started(
    gpus: list[FakeGPU], idle: bool
) -> None:
    """The honest half of contention: nothing is forbidden, the conditions are simply recorded."""
    bus = Recorder()
    machine = FakeMachine(FakeHost(used_gb=1.0, limit_gb=8.0), gpus)
    published = sampler(bus, machine).attest()
    assert published.topic is Topic.ATTESTED
    assert published.job == _JOB and published.batch == _STREAM
    assert published.data["idle"] is idle
    # The whole reading rides along, so a reader weighs the conditions rather than taking one
    # word for them.
    assert published.data["host_cap_gb"] == 8.0
    assert [line.topic for line in bus.replay()] == [Topic.ATTESTED]


def test_the_attestation_is_this_tools_own_verb_carrying_the_staged_credential() -> None:
    """A reading taken beside the command describes the command, not the conditions it got."""
    assert attesting(root="/repo", stream=_STREAM, job=_JOB) == ToolCall(
        args=("attest", _STREAM, "--job", _JOB), credentials=host_env("/repo")
    )


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
    ("interval", "seconds", "sampled"),
    [
        (0.0, 1800.0, None),
        (10.0, 1800.0, ("--interval", "10", "--seconds", "1800")),
        (5.0, 0.0, ("--interval", "5")),
    ],
    ids=["a job that samples nothing", "a job that watches itself", "no wall budget"],
)
def test_a_dispatched_job_starts_the_sampler_itself_or_is_left_alone(
    interval: float, seconds: float, sampled: tuple[str, ...] | None
) -> None:
    """The seam that carries the live lane onto a machine that is not this one."""
    call = sampling(root="/work/p", stream=_STREAM, job=_JOB, interval=interval, seconds=seconds)
    expected = (
        None
        if sampled is None
        else ToolCall(
            args=("sample", _STREAM, "--job", _JOB, *sampled), credentials=host_env("/work/p")
        )
    )
    assert call == expected
    assert host_env("/work/p") == "/work/p/.mainboard/tracking.json"
