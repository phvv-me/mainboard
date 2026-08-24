from contextlib import nullcontext
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError, Project, Resolver, load
from mainboard.context import admit, evaluate
from mainboard.manifest import HostProfile, QueuePolicy

if TYPE_CHECKING:
    from pathlib import Path

# The envelope the miyabi queues really declare, a walltime ceiling one second under the value
# the scheduler rejects, the memsw cap, and a router queue nothing may target.
_PROFILE = HostProfile(
    queues={
        "short-g": QueuePolicy(max_walltime="07:59:59", mem_ceiling_gb=100, notes="memsw cap"),
        "router": QueuePolicy(submittable=False, notes="leaf queues only"),
    }
)


@pytest.fixture
def resolver(workspace: Path) -> Resolver:
    """A resolver over the full-featured fixture workspace."""
    return Resolver(load(workspace / Project().manifest))


def test_a_plan_resolves_where_and_how_one_command_runs(resolver: Resolver) -> None:
    """Host, profile, environment, container and vars, with an override winning over each."""
    miyabi = resolver.plan("miyabi-g")
    assert miyabi.containerized
    assert miyabi.container is not None and miyabi.container.image.startswith("nvcr.io")
    assert miyabi.env == "default"
    assert miyabi.vars["cuda"] == "13.0"
    assert miyabi.prefix("/work/x/projects") == "/work/x/projects/.mainboard/.pixi/envs/default"
    assert not resolver.plan("miyabi-g", container="none").containerized
    assert resolver.plan("gold").env == "serving"
    assert resolver.plan("gold", env="default").env == "default"
    local = resolver.plan()
    assert local.host == "local"
    assert not local.containerized


@pytest.mark.parametrize(
    ("env", "container", "match"),
    [("ghost", "", "declared environments"), ("", "ghost", "declared containers")],
)
def test_a_plan_refuses_a_reference_the_manifest_never_declared(
    resolver: Resolver, env: str, container: str, match: str
) -> None:
    """The roster comes back with the refusal, so the typo is fixable from the message."""
    with pytest.raises(MissionError, match=match):
        resolver.plan("gold", env=env, container=container)


@given(attempt=st.integers(min_value=1, max_value=6))
def test_a_resource_expression_escalates_with_the_attempt_and_saturates(attempt: int) -> None:
    """The whole tiny language, over every retry number a resubmission can carry."""
    assert evaluate("min(100, attempt * 50)", attempt=attempt) == min(100, attempt * 50)
    assert evaluate("max(8, 100 // 3 - 1)", attempt=attempt) == 32
    assert evaluate("attempt + 1", attempt=attempt) == attempt + 1
    assert evaluate("100 / 4 - attempt", attempt=attempt) == 25 - attempt
    assert evaluate("64", attempt=attempt) == 64


@pytest.mark.parametrize(
    ("hostile", "match"),
    [
        ("50 +", "does not parse"),
        ("__import__('os')", "more than integers"),
        ("attempt ** 9", "more than integers"),
        ("walltime", "more than integers"),
        ("min(1)[0]", "more than integers"),
    ],
)
def test_a_resource_expression_outside_the_language_refuses(hostile: str, match: str) -> None:
    """Nothing but integers, `attempt`, arithmetic and the two clamps can execute here."""
    with pytest.raises(MissionError, match=match):
        evaluate(hostile, attempt=1)


@pytest.mark.parametrize(
    ("queue", "walltime", "mem_gb", "match"),
    [
        ("short-g", "06:00:00", 100, ""),
        ("undeclared", "99:00:00", 999, ""),
        ("short-g", "08:00:00", 50, "exceeds the 'short-g' ceiling 07:59:59"),
        ("short-g", "01:00:00", 110, "exceeds the 'short-g' ceiling 100GB"),
        ("router", "00:10:00", 1, "not submittable"),
    ],
)
def test_admission_enforces_the_declared_envelope_before_the_round_trip(
    queue: str, walltime: str, mem_gb: int, match: str
) -> None:
    """The scheduler's own rejection arrives minutes later, this one names the ceiling now."""
    expectation = pytest.raises(MissionError, match=match) if match else nullcontext()
    with expectation:
        admit(_PROFILE, queue=queue, walltime=walltime, mem_gb=mem_gb)
