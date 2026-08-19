from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError, Resolver, load
from mainboard.context import admit, evaluate
from mainboard.manifest import HostProfile, QueuePolicy

from ..conftest import MANIFEST


@pytest.fixture
def resolver(workspace: Path) -> Resolver:
    return Resolver(load(workspace / MANIFEST))


def test_plan_resolves_profile_env_container_and_vars(resolver: Resolver) -> None:
    plan = resolver.plan("miyabi-g")
    assert plan.containerized
    assert plan.container is not None and plan.container.image.startswith("nvcr.io")
    assert plan.env == "default"
    assert plan.vars["cuda"] == "13.0"
    assert plan.prefix("/work/x/projects") == "/work/x/projects/.mainboard/.pixi/envs/default"


def test_plan_overrides_win_and_none_forces_bare(resolver: Resolver) -> None:
    assert resolver.plan("gold").env == "serving"
    assert resolver.plan("gold", env="default").env == "default"
    assert not resolver.plan("miyabi-g", container="none").containerized


def test_plan_refuses_unknown_references(resolver: Resolver) -> None:
    with pytest.raises(MissionError, match="declared environments"):
        resolver.plan("gold", env="ghost")
    with pytest.raises(MissionError, match="declared containers"):
        resolver.plan("gold", container="ghost")


def test_local_host_plans_bare_with_defaults(resolver: Resolver) -> None:
    plan = resolver.plan()
    assert plan.host == "local" and not plan.containerized


@given(attempt=st.integers(min_value=1, max_value=6))
def test_expressions_escalate_and_saturate(attempt: int) -> None:
    assert evaluate("min(100, attempt * 50)", attempt=attempt) == min(100, attempt * 50)


def test_expressions_accept_bare_integers_and_arithmetic() -> None:
    assert evaluate("64", attempt=1) == 64
    assert evaluate("max(8, 100 // 3 - 1)", attempt=1) == 32
    assert evaluate("attempt + 1", attempt=2) == 3


def test_expressions_refuse_everything_else() -> None:
    with pytest.raises(MissionError, match="does not parse"):
        evaluate("50 +", attempt=1)
    for hostile in ("__import__('os')", "attempt ** 9", "walltime", "min(1)[0]"):
        with pytest.raises(MissionError, match="more than integers"):
            evaluate(hostile, attempt=1)


def test_admission_enforces_the_declared_envelope() -> None:
    profile = HostProfile(
        queues={
            "short-g": QueuePolicy(max_walltime="07:59:59", mem_ceiling_gb=100, notes="memsw cap"),
            "router": QueuePolicy(submittable=False, notes="leaf queues only"),
        }
    )
    admit(profile, queue="short-g", walltime="06:00:00", mem_gb=100)
    with pytest.raises(MissionError, match="exceeds the 'short-g' ceiling 07:59:59"):
        admit(profile, queue="short-g", walltime="08:00:00", mem_gb=50)
    with pytest.raises(MissionError, match="exceeds the 'short-g' ceiling 100GB"):
        admit(profile, queue="short-g", walltime="01:00:00", mem_gb=110)
    with pytest.raises(MissionError, match="not submittable"):
        admit(profile, queue="router", walltime="00:10:00", mem_gb=1)
    admit(profile, queue="undeclared", walltime="99:00:00", mem_gb=999)
