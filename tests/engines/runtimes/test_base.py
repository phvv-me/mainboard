from collections.abc import Callable

import pytest

from mainboard import MissionError
from mainboard.engines import ContainerRuntime, Docker, Podman
from mainboard.engines.runtimes import resolve
from mainboard.manifest import Container, Guardrail


def test_resolve_auto_picks_the_first_available_runtime(which: Callable[..., None]) -> None:
    which("podman")
    assert resolve("auto") is Podman


def test_resolve_explicit_name_ignores_availability(which: Callable[..., None]) -> None:
    which()
    assert resolve("docker") is Docker


def test_resolve_unknown_name_lists_known_runtimes(which: Callable[..., None]) -> None:
    which()
    with pytest.raises(
        MissionError, match=r"known runtimes are \['apptainer', 'docker', 'podman'\]"
    ):
        resolve("bogus")


def test_resolve_auto_with_nothing_installed_lists_known_runtimes(
    which: Callable[..., None],
) -> None:
    which()
    with pytest.raises(MissionError, match="no container runtime available for 'auto'"):
        resolve("auto")


def test_env_flags_pairs_each_passthrough_variable() -> None:
    assert ContainerRuntime.env_flags(["A", "B"]) == ["--env", "A", "--env", "B"]
    assert ContainerRuntime.env_flags([]) == []


def test_guarded_argv_wraps_only_when_the_guardrail_is_declared() -> None:
    guarded = Container(image="x", guardrails=[Guardrail.UNSET_PIP_CONSTRAINT])
    unguarded = Container(image="x", guardrails=[Guardrail.PIN_SYSTEM_PACKAGES])
    assert ContainerRuntime.guarded_argv(guarded, ["run"]) == [
        "env",
        "-u",
        "PIP_CONSTRAINT",
        "run",
    ]
    assert ContainerRuntime.guarded_argv(unguarded, ["run"]) == ["run"]
