from collections.abc import Callable

from mainboard.engines import Docker
from mainboard.manifest import Container, Guardrail


def test_is_available_uses_plain_which(which: Callable[..., None]) -> None:
    which()
    assert not Docker.is_available()
    which("docker")
    assert Docker.is_available()


def test_command_with_gpu_binds_workdir_passthrough_and_guardrail() -> None:
    container = Container(
        image="img",
        binds=["/data"],
        workdir="/app",
        passthrough=["A", "B"],
        guardrails=[Guardrail.UNSET_PIP_CONSTRAINT],
    )
    argv = Docker.command(container, prefix_bind="/h:/p", argv=["run"])
    assert argv == [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        "/data",
        "-v",
        "/h:/p",
        "-w",
        "/app",
        "--env",
        "A",
        "--env",
        "B",
        "img",
        "env",
        "-u",
        "PIP_CONSTRAINT",
        "run",
    ]


def test_command_without_gpu_workdir_passthrough_or_guardrails() -> None:
    container = Container(image="img", gpus=False, guardrails=[])
    argv = Docker.command(container, prefix_bind="p:p", argv=["run"])
    assert argv == ["docker", "run", "--rm", "-v", "p:p", "img", "run"]
