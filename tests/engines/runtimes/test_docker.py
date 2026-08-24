import pytest

from mainboard.engines import Docker
from mainboard.manifest import Container, Guardrail


@pytest.mark.parametrize(
    ("container", "argv"),
    [
        pytest.param(
            Container(
                image="img",
                binds=["/data"],
                workdir="/app",
                passthrough=["A", "B"],
                guardrails=[Guardrail.UNSET_PIP_CONSTRAINT],
            ),
            [
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
            ],
            id="gpu-binds-workdir-passthrough-and-a-guardrail",
        ),
        pytest.param(
            Container(image="img", gpus=False, guardrails=[]),
            ["docker", "run", "--rm", "-v", "/h:/p", "img", "run"],
            id="nothing-declared-beyond-the-image",
        ),
    ],
)
def test_the_run_argv_carries_what_the_container_declared(
    container: Container, argv: list[str]
) -> None:
    assert Docker.command(container, prefix_bind="/h:/p", argv=["run"]) == argv
