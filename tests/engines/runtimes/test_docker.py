import pytest

from mainboard.engines import Docker, Podman
from mainboard.engines.runtimes.docker import DockerCompatible
from mainboard.manifest import Container, Guardrail

_DECLARED = Container(
    image="img",
    binds=["/data"],
    workdir="/app",
    passthrough=["A", "B"],
    guardrails=[Guardrail.UNSET_PIP_CONSTRAINT],
)
_TAIL = "-v /data -v /h:/p -w /app --env A --env B img env -u PIP_CONSTRAINT run"


@pytest.mark.parametrize(
    ("runtime", "container", "argv"),
    [
        pytest.param(
            Docker,
            _DECLARED,
            f"docker run --rm --gpus all {_TAIL}",
            id="docker-gpu-binds-workdir-passthrough-and-a-guardrail",
        ),
        pytest.param(
            Podman,
            _DECLARED,
            f"podman run --rm --device nvidia.com/gpu=all {_TAIL}",
            id="podman-reaches-a-gpu-through-the-container-device-interface",
        ),
        pytest.param(
            Docker,
            Container(image="img", gpus=False, guardrails=[]),
            "docker run --rm -v /h:/p img run",
            id="nothing-declared-beyond-the-image",
        ),
        pytest.param(
            Podman,
            Container(image="img", gpus=False),
            "podman run --rm -v /h:/p img env -u PIP_CONSTRAINT run",
            id="podman-without-a-gpu-carries-no-device-flag",
        ),
    ],
)
def test_the_run_argv_carries_what_the_container_declared(
    runtime: type[DockerCompatible], container: Container, argv: str
) -> None:
    assert runtime.command(container, prefix_bind="/h:/p", argv=["run"]) == argv.split()
