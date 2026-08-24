import pytest

from mainboard.engines import Podman
from mainboard.manifest import Container


@pytest.mark.parametrize(
    ("container", "argv"),
    [
        pytest.param(
            Container(image="img"),
            [
                "podman",
                "run",
                "--rm",
                "--device",
                "nvidia.com/gpu=all",
                "-v",
                "p:p",
                "img",
                "env",
                "-u",
                "PIP_CONSTRAINT",
                "run",
            ],
            id="a-gpu-reaches-podman-through-the-container-device-interface",
        ),
        pytest.param(
            Container(image="img", gpus=False),
            ["podman", "run", "--rm", "-v", "p:p", "img", "env", "-u", "PIP_CONSTRAINT", "run"],
            id="no-gpu-carries-no-device-flag",
        ),
    ],
)
def test_the_run_argv_swaps_dockers_gpu_flag_for_the_cdi_device(
    container: Container, argv: list[str]
) -> None:
    """Podman has no `--gpus`, and NVIDIA's own guidance for it is the device interface."""
    assert Podman.command(container, prefix_bind="p:p", argv=["run"]) == argv
