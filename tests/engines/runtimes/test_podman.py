from collections.abc import Callable

from mainboard.engines import Podman
from mainboard.manifest import Container


def test_is_available_uses_plain_which(which: Callable[..., None]) -> None:
    which()
    assert not Podman.is_available()
    which("podman")
    assert Podman.is_available()


def test_command_uses_the_cdi_device_flag_for_gpu() -> None:
    container = Container(image="img")
    argv = Podman.command(container, prefix_bind="p:p", argv=["run"])
    assert argv[:5] == ["podman", "run", "--rm", "--device", "nvidia.com/gpu=all"]
    assert argv[-1] == "run"


def test_command_without_gpu_has_no_device_flag() -> None:
    container = Container(image="img", gpus=False)
    argv = Podman.command(container, prefix_bind="p:p", argv=["run"])
    assert argv == [
        "podman",
        "run",
        "--rm",
        "-v",
        "p:p",
        "img",
        "env",
        "-u",
        "PIP_CONSTRAINT",
        "run",
    ]
