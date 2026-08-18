from typing import ClassVar

from .docker import DockerCompatible


class Podman(DockerCompatible):
    """Wraps argv for `podman run`.

    Podman has no `--gpus` flag. NVIDIA's own guidance for Podman is the Container
    Device Interface, exposed through `--device nvidia.com/gpu=all`.
    """

    binary: ClassVar[str] = "podman"

    @classmethod
    def gpu_flags(cls) -> list[str]:
        return ["--device", "nvidia.com/gpu=all"]
