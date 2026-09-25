from typing import ClassVar

from .docker import DockerCompatible


class Podman(DockerCompatible):
    """Wraps argv for `podman run`, exposing GPUs through NVIDIA's CDI since it has no `--gpus`."""

    binary: ClassVar[str] = "podman"

    @classmethod
    def gpu_flags(cls) -> list[str]:
        return ["--device", "nvidia.com/gpu=all"]
