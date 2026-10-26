# Artifacts built from one environment's lock, for a machine that should not install it itself:
# a self-extracting pixi-pack executable, an OCI image and an Apptainer SIF.
#
# The lock stays the one truth; each artifact is named by the environment's digest (the address
# its prefix already has), so the same lock never builds twice and a changed one never reuses a
# stale artifact. Built where it will be used, a Linux host with fast package mirrors: from this
# workstation the upload was the bottleneck (4 GB at ~14 MB/s against a 43 s cold install).
#
# Measured on crimson for the lean `gpu` environment (5.7 GB): the executable 3.96 GB, the image
# 3.0 GB compressed and 6.1 GB unpacked, starting `import torch` on CUDA in 4 s.

import os
import shutil
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs pixi-pack, docker and apptainer with fixed argv since=2026-09-29
import time
from pathlib import Path

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict

from ..core.errors import MissionError
from ..core.execution.host import current_platform
from .compile.prefixes import digest_of
from .compile.provisioner import environment_shard

# The image: the pack unpacked at its final path in a throwaway stage, then copied alone onto
# slim Debian. CUDA comes from the environment's own NVIDIA wheels and the host driver from the
# NVIDIA runtime (`--gpus all`), so no CUDA base image is needed. pixi-unpack wants CA
# certificates even offline; only the throwaway stage has them.
_DOCKERFILE = """FROM debian:bookworm-slim AS unpack
RUN apt-get update -qq && apt-get install -y -qq ca-certificates >/dev/null
COPY pack.sh /pack.sh
RUN mkdir -p /opt/mb && cd /opt/mb && bash /pack.sh

FROM debian:bookworm-slim
COPY --from=unpack /opt/mb/env /opt/mb/env
ENV PATH=/opt/mb/env/bin:$PATH CONDA_PREFIX=/opt/mb/env
CMD ["python"]
"""


class Packed(FrozenModel):
    """What one `pack` left behind.

    executable: the self-extracting environment (`./<file>` unpacks `env/` and `activate.sh`).
    image: the local image tag, empty when none was asked for.
    pushed: where the image was pushed, empty when it was not.
    sif: the Apptainer image file, empty when none was asked for.
    seconds: how long this took.
    """

    env: str
    digest: str
    platform: str
    executable: str
    image: str = ""
    pushed: str = ""
    sif: str = ""
    seconds: float


@Parameter(name="*")
class Formats(FrozenModel):
    """The artifacts `pack` adds to the self-extracting executable it always builds."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    image: bool = False
    """also build an OCI image (docker)."""
    sif: bool = False
    """also build an Apptainer SIF from the image."""
    push: str = ""
    """also push the image to this repository, `ghcr.io/OWNER/NAME`, tagged by digest."""

    @property
    def flags(self) -> list[str]:
        """These formats as `mb pack` flags, for a host that packs for itself."""
        flags = [*(["--image"] if self.image else []), *(["--sif"] if self.sif else [])]
        return flags + (["--push", self.push] if self.push else [])

    @property
    def imaged(self) -> bool:
        """Whether any asked format needs the OCI image."""
        return self.image or self.sif or bool(self.push)


class _Build(FrozenModel):
    """One installed environment's artifacts, each named by the environment's digest."""

    env: str
    shard: Path
    digest: str
    platform: str

    @property
    def out(self) -> Path:
        """The folder every artifact of every environment lands in."""
        return self.shard.parent.parent / "packs"

    def executable(self) -> Path:
        """The self-extracting environment, built unless this digest already has one."""
        executable = self.out / f"{self.env}-{self.digest}-{self.platform}.sh"
        if executable.is_file():
            return executable
        self.out.mkdir(parents=True, exist_ok=True)
        # pixi-pack names an executable `.sh` whatever it is asked for, so the staged name is one.
        staged = executable.with_name(f".{executable.stem}.partial.sh")
        _run(
            "pixi",
            "exec",
            "pixi-pack",
            "--environment",
            self.env,
            "--platform",
            self.platform,
            "--ignore-pypi-non-wheel",
            "--create-executable",
            "--use-cache",
            str(Path.home() / ".cache" / "mb" / "pack"),
            "--output-file",
            str(staged),
            str(self.shard / "pixi.toml"),
        )
        staged.replace(executable)
        return executable

    def image(self, executable: Path) -> str:
        """The local tag of the OCI image wrapping `executable`."""
        tag = f"mb-{self.env}:{self.digest}"
        context = self.out / f"{self.env}-{self.digest}.docker"
        context.mkdir(exist_ok=True)
        (context / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8", newline="\n")
        linked = context / "pack.sh"
        if not linked.exists():
            os.link(executable, linked)
        _run("docker", "build", "-q", "-t", tag, str(context))
        return tag

    def pushed(self, tag: str, *, repository: str) -> str:
        """Where `tag` was pushed in `repository`, tagged by digest."""
        pushed = f"{repository}:{self.digest}"
        _run("docker", "tag", tag, pushed)
        _run("docker", "push", "-q", pushed)
        return pushed

    def sif(self, tag: str) -> str:
        """The Apptainer file built from `tag`, unless this digest already has one."""
        built = self.out / f"{self.env}-{self.digest}.sif"
        if not built.is_file():
            _run("apptainer", "build", "--force", str(built), f"docker-daemon://{tag}")
        return str(built)


def pack(root: Path, env: str, formats: Formats) -> Packed:
    """Build `env`'s artifacts here from the lock this workspace installed it from.

    root: the workspace root; `env` must have been installed here (`install`, `host setup`).
    """
    started = time.monotonic()
    shard = root / environment_shard(env, root)
    if not (shard / "pixi.toml").is_file():
        raise MissionError(f"{env!r} is not compiled here; run `mb install {env}` first")
    build = _Build(env=env, shard=shard, digest=digest_of(shard), platform=current_platform())
    executable = build.executable()
    tag = build.image(executable) if formats.imaged else ""
    return Packed(
        env=env,
        digest=build.digest,
        platform=build.platform,
        executable=str(executable),
        image=tag,
        pushed=build.pushed(tag, repository=formats.push) if formats.push else "",
        sif=build.sif(tag) if formats.sif else "",
        seconds=round(time.monotonic() - started, 1),
    )


def _run(*argv: str) -> None:
    """Run one build step with its output on stderr, refusing a missing tool or a failure."""
    if shutil.which(argv[0]) is None:
        raise MissionError(f"`{argv[0]}` is not installed here, and this step needs it")
    status = subprocess.call(list(argv), stdout=2)
    if status:
        raise MissionError(f"`{' '.join(argv[:3])} ...` exited {status}")
