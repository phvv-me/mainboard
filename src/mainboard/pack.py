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

from patos import FrozenModel

from .core.errors import MissionError
from .core.host import current_platform
from .engines.compile.prefixes import digest_of
from .engines.compile.provisioner import environment_shard

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


def pack(
    root: Path, env: str, *, image: bool = False, sif: bool = False, push: str = ""
) -> Packed:
    """Build `env`'s artifacts here from the lock this workspace installed it from.

    root: the workspace root; `env` must have been installed here (`install`, `host setup`).
    image: also build an OCI image (docker).
    sif: also build an Apptainer SIF from that image.
    push: also push the image to this repository (`ghcr.io/<owner>/<name>`), tagged by digest.
    """
    started = time.monotonic()
    shard = root / environment_shard(env, root)
    manifest = shard / "pixi.toml"
    if not manifest.is_file():
        raise MissionError(f"{env!r} is not compiled here; run `mb install {env}` first")
    digest = digest_of(shard)
    platform = current_platform()
    out = shard.parent.parent / "packs"
    out.mkdir(parents=True, exist_ok=True)
    executable = out / f"{env}-{digest}-{platform}.sh"
    if not executable.is_file():
        # pixi-pack names an executable `.sh` whatever it is asked for, so the staged name is one.
        staged = executable.with_name(f".{executable.stem}.partial.sh")
        _run(
            "pixi",
            "exec",
            "pixi-pack",
            "--environment",
            env,
            "--platform",
            platform,
            "--ignore-pypi-non-wheel",
            "--create-executable",
            "--use-cache",
            str(Path.home() / ".cache" / "mb" / "pack"),
            "--output-file",
            str(staged),
            str(manifest),
        )
        staged.replace(executable)
    tag = f"mb-{env}:{digest}" if image or sif or push else ""
    if tag:
        context = out / f"{env}-{digest}.docker"
        context.mkdir(exist_ok=True)
        (context / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8", newline="\n")
        linked = context / "pack.sh"
        if not linked.exists():
            os.link(executable, linked)
        _run("docker", "build", "-q", "-t", tag, str(context))
    pushed = f"{push}:{digest}" if push else ""
    if pushed:
        _run("docker", "tag", tag, pushed)
        _run("docker", "push", "-q", pushed)
    built = out / f"{env}-{digest}.sif" if sif else None
    if built is not None and not built.is_file():
        _run("apptainer", "build", "--force", str(built), f"docker-daemon://{tag}")
    return Packed(
        env=env,
        digest=digest,
        platform=platform,
        executable=str(executable),
        image=tag,
        pushed=pushed,
        sif=str(built or ""),
        seconds=round(time.monotonic() - started, 1),
    )


def _run(*argv: str) -> None:
    """Run one build step with its output on stderr, refusing a missing tool or a failure."""
    if shutil.which(argv[0]) is None:
        raise MissionError(f"`{argv[0]}` is not installed here, and this step needs it")
    status = subprocess.call(list(argv), stdout=2)
    if status:
        raise MissionError(f"`{' '.join(argv[:3])} ...` exited {status}")
