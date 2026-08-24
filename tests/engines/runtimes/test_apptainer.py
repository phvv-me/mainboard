from collections.abc import Callable

import pytest

from mainboard.engines import Apptainer
from mainboard.manifest import Container, Guardrail

_IMAGE = "nvcr.io/nvidia/pytorch:25.06-py3"


@pytest.mark.parametrize(
    ("installed", "available", "launcher"),
    [
        pytest.param((), False, "apptainer", id="a-host-shipping-neither"),
        pytest.param(("singularity",), True, "singularity", id="only-the-legacy-alias"),
        pytest.param(("apptainer",), True, "apptainer", id="only-the-maintained-successor"),
        pytest.param(
            ("apptainer", "singularity"), True, "apptainer", id="both-prefers-the-successor"
        ),
    ],
)
def test_either_apptainer_or_its_singularity_alias_makes_a_host_usable(
    installed: tuple[str, ...],
    launcher: str,
    which: Callable[..., None],
    *,
    available: bool,
) -> None:
    """Apptainer is the maintained successor of Singularity and stays command-line compatible
    with it, so a host that only ships the legacy binary is still usable."""
    which(*installed)
    assert Apptainer.is_available() is available
    assert Apptainer.launcher() == launcher


@pytest.mark.parametrize(
    ("container", "argv"),
    [
        pytest.param(
            Container(
                image=_IMAGE,
                binds=["/scratch"],
                workdir="/workspace",
                passthrough=["HF_TOKEN"],
                guardrails=[Guardrail.UNSET_PIP_CONSTRAINT],
            ),
            [
                "apptainer",
                "exec",
                "--nv",
                "--bind",
                "/scratch",
                "--bind",
                "/host/prefix:/prefix",
                "--pwd",
                "/workspace",
                "--env",
                "HF_TOKEN",
                _IMAGE,
                "env",
                "-u",
                "PIP_CONSTRAINT",
                "python",
                "run.py",
            ],
            id="gpu-binds-workdir-passthrough-and-a-guardrail",
        ),
        pytest.param(
            Container(image=_IMAGE, gpus=False, guardrails=[]),
            ["apptainer", "exec", "--bind", "/host/prefix:/prefix", _IMAGE, "python", "run.py"],
            id="nothing-declared-beyond-the-image",
        ),
    ],
)
def test_the_launcher_argv_wraps_the_command_in_what_the_container_declared(
    container: Container, argv: list[str], which: Callable[..., None]
) -> None:
    """No runtime flag can unset a variable baked into the image, so `UNSET_PIP_CONSTRAINT` is
    enforced by wrapping the exec'd command in a plain `env -u`."""
    which()
    assert (
        Apptainer.command(container, prefix_bind="/host/prefix:/prefix", argv=["python", "run.py"])
        == argv
    )
