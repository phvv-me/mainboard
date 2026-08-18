from collections.abc import Callable

from mainboard.engines import Apptainer
from mainboard.manifest import Container, Guardrail


def test_is_available_accepts_either_apptainer_or_singularity(which: Callable[..., None]) -> None:
    which()
    assert not Apptainer.is_available()
    which("singularity")
    assert Apptainer.is_available()
    which("apptainer")
    assert Apptainer.is_available()


def test_launcher_prefers_apptainer_over_the_singularity_alias(
    which: Callable[..., None],
) -> None:
    which()
    assert Apptainer.launcher() == "apptainer"
    which("singularity")
    assert Apptainer.launcher() == "singularity"
    which("apptainer", "singularity")
    assert Apptainer.launcher() == "apptainer"


def test_command_with_gpu_binds_workdir_passthrough_and_guardrail(
    which: Callable[..., None],
) -> None:
    which()
    container = Container(
        image="nvcr.io/nvidia/pytorch:25.06-py3",
        binds=["/scratch"],
        workdir="/workspace",
        passthrough=["HF_TOKEN"],
        guardrails=[Guardrail.UNSET_PIP_CONSTRAINT],
    )
    argv = Apptainer.command(
        container, prefix_bind="/host/prefix:/prefix", argv=["python", "run.py"]
    )
    assert argv == [
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
        "nvcr.io/nvidia/pytorch:25.06-py3",
        "env",
        "-u",
        "PIP_CONSTRAINT",
        "python",
        "run.py",
    ]


def test_command_without_gpu_workdir_passthrough_or_guardrails(
    which: Callable[..., None],
) -> None:
    which()
    container = Container(image="x", gpus=False, guardrails=[])
    argv = Apptainer.command(container, prefix_bind="p:p", argv=["echo", "hi"])
    assert argv == ["apptainer", "exec", "--bind", "p:p", "x", "echo", "hi"]
