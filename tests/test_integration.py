from typing import TYPE_CHECKING

import pytest

from mainboard import Resolver, load
from mainboard.context import admit
from mainboard.dispatch.wrapping import wrap
from mainboard.engines.runtimes import resolve

from .conftest import MANIFEST

if TYPE_CHECKING:
    from pathlib import Path


def test_the_whole_promise_composes(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One manifest resolves into a containerized, module-loaded remote command.

    The end-to-end seam no single subsystem owns: manifest to plan, plan to
    container argv, argv into the wrapped shell line a PBS host executes, with
    the queue policy consulted on the way. The fused tool's promise in one
    assertion block.
    """
    manifest = load(workspace / MANIFEST)
    plan = Resolver(manifest).plan("miyabi-g")
    admit(plan.profile, queue="short-g", walltime="06:00:00", mem_gb=100)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/apptainer")
    assert plan.container is not None
    runtime = resolve(plan.container.runtime)()
    root = "/work/xg25g007/x10537/projects"
    line = wrap(
        plan,
        root,
        command="python -m experiments.run",
        containerize=lambda argv: runtime.command(
            plan.container, prefix_bind=plan.prefix(root), argv=argv
        ),
    )
    assert f"cd {root}" in line
    assert "module load singularity/4.2.1" in line
    assert "apptainer exec --nv" in line
    assert "nvcr.io/nvidia/pytorch:25.06-py3" in line
    assert "python -m experiments.run" in line


def test_a_plan_carries_its_container_stage_or_refuses_to_be_wrapped_without_one(
    workspace: Path,
) -> None:
    """A bare host skips the stage entirely, and a containerized one will not go without it."""
    manifest = load(workspace / MANIFEST)
    line = wrap(Resolver(manifest).plan("gold"), "/home/pedro/projects", command="nvidia-smi")
    assert "apptainer" not in line and "docker" not in line
    assert "nvidia-smi" in line
    with pytest.raises(LookupError, match="containerized"):
        wrap(Resolver(manifest).plan("miyabi-g"), "/work/projects", command="true")
