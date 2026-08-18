from pathlib import Path

import pytest
from mainboard import Manifest, MissionError, load
from mainboard.manifest import EnvMode, Guardrail

from ..conftest import MANIFEST


def test_load_renders_and_validates_the_full_fixture(workspace: Path) -> None:
    manifest = load(workspace / MANIFEST)
    assert isinstance(manifest, Manifest)
    assert manifest.workspace.name == "lab"
    assert manifest.vars["scratch"] == "/scratch/lab"
    assert manifest.containers["ngc"].binds == ["/scratch/lab"]
    assert manifest.envs["serving"].system == {"cuda": "13.0"}


def test_vars_render_in_declaration_order_and_reach_later_tables(workspace: Path) -> None:
    manifest = load(workspace / MANIFEST)
    assert manifest.vars["station"].startswith(("linux-", "macos-"))


def test_submit_time_expressions_pass_through_unrendered(workspace: Path) -> None:
    manifest = load(workspace / MANIFEST)
    assert manifest.profile("miyabi-g").defaults.mem_gb == "min(100, attempt * 50)"


def test_missing_manifest_and_bad_toml_fail_with_named_errors(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="no manifest"):
        load(tmp_path / MANIFEST)
    (tmp_path / MANIFEST).write_text("workspace = [broken")
    with pytest.raises(MissionError, match="not valid TOML"):
        load(tmp_path / MANIFEST)


def test_toml_1_1_syntax_parses(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).write_text(
        """[workspace]
name = "lab"

[containers.ngc]
image = "nvcr.io/nvidia/pytorch:25.06-py3"
binds = [
    "/scratch:/scratch",
]
runtime = "apptainer"
"""
    )
    manifest = load(tmp_path / MANIFEST)
    assert manifest.containers["ngc"].runtime == "apptainer"


def test_template_failures_name_their_spot(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).write_text(
        '[workspace]\nname = "lab"\n\n[vars]\nbroken = "{{ nope }}"\n'
    )
    with pytest.raises(MissionError, match=r"vars\.broken"):
        load(tmp_path / MANIFEST)


def test_schema_failures_point_at_the_file(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).write_text('[workspace]\nname = "lab"\n\n[hosts.gold]\nkind = 3\n')
    with pytest.raises(MissionError, match="failed validation"):
        load(tmp_path / MANIFEST)


def test_container_defaults_carry_both_guardrails(workspace: Path) -> None:
    container = load(workspace / MANIFEST).containers["ngc"]
    assert container.env_mode is EnvMode.VENV_SYSTEM_SITE
    assert set(container.guardrails) == {
        Guardrail.UNSET_PIP_CONSTRAINT,
        Guardrail.PIN_SYSTEM_PACKAGES,
    }
