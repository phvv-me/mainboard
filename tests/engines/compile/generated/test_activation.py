from typing import TYPE_CHECKING

from mainboard.engines.compile.generated import ActivationScript, module_init_snippet

if TYPE_CHECKING:
    from pathlib import Path


def test_module_init_snippet_tries_every_candidate_and_stops_at_the_first() -> None:
    snippet = module_init_snippet(("/a/init.sh", "/b/init.sh"))
    assert snippet.startswith("for _modinit in /a/init.sh /b/init.sh; do")
    assert "&& break; done; unset _modinit" in snippet


def test_render_omits_the_module_block_without_modules() -> None:
    script = ActivationScript(path=None, hook="export FOO=bar").render({})
    assert "module purge" not in script
    assert "module load" not in script
    assert "export FOO=bar" in script


def test_render_formats_modules_as_name_slash_version_and_guards_the_load() -> None:
    script = ActivationScript(path=None, hook="export FOO=bar").render(
        {"singularity": "4.2.1", "gcc": "13.2.0"}
    )
    assert "module purge" in script
    assert "module load singularity/4.2.1 gcc/13.2.0" in script
    assert "command -v module" in script


def test_render_strips_the_hook_and_relaxes_then_restores_nounset() -> None:
    script = ActivationScript(path=None, hook="\n  export FOO=bar  \n").render({})
    assert "export FOO=bar" in script
    assert "_mainboard_nounset=1" in script
    assert "set -u" in script


def test_render_omits_the_path_export_without_second_stage_directories() -> None:
    assert "export PATH=" not in ActivationScript(path=None, hook="export FOO=bar").render({})


def test_render_exports_second_stage_directories_ahead_of_the_env(tmp_path: Path) -> None:
    """A shell sourcing the script reaches an npm-installed tool, not only a conda one."""
    linked = tmp_path / "node modules" / ".bin"
    script = ActivationScript(path=None, hook="export FOO=bar", binaries=[linked]).render({})
    assert f"export PATH='{linked}':\"$PATH\"" in script


def test_write_persists_the_rendered_script_and_returns_its_path(tmp_path: Path) -> None:
    path = tmp_path / "activate.sh"
    result = ActivationScript(path, hook="export FOO=bar").write({"cuda": "13.0"})
    assert result == path
    text = path.read_text()
    assert "module load cuda/13.0" in text
    assert "export FOO=bar" in text
