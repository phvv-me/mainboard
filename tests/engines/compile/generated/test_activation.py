from typing import TYPE_CHECKING

from mainboard.engines.compile.generated import ActivationScript, module_init_snippet

if TYPE_CHECKING:
    from pathlib import Path


def test_module_init_snippet_tries_every_candidate_and_stops_at_the_first() -> None:
    """The Lmod init is sourced before any load.

    `module` is a shell function and is undefined in a PBS non-login shell, so a job has to
    source the init first, and a host shipping none degrades to a no-op.
    """
    snippet = module_init_snippet(("/a/init.sh", "/b/init.sh"))
    assert snippet.startswith("for _modinit in /a/init.sh /b/init.sh; do")
    assert "&& break; done; unset _modinit" in snippet


def test_the_written_script_loads_the_modules_applies_the_hook_and_exports_the_stage(
    tmp_path: Path,
) -> None:
    """One `source` sets the whole runtime up.

    A shell reaches an npm-installed tool exactly like a conda one.
    """
    linked = tmp_path / "node modules" / ".bin"
    path = tmp_path / "activate.sh"

    written = ActivationScript(path, hook="\n  export FOO=bar  \n", binaries=[linked]).write(
        {"singularity": "4.2.1", "gcc": "13.2.0"}
    )

    assert written == path
    text = path.read_text()
    assert "module purge" in text
    assert "module load singularity/4.2.1 gcc/13.2.0" in text
    assert "command -v module" in text
    assert "export FOO=bar" in text
    assert "_mainboard_nounset=1" in text
    assert "set -u" in text
    assert f"export PATH='{linked}':\"$PATH\"" in text


def test_render_omits_every_block_the_host_declared_nothing_for(tmp_path: Path) -> None:
    """A bare workspace's activation touches nothing it does not own.

    With no modules the script never purges whatever stack the surrounding job had loaded,
    and with nothing installed beside pixi it exports no PATH of its own.
    """
    script = ActivationScript(tmp_path / "activate.sh", hook="export FOO=bar").render({})
    assert "module purge" not in script
    assert "module load" not in script
    assert "export PATH=" not in script
    assert "export FOO=bar" in script
