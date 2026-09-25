import os
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.generated import ActivationScript, module_init_snippet
from mainboard.engines.compile.generated import activation as activation_module

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
    runtime = shlex.join([sys.executable, "-m", "mainboard.runtime.activation"])
    assert f'eval "$({runtime})"' in text
    assert text.index("export FOO=bar") < text.index(runtime)


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell sources the activation")
def test_sourcing_the_script_finishes_with_what_the_runtime_step_adds(
    tmp_path: Path, posix_bash: str
) -> None:
    """The prefix pixi's hook entered gets its build search path from the runtime step."""
    prefix = tmp_path / "prefix"
    (prefix / "lib" / "pkgconfig").mkdir(parents=True)
    hook = f"export CONDA_PREFIX={shlex.quote(str(prefix))}"
    path = ActivationScript(tmp_path / "activate.sh", hook=hook).write({})
    result = subprocess.run(
        [posix_bash, "-c", f'source {shlex.quote(str(path))} && printf %s "$PKG_CONFIG_PATH"'],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert result.stdout == str(prefix / "lib" / "pkgconfig"), result.stderr


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


def test_render_preserves_module_order_and_omits_empty_version_slash(tmp_path: Path) -> None:
    script = ActivationScript(tmp_path / "activate.sh", hook="").render(
        {"cuda": "13.0", "gcc": ""}
    )
    assert "module load cuda/13.0 gcc || return $?\n" in script


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("unset -f module", 1),
        ("module() { return 7; }", 7),
        ("module() { case $1 in load) return 9;; esac; }", 9),
        ("module() { return 0; }", 0),
    ],
)
def test_declared_modules_must_load_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, posix_bash: str, setup: str, expected: int
) -> None:
    monkeypatch.setattr(activation_module, "module_init_snippet", lambda: ":")
    path = ActivationScript(tmp_path / "activate.sh", hook="export READY=yes").write(
        {"cuda": "13.0"}
    )
    command = f"PATH=''; {setup}; source {shlex.quote(path.as_posix())} || exit $?"
    command += '\n[ "$READY" = yes ]'
    result = subprocess.run(
        [posix_bash, "--noprofile", "--norc", "-c", command],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == expected
    if expected == 1:
        assert "require a working module system" in result.stderr
