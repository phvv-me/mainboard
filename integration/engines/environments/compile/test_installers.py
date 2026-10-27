import subprocess
from pathlib import Path

import pytest

from mainboard.dispatch import Dialect
from mainboard.engines.compile import backend

# What a fetched installer does, minus the install: it records only whether the switch that keeps
# it out of the user's startup files reached it.
_INSTALLER = 'if [ -n "${%s:-}" ]; then touch "$HOME/startup-files-left-alone"; fi'


@pytest.mark.parametrize(
    ("command", "switch"),
    [
        pytest.param(backend.PixiEngine.INSTALLER, "PIXI_NO_PATH_UPDATE", id="the-pixi-installer"),
        pytest.param(Dialect.uv_bootstrap[1], "UV_NO_MODIFY_PATH", id="the-uv-installer"),
    ],
)
def test_an_installer_the_tool_runs_leaves_the_startup_files_to_the_dotfiles(
    command: str, switch: str, tmp_path: Path
) -> None:
    curl = tmp_path / "bin" / "curl"
    curl.parent.mkdir()
    curl.write_text(f"#!/bin/sh\ncat <<'EOF'\n{_INSTALLER % switch}\nEOF\n")
    curl.chmod(0o755)

    subprocess.run(
        ["sh", "-c", command],
        env={"PATH": f"{curl.parent}:/usr/bin:/bin", "HOME": str(tmp_path)},
        check=True,
    )

    assert (tmp_path / "startup-files-left-alone").exists()
