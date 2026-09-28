import os
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from mainboard.engines.compile.generated import activation as activation_module
from mainboard.engines.compile.generated import module_init_snippet
from mainboard.engines.compile.generated.activation import activation, write

if TYPE_CHECKING:
    from pathlib import Path


def test_module_init_snippet_tries_every_candidate_and_stops_at_the_first() -> None:
    snippet = module_init_snippet(("/a/init.sh", "/b/init.sh"))
    assert snippet.startswith("for _modinit in /a/init.sh /b/init.sh; do")
    assert "&& break; done; unset _modinit" in snippet


def test_the_written_activation_loads_the_modules_applies_the_hook_and_exports_the_stage(
    tmp_path: Path,
) -> None:
    linked = tmp_path / "node modules" / ".bin"
    path = tmp_path / "activate.sh"

    written = write(
        path,
        "\n  export FOO=bar  \n",
        modules={"singularity": "4.2.1", "gcc": "13.2.0"},
        binaries=[linked],
    )

    assert written == path
    text = path.read_text()
    assert "module purge" in text
    assert "module load singularity/4.2.1 gcc/13.2.0" in text
    assert "command -v module" in text
    assert "export FOO=bar" in text
    assert "_mb_nounset=1" in text
    assert "set -u" in text
    assert f"export PATH='{linked}':\"$PATH\"" in text
    runtime = shlex.join([sys.executable, "-m", "mainboard.runtime.activation"])
    assert f'eval "$({runtime})"' in text
    assert text.index("export FOO=bar") < text.index(runtime)


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell sources the activation")
def test_sourcing_the_activation_finishes_with_what_the_runtime_step_adds(
    tmp_path: Path, posix_bash: str
) -> None:
    prefix = tmp_path / "prefix"
    (prefix / "lib" / "pkgconfig").mkdir(parents=True)
    hook = f"export CONDA_PREFIX={shlex.quote(str(prefix))}"
    path = write(tmp_path / "activate.sh", hook, modules={})
    result = subprocess.run(
        [posix_bash, "-c", f'source {shlex.quote(str(path))} && printf %s "$PKG_CONFIG_PATH"'],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
    )
    assert result.stdout == str(prefix / "lib" / "pkgconfig"), result.stderr


def test_nothing_is_written_for_a_block_the_host_declared_nothing_for() -> None:
    text = activation("export FOO=bar")
    assert "module purge" not in text
    assert "module load" not in text
    assert "export PATH=" not in text
    assert "export FOO=bar" in text
    assert "runtime.activation" not in activation("", runtime=False)


def test_modules_keep_their_order_and_a_bare_name_has_no_version_slash() -> None:
    written = activation_module.module_specs({"cuda": "13.0", "gcc": ""})
    assert "module load cuda/13.0 gcc ||" in activation("", modules=written)


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("unset -f module", 1),
        ("module() { return 7; }", 7),
        ("module() { case $1 in load) return 9;; esac; }", 9),
        ("module() { return 0; }", 0),
    ],
)
@pytest.mark.parametrize("entered", ["sourced", "run"])
def test_declared_modules_must_load_before_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    posix_bash: str,
    setup: str,
    expected: int,
    entered: str,
) -> None:
    """Sourced, a failure returns its status to the caller; run by `bash -c`, it exits with it."""
    monkeypatch.setattr(activation_module, "module_init_snippet", lambda: ":")
    path = write(tmp_path / "activate.sh", "export READY=yes", modules={"cuda": "13.0"})
    body = (
        f"source {shlex.quote(path.as_posix())} || exit $?"
        if entered == "sourced"
        else path.read_text(encoding="utf-8")
    )
    command = f"PATH=''; {setup}; {body}\n[ \"$READY\" = yes ]"
    result = subprocess.run(
        [posix_bash, "--noprofile", "--norc", "-c", command],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == expected
    if expected == 1:
        assert "require a working module system" in result.stderr
