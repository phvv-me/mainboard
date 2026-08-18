from pathlib import Path

from mainboard.engines import PixiPrefix
from mainboard.manifest import Guardrail


def test_provision_argv_is_a_locked_pixi_install() -> None:
    assert PixiPrefix.provision_argv(Path("/prefix")) == [["pixi", "install", "--locked"]]
    assert PixiPrefix.provision_argv(Path("/prefix"), python="python3.14") == [
        ["pixi", "install", "--locked"]
    ]


def test_activation_snippet_prefers_the_shell_hook_over_a_path_prepend() -> None:
    snippet = PixiPrefix.activation_snippet(Path("/prefix"))
    assert snippet == (
        """if [ -f "/prefix/activate.sh" ]; then
    source "/prefix/activate.sh"
else
    export PATH="/prefix/bin:$PATH"
fi"""
    )


def test_activation_snippet_is_unaffected_by_guardrails() -> None:
    guarded = PixiPrefix.activation_snippet(
        Path("/prefix"), guardrails=[Guardrail.UNSET_PIP_CONSTRAINT]
    )
    assert guarded == PixiPrefix.activation_snippet(Path("/prefix"))
