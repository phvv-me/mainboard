from pathlib import PurePosixPath

from mainboard.engines import PixiPrefix
from mainboard.manifest import Guardrail

_PREFIX = PurePosixPath("/prefix")


def test_provision_argv_is_a_locked_pixi_install_whatever_interpreter_is_offered() -> None:
    locked = [["pixi", "install", "--locked"]]
    assert PixiPrefix.provision_argv(_PREFIX) == locked
    assert PixiPrefix.provision_argv(_PREFIX, python="python3.14") == locked


def test_activation_prefers_the_shell_hook_over_a_path_prepend_whatever_is_guarded() -> None:
    snippet = PixiPrefix.activation_snippet(_PREFIX)
    assert snippet == (
        """if [ -f "/prefix/activate.sh" ]; then
    source "/prefix/activate.sh"
else
    export PATH="/prefix/bin:$PATH"
fi"""
    )
    assert PixiPrefix.activation_snippet(_PREFIX, guardrails=[Guardrail.UNSET_PIP_CONSTRAINT]) == (
        snippet
    )
