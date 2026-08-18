from pathlib import Path

from mainboard.engines import VenvSystemSite
from mainboard.manifest import Guardrail


def test_provision_argv_is_a_system_site_venv() -> None:
    assert VenvSystemSite.provision_argv(Path("/prefix")) == [
        ["python3", "-m", "venv", "--system-site-packages", "/prefix"]
    ]
    assert VenvSystemSite.provision_argv(Path("/prefix"), python="python3.14") == [
        ["python3.14", "-m", "venv", "--system-site-packages", "/prefix"]
    ]


def test_activation_snippet_sources_the_venv() -> None:
    assert VenvSystemSite.activation_snippet(Path("/prefix")) == 'source "/prefix/bin/activate"'


def test_activation_snippet_unsets_pip_constraint_when_guarded() -> None:
    snippet = VenvSystemSite.activation_snippet(
        Path("/prefix"), guardrails=[Guardrail.UNSET_PIP_CONSTRAINT]
    )
    assert snippet == 'source "/prefix/bin/activate"\nunset PIP_CONSTRAINT'


def test_activation_snippet_ignores_unrelated_guardrails() -> None:
    snippet = VenvSystemSite.activation_snippet(
        Path("/prefix"), guardrails=[Guardrail.PIN_SYSTEM_PACKAGES]
    )
    assert "unset" not in snippet
