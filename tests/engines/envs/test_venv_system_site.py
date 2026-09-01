from pathlib import PurePosixPath

import pytest

from mainboard.engines import VenvSystemSite
from mainboard.manifest import Guardrail

_PREFIX = PurePosixPath("/prefix")
_SOURCE = 'source "/prefix/bin/activate"'


def test_provision_argv_lays_a_venv_over_the_images_system_site_packages() -> None:
    """The image's tuned wheels stay visible, so provisioning only ever adds on top."""
    assert VenvSystemSite.provision_argv(_PREFIX) == [
        ["python3", "-m", "venv", "--system-site-packages", "/prefix"]
    ]
    assert VenvSystemSite.provision_argv(_PREFIX, python="python3.14") == [
        ["python3.14", "-m", "venv", "--system-site-packages", "/prefix"]
    ]


@pytest.mark.parametrize(
    ("guardrails", "snippet"),
    [
        pytest.param([], _SOURCE, id="no-guardrail-leaves-the-inherited-environment-alone"),
        pytest.param(
            [Guardrail.UNSET_PIP_CONSTRAINT],
            f"{_SOURCE}\nunset PIP_CONSTRAINT",
            id="an-ngc-image-bakes-a-pin-that-has-to-be-cleared-at-activation",
        ),
        pytest.param(
            [Guardrail.PIN_SYSTEM_PACKAGES], _SOURCE, id="an-unrelated-guardrail-changes-nothing"
        ),
    ],
)
def test_activation_sources_the_venv_and_clears_only_the_pin_it_is_asked_to(
    guardrails: list[Guardrail], snippet: str
) -> None:
    assert VenvSystemSite.activation_snippet(_PREFIX, guardrails=guardrails) == snippet
