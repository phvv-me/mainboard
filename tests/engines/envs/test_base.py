from typing import cast

import pytest

from mainboard import MissionError
from mainboard.engines import EnvBackend, PixiPrefix, VenvSystemSite
from mainboard.engines.envs import resolve
from mainboard.manifest import EnvMode, Guardrail


def test_resolve_finds_the_backend_each_mode_declares_and_lists_them_when_it_cannot() -> None:
    assert resolve(EnvMode.VENV_SYSTEM_SITE) is VenvSystemSite
    assert resolve(EnvMode.PIXI_PREFIX) is PixiPrefix
    with pytest.raises(MissionError, match="known modes"):
        resolve(cast("EnvMode", "bogus"))


@pytest.mark.parametrize(
    ("guardrails", "pinned"),
    [
        pytest.param([Guardrail.PIN_SYSTEM_PACKAGES], True, id="the-guardrail-itself"),
        pytest.param([], False, id="no-guardrail-at-all"),
        pytest.param([Guardrail.UNSET_PIP_CONSTRAINT], False, id="an-unrelated-guardrail"),
    ],
)
def test_pins_system_packages_reads_only_its_own_guardrail(
    guardrails: list[Guardrail], *, pinned: bool
) -> None:
    """A plain marker an env backend reads, never a container runtime flag."""
    assert EnvBackend.pins_system_packages(guardrails) is pinned
