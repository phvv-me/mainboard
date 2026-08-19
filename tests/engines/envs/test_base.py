from typing import cast

import pytest

from mainboard import MissionError
from mainboard.engines import EnvBackend, PixiPrefix, VenvSystemSite
from mainboard.engines.envs import resolve
from mainboard.manifest import EnvMode, Guardrail


def test_resolve_finds_the_backend_declared_for_each_env_mode() -> None:
    assert resolve(EnvMode.VENV_SYSTEM_SITE) is VenvSystemSite
    assert resolve(EnvMode.PIXI_PREFIX) is PixiPrefix


def test_resolve_unknown_mode_lists_known_modes() -> None:
    with pytest.raises(MissionError, match="known modes"):
        resolve(cast("EnvMode", "bogus"))


def test_pins_system_packages_reads_the_guardrail() -> None:
    assert EnvBackend.pins_system_packages([Guardrail.PIN_SYSTEM_PACKAGES])
    assert not EnvBackend.pins_system_packages([])
    assert not EnvBackend.pins_system_packages([Guardrail.UNSET_PIP_CONSTRAINT])
