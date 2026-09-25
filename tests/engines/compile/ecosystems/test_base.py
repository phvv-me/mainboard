from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.engines.compile.ecosystems import Ecosystem, Go

if TYPE_CHECKING:
    from ..support import Bind


def test_the_default_frozen_contract_passes_an_empty_table_and_refuses_a_declared_one(
    bind: Bind,
) -> None:
    """A toolchain naming no locked mode can still ship a table that declares nothing.

    Read through a real implementation's binding, since every enrolled one refines the default,
    and a stand-in subclass would enroll in the registry every other test reads.
    """
    assert Ecosystem.frozen_inputs(bind(Go, {})) == ()
    with pytest.raises(MissionError, match=r"\[go\] has no frozen installation contract"):
        Ecosystem.frozen_inputs(bind(Go, {"deps": {"example.com/tool": "v1.4.0"}}))
