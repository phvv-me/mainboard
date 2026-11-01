import os
from pathlib import Path
from types import SimpleNamespace

import gpuhunt
import pytest

from mainboard.costs import Ledger
from mainboard.state import Lake

from ...costs.builders import landings
from .constants import KEY_SUFFIXES


def _item(provider: str, price: float) -> SimpleNamespace:
    """One RTX 4090 offer as gpuhunt lists it."""
    return SimpleNamespace(
        provider=provider,
        gpu_name="RTX4090",
        gpu_count=1,
        price=price,
        spot=False,
        location="fixture",
        instance_name=f"{provider}-4090",
    )


@pytest.fixture
def bare(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The workspace with no provider key in the environment or in its `.env`, run from inside.

    Four RTX 4090 offers at equal and lower prices stand in for gpuhunt's catalog.
    """
    for name in [name for name in os.environ if name.endswith(KEY_SUFFIXES)]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(workspace)
    offers = [
        _item("vastai", 0.5),
        _item("runpod", 0.5),
        _item("lambdalabs", 0.5),
        _item("aws", 0.4),
    ]
    monkeypatch.setattr(gpuhunt, "query", lambda **_filters: offers)
    return workspace


@pytest.fixture
def measured(bare: Path) -> Path:
    """That workspace after five good runpod landings and three good, two slow vast ones."""
    session = Lake.at(bare).ready().session()
    ledger = Ledger(session)
    rows = [
        *landings("runpod", landed=5, gpu="RTX4090", ready_s=40.0),
        *landings("vast", landed=3, slow=2, gpu="RTX4090", ready_s=120.0),
    ]
    for row in rows:
        ledger.record(row)
    session.close()
    return bare
