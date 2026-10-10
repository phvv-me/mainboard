"""A prefix sweep waiting out an environment build holds no pin lock, which a launch waits on."""

import threading
from pathlib import Path

import pytest
from filelock import FileLock

from mainboard.core.retention import Retention
from mainboard.engines.compile import Prefixes
from mainboard.manifest import Header, Manifest


@pytest.fixture
def prefixes(tmp_path: Path) -> Prefixes:
    built = Prefixes(tmp_path, Manifest(workspace=Header(name="locks")))
    built.base.mkdir(parents=True)
    built.retention.base.mkdir(parents=True)
    return built


def test_a_sweep_waiting_out_a_build_does_not_make_a_pin_wait(prefixes: Prefixes) -> None:
    sweep = threading.Thread(target=prefixes.prune, kwargs={"live": (), "inventory": Retention()})
    # The lock every release's builds take, one level above the environment's own directory.
    with FileLock(prefixes.base.parent / ".sync.lock"):
        sweep.start()
        sweep.join(0.5)
        assert sweep.is_alive()
        with FileLock(prefixes.retention.base / ".pin.lock").acquire(timeout=0):
            pass
    sweep.join(10)
    assert not sweep.is_alive()
