import os
from pathlib import Path

import pytest

from mainboard.manifest.render.interpolate import Interpolator


def test_num_cpus_counts_the_cpus_this_process_may_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3)
    rendered = Interpolator(tmp_path).rendered({"jobs": "{{ num_cpus() }}"})
    assert rendered["jobs"] == "3"
