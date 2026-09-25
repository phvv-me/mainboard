from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard.manifest.loading import load
from mainboard.trials.admission import admit, policy


def workspace(tmp_path: Path, table: str = "") -> Path:
    (tmp_path / "mainboard.toml").write_text('[workspace]\nname = "w"\n' + table, encoding="utf-8")
    return tmp_path


class Device:
    def __init__(self, label: str, holders: tuple[int, ...] = (), gpu_pct: int = 3) -> None:
        self.label = label
        self.holders = holders
        self.utilization = SimpleNamespace(gpu_pct=gpu_pct, memory_pct=1)

    def snapshot(self):
        return SimpleNamespace(processes=[SimpleNamespace(pid=pid) for pid in self.holders])


def test_the_manifest_declares_a_standard_per_card_and_defaults_the_rest(tmp_path: Path) -> None:
    root = workspace(
        tmp_path,
        '[admission."NVIDIA GeForce RTX 5080"]\nutilization_pct = 40\n'
        '[admission."NVIDIA GB10"]\nholders = "record"\n',
    )
    manifest = load(root / "mainboard.toml")
    assert manifest.admission["NVIDIA GeForce RTX 5080"].utilization_pct == 40
    assert manifest.admission["NVIDIA GB10"].holders == "record"
    assert policy("NVIDIA GeForce RTX 4090", root) == policy("anything", root)
    assert policy("NVIDIA GeForce RTX 4090", root).utilization_pct == 10
    assert policy("NVIDIA GeForce RTX 4090", root).holders == "refuse"


def test_a_shared_card_is_recorded_and_an_exclusive_one_refused(tmp_path: Path) -> None:
    root = workspace(tmp_path, '[admission."NVIDIA GB10"]\nholders = "record"\n')
    asked: list[int] = []

    def idle(*, timeout: float, util_threshold: int) -> bool:
        asked.append(util_threshold)
        return True

    shared = admit(Device("NVIDIA GB10", holders=(2786, 3714402)), root=root, idle=idle)
    assert shared.holders == (2786, 3714402)
    assert shared.threshold_pct == 10 and shared.utilization_pct == 3
    with pytest.raises(RuntimeError, match="other compute processes"):
        admit(Device("NVIDIA GeForce RTX 4090", holders=(1,)), root=root, idle=idle)
    assert asked == [10, 10]


def test_a_busy_card_is_refused_with_its_standard_named(tmp_path: Path) -> None:
    root = workspace(tmp_path, '[admission."NVIDIA GeForce RTX 5080"]\nutilization_pct = 40\n')
    with pytest.raises(RuntimeError, match="above 40% utilization"):
        admit(Device("NVIDIA GeForce RTX 5080"), root=root, idle=lambda **_: False, timeout=1)
