from mainboard.profile import arch_config

from .conftest import FakeGPU


def test_arch_config_selects_the_entry_for_this_gpu() -> None:
    """`arch_config` returns the table value keyed on the given GPU's arch."""
    gpu = FakeGPU(arch_key="sm_90")
    table = {"sm_89": ("ada", 32), "sm_90": ("hopper", 64)}
    assert arch_config(table, default=("cpu", 8), gpu=gpu) == ("hopper", 64)


def test_arch_config_falls_back_when_arch_absent() -> None:
    """An arch not in the table yields the default rather than raising."""
    gpu = FakeGPU(arch_key="sm_90")
    assert arch_config({"sm_121": 16}, default=99, gpu=gpu) == 99


def test_arch_config_defaults_when_no_gpu() -> None:
    """On a CPU-only host (no GPU) `arch_config` returns the default."""
    assert arch_config({"sm_90": 1}, default=-1, gpu=None) == -1
