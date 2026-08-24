import pytest

from mainboard.profile import arch_config

from .conftest import FakeGPU

_TABLE = {"sm_89": ("ada", 32), "sm_90": ("hopper", 64)}
_DEFAULT = ("cpu", 8)


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (FakeGPU(arch_key="sm_90"), ("hopper", 64)),
        (FakeGPU(arch_key="sm_121"), _DEFAULT),
        (None, _DEFAULT),
    ],
    ids=["known_arch", "unknown_arch", "cpu_only_host"],
)
def test_arch_config_selects_this_gpus_entry_or_the_usable_default(
    gpu: FakeGPU | None, expected: tuple[str, int]
) -> None:
    """A known arch picks its own config, and anything else still yields a usable one."""
    assert arch_config(_TABLE, default=_DEFAULT, gpu=gpu) == expected
