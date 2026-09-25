import re
from types import SimpleNamespace

import pytest

from mainboard.experiments import device


def probed(monkeypatch: pytest.MonkeyPatch, *gpus: SimpleNamespace) -> None:
    monkeypatch.setattr(device, "Machine", lambda: SimpleNamespace(gpus=list(gpus)))
    device.device_tag.cache_clear()


def test_the_tag_shortens_a_known_product_and_carries_its_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed(
        monkeypatch,
        SimpleNamespace(label="NVIDIA A100", cuda_architecture=SimpleNamespace(major=8, minor=0)),
    )
    assert device.device_tag() == "A100_CC8.0"
    assert device.device_name() == "NVIDIA A100"


def test_a_device_without_a_capability_is_tagged_by_its_shortened_label_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed(monkeypatch, SimpleNamespace(label="NVIDIA Fancy Card 9000", cuda_architecture=None))
    assert device.device_tag() == "Fancy_Card_9000"


def test_a_machine_without_that_accelerator_is_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    probed(monkeypatch)
    assert device.device_tag() == "CPU"
    assert device.device_name() == "cpu"


@pytest.mark.parametrize("index", [0, 7])
def test_the_real_probe_tags_a_slug_or_the_cpu(index: int) -> None:
    device.device_tag.cache_clear()
    tag = device.device_tag(index)
    assert tag == "CPU" or re.fullmatch(r"[A-Za-z0-9_.]+", tag), tag
