"""The HPC-AI backend against its API's recorded answers, without a key or a bill."""

import pytest

from mainboard import ExecutionPlan, MissionError
from mainboard.dispatch import Resources
from mainboard.dispatch.backends import HpcAiBackend
from mainboard.manifest import HostProfile

from ..fakes import Recorded
from .builders import hpcai_types


def test_hpcai_picks_the_cheapest_type_in_stock_for_the_architecture(monkeypatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "test-hpcai")
    catalog = hpcai_types(
        ("", 0, 0.24, True),
        ("RTX-5090", 8, 5.2, True),
        ("B200-SXM-180GB-SPOT", 8, 11.92, True),
        ("B200-SXM-180GB", 8, 24.4, True),
    )
    backend = HpcAiBackend(transport=Recorded({("POST", "/resource/user/instance/list"): catalog}))
    plan = ExecutionPlan(
        host="hpc-ai", profile=HostProfile(vars={"image-id": "img"}), env="default"
    )
    spot_dc = backend.chosen(plan, Resources(gpus=8, arch="sm_100", spot=True))
    assert (spot_dc.instance_type_id, spot_dc.spot) == ("type-B200-SXM-180GB-SPOT", True)
    assert backend.chosen(plan, Resources(gpus=1, arch="sm_120")).instance_type_id == (
        "type-RTX-5090"
    )
    assert backend.chosen(plan, Resources()).instance_type_id == "type-"
    with pytest.raises(MissionError, match="no 8x H100"):
        backend.chosen(plan, Resources(gpus=8, gpu_name="H100"))


def test_hpcai_gives_up_on_an_image_that_will_not_pull(monkeypatch) -> None:
    monkeypatch.setenv("HPCAI_API_KEY", "test-hpcai")
    stuck = {
        "instanceMetadata": {"instanceId": "nb-1"},
        "instanceRuntimeInfo": {
            "status": "Starting",
            "phase": "DownloadImage",
            "diagnosisReason": "BackOff",
            "diagnosisMessage": "Back-off restarting failed container download-image",
        },
        "instanceSpecInfo": {"nodePorts": [{"port": 22, "nodePort": 0}]},
    }
    listing = {"instances": [stuck], "pager": {"totalEntries": 1}}
    slept: list[float] = []
    backend = HpcAiBackend(
        transport=Recorded({("POST", "/instance/list"): listing}), sleeper=slept.append
    )
    with pytest.raises(MissionError, match="DownloadImage BackOff.*image-id"):
        backend.endpoint("nb-1")
    assert len(slept) < 20  # minutes of back-off, not the whole quarter hour
