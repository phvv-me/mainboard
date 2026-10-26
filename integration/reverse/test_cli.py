import json
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.cli import build
from mainboard.probe.reverse import CuobjdumpProvider, Evidence, Request
from mainboard.probe.reverse.models import DeviceResult
from mainboard.probe.reverse.service import DeviceProbe

from .fixtures import FixtureRunner, artifact


def test_probe_device_json_and_tsv_keep_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    found = CuobjdumpProvider(FixtureRunner()).analyze(
        Request(artifact=artifact(), operation="sass")
    )

    def run(
        self: DeviceProbe,
        library: Path,
        operation: str,
        *,
        arch: str = "",
        function: str = "",
        against: str = "",
    ) -> Evidence[DeviceResult]:
        assert library == Path("lib.so") and operation == "sass"
        return found

    monkeypatch.setattr(DeviceProbe, "run", run)
    app = build(tmp_path)
    app(["probe", "device", "lib.so", "sass", "--json"], result_action="return_value")
    assert json.loads(capsys.readouterr().out) == found.model_dump(mode="json")
    app(["probe", "device", "lib.so", "sass"], result_action="return_value")
    output = capsys.readouterr().out
    assert "\t" in output and "BRX" in output and "unknown" in output and "partial" in output


def test_probe_device_remote_refuses_before_reading_or_running(tmp_path: Path) -> None:
    with pytest.raises(MissionError, match="locally"):
        build(tmp_path)(["probe", "device", "absent.so", "sass", "--on", "gold"])
