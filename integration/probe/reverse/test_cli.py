import json
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.probe.reverse import CuobjdumpProvider, Evidence, Request
from mainboard.probe.reverse.evidence import DeviceResult
from mainboard.probe.reverse.service import DeviceProbe
from mainboard.verbs.constants import TREE

from .builders import artifact
from .fakes import FixtureRunner


def test_probe_device_json_and_tsv_keep_uncertainty(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    found = CuobjdumpProvider(FixtureRunner()).analyze(
        Request(artifact=artifact(), operation="sass")
    )

    def run(
        self: DeviceProbe, library: Path, operation: str, **options: str
    ) -> Evidence[DeviceResult]:
        assert library == Path("lib.so") and operation == "sass"
        return found

    monkeypatch.setattr(DeviceProbe, "run", run)
    monkeypatch.chdir(workspace)
    app = TREE.cli()
    app(["data", "probe", "lib.so", "sass", "--json"], result_action="return_value")
    assert json.loads(capsys.readouterr().out) == found.model_dump(mode="json")
    app(["data", "probe", "lib.so", "sass"], result_action="return_value")
    output = capsys.readouterr().out
    assert "\t" in output and "BRX" in output and "unknown" in output and "partial" in output


def test_probe_device_remote_refuses_before_reading_or_running() -> None:
    with pytest.raises(MissionError, match="locally"):
        TREE.cli()(["data", "probe", "absent.so", "sass", "--on", "gold"])
