import json
from pathlib import Path

import pytest

from mainboard.engines.compile.backend import CommandResult, Pixi, Process


def test_a_workspace_member_listed_with_no_version_does_not_stop_the_reading_of_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pixi = Pixi(tmp_path)
    pixi.lock.write_text("", encoding="utf-8")
    listed = [{"name": "six", "version": "1.17.0"}, {"name": "member", "version": None}]
    monkeypatch.setattr(
        Process, "capture", lambda command: CommandResult(0, json.dumps(listed), "")
    )

    assert pixi.locked("default") == {"six": "1.17.0", "member": ""}
