import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mainboard import ComputePath, Survey
from mainboard.cli import build
from mainboard.compute import Access
from mainboard.holds import Holds
from mainboard.manifest import HostProfile
from mainboard.manifest.held import Held

if TYPE_CHECKING:
    from pathlib import Path

# One held machine as `hold` answers with it and `release` hands it back.
_HELD = Held(
    alias="box",
    provider="vast",
    handle="7",
    gpu="RTX 5090",
    usd_hr=0.6,
    deadline=datetime(2026, 9, 25, 18, tzinfo=UTC),
    profile=HostProfile(kind="ssh", root="/root/projects"),
)


def test_hold_and_release_translate_their_flags_and_print_the_held_machine(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[tuple[str, dict[str, str | int | float]]] = []

    def hold(self: Holds, provider: str, **options: str | int | float) -> Held:
        options.pop("watch")
        asked.append((provider, options))
        return _HELD

    monkeypatch.setattr(Holds, "hold", hold)
    monkeypatch.setattr(Holds, "release", lambda self, alias: _HELD)
    argv = ["hold", "vast", "--for", "3h", "--as", "box", "--gpu-name", "RTX 5090"]
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)([*argv, "--max-usd", "5", "--json"])
    assert asked == [
        (
            "vast",
            {
                "duration": "3h",
                "alias": "box",
                "gpu_name": "RTX 5090",
                "gpus": 0,
                "max_usd": 5.0,
                "env": "",
            },
        )
    ]
    held = json.loads(capsys.readouterr().out)
    assert (held["alias"], held["deadline"], held["root"]) == (
        "box",
        "2026-09-25T18:00:00+00:00",
        "/root/projects",
    )
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)(["release", "box", "--agent", "--fields", "alias,handle"])
    assert capsys.readouterr().out.splitlines()[1:] == ["alias\tbox", "handle\t7"]


def test_compute_releases_what_is_past_due_before_it_lists(
    depot: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Holds, "expire", lambda self: [_HELD])
    rented = ComputePath(name="box", kind="rental", access=Access.RENTED, detail="vast 7")
    monkeypatch.setattr(Survey, "paths", lambda self: [rented])
    with pytest.raises(SystemExit, match="^0$"):
        build(depot)(["compute", "--json"])
    printed = capsys.readouterr()
    assert "released box, vast 7" in printed.err
    assert json.loads(printed.out)[0]["access"] == "rented"
