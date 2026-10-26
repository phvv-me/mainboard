"""A test lane submits every job as a single submit would: its target's own tokens, every
execution and data option, and a wait over every handle whatever one of them ended on."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard import Board
from mainboard.cli import build
from mainboard.dispatch import Request
from mainboard.jobs.testing.lanes import Cell


class _Collector:
    """`localhost[tool][argv]()`, the lane's collection, which prints nothing here."""

    def __call__(self) -> str:
        return ""

    def __getitem__(self, _: object) -> _Collector:
        return self


def test_a_lane_keeps_every_option_and_the_targets_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mainboard.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    cells = tuple(
        Cell(nodeid=f"test_a.py::test_b[{model}]", key=model, params={"model": model})
        for model in ("small", "large")
    )
    monkeypatch.setattr("mainboard.verbs.work.submit.localhost", _Collector())
    monkeypatch.setattr("mainboard.verbs.work.submit.lanes_module.parsed", lambda text: cells)
    submitted: list[Request] = []

    def dispatch(board: Board, asked: Request, **_: object) -> SimpleNamespace:
        submitted.append(asked)
        return SimpleNamespace(handle=SimpleNamespace(id=str(len(submitted))))

    monkeypatch.setattr(Board, "dispatch", dispatch)

    with pytest.raises(SystemExit) as stopped:
        build(tmp_path)(
            [
                "job", "submit", "--on", "gold", "--split", "model", "--yes",
                "--fetch", "results", "--needs", "data/table.parquet", "--env", "cutok",
                "--container", "none", "--", "test_a.py::test_b", "-k", "not slow",
            ]
        )  # fmt: skip

    assert stopped.value.code == 0
    assert [asked.command.split(" -- ")[0] for asked in submitted] == [
        "test_a.py::test_b -k 'not slow'"
    ] * 2
    for asked in submitted:
        assert (asked.fetch, asked.needs, asked.env, asked.container) == (
            "results",
            ("data/table.parquet",),
            "cutok",
            "none",
        )
