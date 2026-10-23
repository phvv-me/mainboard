"""A test lane submits every job as a single submit would: its target's own tokens, every
execution and data option, and a wait over every handle whatever one of them ended on."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mainboard.board import Board
from mainboard.cli import build
from mainboard.jobs.lanes import Cell


class _Collector:
    """`localhost[tool][argv]()`, the lane's collection, which prints nothing here."""

    def __getitem__(self, _: object) -> _Collector:
        return self

    def __call__(self) -> str:
        return ""


def test_a_lane_keeps_every_option_and_the_targets_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mainboard.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    cells = tuple(
        Cell(nodeid=f"test_a.py::test_b[{model}]", key=model, params={"model": model})
        for model in ("small", "large")
    )
    monkeypatch.setattr("mainboard.cli.localhost", _Collector())
    monkeypatch.setattr("mainboard.cli.lanes_module.parsed", lambda text: cells)
    submitted: list[tuple[str, dict[str, object]]] = []

    def submit(board: Board, command: str, **options: object) -> SimpleNamespace:
        submitted.append((command, options))
        return SimpleNamespace(handle=SimpleNamespace(id=str(len(submitted))))

    monkeypatch.setattr(Board, "submit", submit)

    with pytest.raises(SystemExit) as stopped:
        build(tmp_path)(
            [
                "job", "submit", "--on", "gold", "--split", "model", "--yes",
                "--fetch", "results", "--needs", "data/table.parquet", "--env", "cutok",
                "--container", "none", "--", "test_a.py::test_b", "-k", "not slow",
            ]
        )  # fmt: skip

    assert stopped.value.code == 0
    assert [command.split(" -- ")[0] for command, _ in submitted] == [
        "test_a.py::test_b -k 'not slow'"
    ] * 2
    for _, options in submitted:
        assert (options["fetch"], options["needs"], options["env"], options["container"]) == (
            "results",
            ("data/table.parquet",),
            "cutok",
            "none",
        )
