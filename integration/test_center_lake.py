"""A center migration carries the lake its manifest names: a lake inside the workspace ships as
a consistent catalog snapshot beside its Parquet files, and one outside it is named as left
behind while the destination's own readiness fails until that machine reaches it."""

import sqlite3
from contextlib import closing
from pathlib import Path

from mainboard.board import Board
from mainboard.center.state import Carried, Destination
from mainboard.center.verify import Verification
from mainboard.core import Project
from mainboard.core.section import Verdict
from mainboard.engines.compile import Provisioner
from mainboard.probe.system import System


def _workspace(root: Path, lake: Path | None) -> Board:
    (root / ".git").mkdir(parents=True)
    line = f'lake = "{lake}"\n' if lake else ""
    (root / "mainboard.toml").write_text(f'[workspace]\nname = "w"\n{line}', encoding="utf-8")
    board = Board(root)
    Provisioner(root, board.manifest).recompiled()
    return board


def _catalog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as catalog:
        catalog.execute("CREATE TABLE ducklake_metadata (key TEXT)")
        catalog.commit()


def _carried(board: Board, tmp_path: Path) -> Carried:
    place = Destination(root="/srv/projects", home="/home/someone")
    return Carried(board.root, board.manifest, place, home=tmp_path / "home", user="someone")


def _readiness(board: Board) -> Verification:
    return Verification(board, system=System.model_construct())


def test_a_lake_inside_the_workspace_ships_its_catalog_as_a_snapshot(tmp_path: Path) -> None:
    board = _workspace(tmp_path / "w", None)
    out = Project().out(board.root)
    _catalog(out / "lake.sqlite")
    (out / "lake" / "main").mkdir(parents=True)
    (out / "lake" / "main" / "part.parquet").write_bytes(b"PAR1")

    parcels = {parcel.path: parcel for parcel in _carried(board, tmp_path).workspace()}

    assert parcels[f"{out.name}/lake.sqlite"].data.startswith(b"SQLite format 3")
    assert parcels[f"{out.name}/lake/main/part.parquet"].source is not None
    assert not any(row.section == "left: lake" for row in _carried(board, tmp_path).left())
    assert _readiness(board).lake().verdict is Verdict.PASS


def test_an_external_lake_is_left_named_and_fails_readiness_until_reached(
    tmp_path: Path,
) -> None:
    external = tmp_path / "PORTABLE" / "mainboard-lake"
    board = _workspace(tmp_path / "w", external)

    unreached = _readiness(board).lake()
    _catalog(external / "lake.sqlite")
    carried = _carried(board, tmp_path)

    assert unreached.verdict is Verdict.FAIL and str(external) in unreached.fix
    assert not any("lake" in parcel.path for parcel in carried.workspace())
    assert [row.verdict for row in carried.left() if row.section == "left: lake"] == [Verdict.WARN]
    assert _readiness(board).lake().verdict is Verdict.PASS
