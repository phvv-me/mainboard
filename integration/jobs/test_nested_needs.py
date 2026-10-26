"""Needs nested in one another: the closure keeps the outermost, and a host handed a need beneath
one it already linked to the mirror refuses rather than place it over the mirror's own file."""

import os
from pathlib import Path

import pytest

from mainboard.dispatch.agent.program import Refusal, Sealed
from mainboard.jobs import Closure, Target


def test_the_closure_keeps_each_need_once_and_only_the_outermost(tmp_path: Path) -> None:
    (tmp_path / "mainboard.toml").write_text('[workspace]\nname = "w"\n', encoding="utf-8")
    (tmp_path / "job.py").write_text("def main() -> None: ...\n", encoding="utf-8")
    for need in ("data/x", "other/y"):
        (tmp_path / need).parent.mkdir(parents=True)
        (tmp_path / need).write_bytes(b"kept")

    closure = Closure.of(
        Target(file="job.py", name="main"),
        root=tmp_path,
        distributions=(),
        environment=tmp_path / "env",
        needs=("data/x", "data/", "./other", "other/y"),
    )

    assert closure.needs == ("data", "other")


def test_a_need_under_a_linked_need_is_refused_and_the_mirror_keeps_its_file(
    tmp_path: Path,
) -> None:
    mirror, snap = tmp_path / "mirror", tmp_path / "snap"
    (mirror / "lake").mkdir(parents=True)
    (mirror / "lake" / "table.parquet").write_bytes(b"PAR1")
    (mirror / "data").mkdir()
    (mirror / "data" / "table.parquet").symlink_to(mirror / "lake" / "table.parquet")
    snap.mkdir()
    image = Sealed(
        {
            "listing": "",
            "digest": "",
            "needs": ["data", "data/table.parquet"],
            "pins": [],
            "staging": "",
            "live": [],
        }
    )

    with pytest.raises(Refusal, match="already linked to the mirror"):
        image.link(str(mirror), snap=str(snap))

    assert os.readlink(mirror / "data" / "table.parquet") == str(mirror / "lake" / "table.parquet")
