"""A fresh cell that runs past its timeout is stopped with every process it started, so no
worker it spawned keeps the card after the cell is reported killed."""

import sys
from pathlib import Path

import psutil
import pytest

from mainboard.jobs.call import Fresh
from mainboard.proc import Exit


def _is_gone(pid: int) -> bool:
    """Whether `pid` has ended, its exit not yet collected counting as ended."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_a_cell_past_its_timeout_takes_its_workers_down_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = tmp_path / "cell"
    cell.write_text('#!/bin/sh\nsleep 60 &\necho $! > "$WORKER"\nwait\n', encoding="utf-8")
    cell.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(cell))
    monkeypatch.setenv("WORKER", str(tmp_path / "worker"))

    code = Fresh(["a"], args=[], timeout=1.0).cells("test_x.py::test_y", dispatched=False)

    worker = int((tmp_path / "worker").read_text(encoding="utf-8"))
    assert code == Exit.TIMED_OUT
    assert _is_gone(worker)
