"""`MB_PROFILE=report mb <verb>` profiles any verb with every module of this tool, whatever
optional extras the machine lacks."""

import os
import subprocess
import sys
from pathlib import Path


def test_a_profiled_verb_writes_its_span_report_whatever_extras_are_installed(
    tmp_path: Path,
) -> None:
    report = tmp_path / "profile.txt"

    result = subprocess.run(
        [sys.executable, "-m", "mainboard", "--version"],
        cwd=tmp_path,
        env={**os.environ, "MB_PROFILE": str(report)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert report.read_text(encoding="utf-8").startswith("Spans")
