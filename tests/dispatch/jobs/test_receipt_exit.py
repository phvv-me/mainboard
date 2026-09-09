import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mainboard.dispatch.evidence import receipts_in
from mainboard.dispatch.jobs import JobSpec
from mainboard.dispatch.shared import state_dir

from ..support import plan


@pytest.mark.skipif(sys.platform == "win32", reason="Exercises POSIX job termination")
@pytest.mark.parametrize("pbs", [False, True], ids=["bash", "pbs"])
@pytest.mark.parametrize(
    ("outcome", "ending", "expected_status"),
    [
        ("term", 'kill -TERM "$PPID"', 143),
        ("success", "exit 0", 0),
        ("failure", "exit 7", 7),
        ("activation", "exit 0", 23),
    ],
)
def test_rendered_job_finalizes_receipts_once_without_changing_exit_status(
    tmp_path: Path, pbs: bool, outcome: str, ending: str, expected_status: int
) -> None:
    """A completed receipt survives TERM while failed activation leaves no invented receipt."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Rendered remote jobs require Bash")
    generated = tmp_path / ".mainboard"
    generated.mkdir()
    activation = "exit 23\n" if outcome == "activation" else "true\n"
    (generated / "activate.sh").write_text(activation, encoding="utf-8")
    receipt = json.dumps({"trial_receipt": {"run_id": "completed", "pad": "x" * 900}})
    command = f"printf '%s\\n' {shlex.quote(receipt)} >> \"$MAINBOARD_RECEIPTS\"\n{ending}"
    script = JobSpec(
        cmd=command,
        plan=plan(),
        root=tmp_path.as_posix(),
        walltime="00:00:10" if pbs else "",
    ).render(pbs=pbs)
    done = subprocess.run(
        [bash, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PBS_JOBID": "42.test", "PBS_O_WORKDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert done.returncode == expected_status, done.stderr
    log = tmp_path / state_dir() / "logs" / "42.log"
    output = log.read_text(encoding="utf-8") if pbs else done.stdout
    expected_receipts = () if outcome == "activation" else (receipt,)
    assert receipts_in(output) == expected_receipts
    assert output.count("mainboard-receipts-begin") == len(expected_receipts)
    assert "unbound variable" not in output + done.stderr
    if pbs:
        assert log.with_suffix(".exit").read_text(encoding="utf-8") == f"exit={expected_status}\n"
