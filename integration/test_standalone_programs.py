"""The programs mainboard sends to another machine as source run there with nothing of it installed.

Each runs in an isolated interpreter without site-packages, the way the host's bare Python meets
it, so a refactor that hands one of them an import of mainboard fails here rather than on a host.
"""

import io
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from mainboard.center.migration.carrier import Carrier
from mainboard.dispatch.collection import pack

_ISOLATED = (sys.executable, "-I", "-S")


def test_the_collection_script_packs_with_the_standard_library_alone(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "result.txt").write_text("kept", encoding="utf-8")
    script = Path(pack.__file__).read_text(encoding="utf-8")
    script += f"\npack({str(tmp_path)!r}, relative='out')\n"
    ran = subprocess.run(
        [*_ISOLATED, "-"], input=script.encode(), capture_output=True, check=False
    )
    assert ran.returncode == 0, ran.stderr.decode()
    assert ZipFile(io.BytesIO(ran.stdout)).namelist() == ["out/result.txt"]


def test_the_migration_agent_answers_with_the_standard_library_alone(tmp_path: Path) -> None:
    ran = subprocess.run(
        [*_ISOLATED, "-c", "import sys;exec(sys.stdin.buffer.readline())"],
        input=Carrier.preamble("where", {"root": str(tmp_path)}),
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr.decode()
    answer = next(
        line for line in ran.stdout.decode().splitlines() if line.startswith("mainboard-")
    )
    assert json.loads(answer.partition(" ")[2])["root"] == str(tmp_path)
