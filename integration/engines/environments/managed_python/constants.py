import re
import shlex
import shutil

import pytest

from mainboard import ExecutionPlan
from mainboard.manifest import HostProfile

HOST = "fakehost"
ROOT = "/srv/jobs"
PYTHON = ">=3.14.0,<3.15"
PYTHON_INSTALL = f"uv python install --upgrade --no-bin {shlex.quote(PYTHON)}"
SHELLS = [
    "sh",
    "bash",
    pytest.param("zsh", marks=pytest.mark.skipif(not shutil.which("zsh"), reason="no zsh")),
]
SYSTEM = re.compile(r"\bpython3\b|-m pip\b|\bpip install\b|--break-system-packages")
PLAN = ExecutionPlan(
    host=HOST, profile=HostProfile(kind="ssh", root=ROOT, platform="linux-64"), env="default"
)
HOMED = PLAN.model_copy(update={"profile": PLAN.profile.model_copy(update={"root": "~/.mb-jobs"})})
