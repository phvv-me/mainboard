from __future__ import annotations

import subprocess
from functools import cache


@cache
def cached_run(*command: str) -> str:
    """Run a small identity command once and return stdout."""
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout
