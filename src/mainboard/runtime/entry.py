# How a job enters its environment on the machine it landed on.
#
# The activation is written when the environment is built (the `activate.sh` of pixi's shell hook
# and the module stack), and entering reads it back into the mapping the command starts with:
# `bash` sources it once, the only thing that can run a shell hook. Which activation counts and
# when a prefix is refused are decided by the job's activation record.

import json
import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs this machine's own generated activation, not untrusted input since=2026-09-25
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
from typing import TYPE_CHECKING

from .exceptions import Refusal

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

# What a shell adds to its environment describing the shell rather than the activation.
_SHELL_OWN = frozenset({"_", "PWD", "OLDPWD", "SHLVL"})

# The interpreter line a sourcing shell ends with, writing the environment it activated to the
# file named after it. A file rather than stdout, since activation scripts print.
_DUMP = """import json, os, sys
with open(sys.argv[1], 'w', encoding='utf-8') as sink:
    json.dump(dict(os.environ), sink)
"""


@contextmanager
def _dump_file() -> Generator[Path]:
    """A fresh temporary file for the activation to dump its environment into, removed after."""
    descriptor, dump = mkstemp(prefix="mainboard-activation-", suffix=".json")
    os.close(descriptor)
    sink = Path(dump)
    try:
        yield sink
    finally:
        sink.unlink(missing_ok=True)


def activated(base: Mapping[str, str], *, script: Path, cwd: str) -> dict[str, str]:
    """`base` after `bash` sources `script` from `cwd`, raising `Refusal` when it fails."""
    with _dump_file() as sink:
        status = subprocess.call(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=a fixed argv sourcing this machine's own activation since=2026-09-25
            [
                "bash",
                "-c",
                '. "$1" && exec "$2" -c "$3" "$4"',
                "mainboard-activation",
                str(script),
                sys.executable,
                _DUMP,
                str(sink),
            ],
            cwd=cwd,
            env=dict(base),
        )
        if status:
            raise Refusal("", status)
        entered: dict[str, str] = json.loads(sink.read_text(encoding="utf-8"))
    return {name: value for name, value in entered.items() if name not in _SHELL_OWN}
