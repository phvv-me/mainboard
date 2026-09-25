# How a job enters its environment on the machine it landed on.
#
# The activation is written when the environment is built (POSIX: the `activate.sh` of pixi's
# shell hook and the module stack; Windows: the activation pixi recorded as JSON), and entering
# reads it back into the mapping the command starts with. POSIX asks a shell once, the only thing
# that can run a shell hook. Which activation counts and when a prefix is refused are decided by
# the job's activation record, the same way for both.

import json
import os
import platform
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs this machine's own generated activation, not untrusted input since=2026-09-25
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import mkstemp
from typing import TYPE_CHECKING

from ..engines.compile.backend.pixi import Pixi

if TYPE_CHECKING:
    from collections.abc import Mapping

# What a shell adds to its environment describing the shell rather than the activation.
_SHELL_OWN = frozenset({"_", "PWD", "OLDPWD", "SHLVL"})

# The interpreter line a sourcing shell ends with, writing the environment it activated to the
# file named after it. A file rather than stdout, since activation scripts print.
_DUMP = (
    "import json, os, sys\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as sink:\n"
    "    json.dump(dict(os.environ), sink)\n"
)


class Refusal(Exception):
    """An environment that could not be entered, with the exit status the job ends on.

    message: what the job says about it, empty when the activation already said it.
    """

    def __init__(self, message: str, status: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class Entering(ABC):
    """How this machine turns an environment's generated activation into a mapping."""

    @abstractmethod
    def ready(self, shard: Path, script: Path) -> bool:
        """Whether the activation for the environment described in `shard` was ever written."""

    @abstractmethod
    def activated(
        self, base: Mapping[str, str], *, shard: Path, script: Path, env: str, cwd: str
    ) -> dict[str, str]:
        """`base` after entering `env` described in `shard`, raising `Refusal` when it fails."""

    @abstractmethod
    def executables(self, prefix: Path) -> list[Path]:
        """The directories an installed `prefix` keeps its commands in."""


class Sourcing(Entering):
    """POSIX: `bash` sources the generated `activate.sh` once and reports what it built."""

    def ready(self, shard: Path, script: Path) -> bool:
        return script.is_file()

    def activated(
        self, base: Mapping[str, str], *, shard: Path, script: Path, env: str, cwd: str
    ) -> dict[str, str]:
        descriptor, dump = mkstemp(prefix="mainboard-activation-", suffix=".json")
        os.close(descriptor)
        sink = Path(dump)
        try:
            status = subprocess.call(  # ruff:ignore[subprocess-without-shell-equals-true]  reason=a fixed argv sourcing this machine's own activation since=2026-09-25
                [
                    "bash",
                    "-c",
                    '. "$1" && exec "$2" -c "$3" "$4"',
                    "mainboard-activation",
                    str(script),
                    sys.executable,
                    _DUMP,
                    dump,
                ],
                cwd=cwd,
                env=dict(base),
            )
            if status:
                raise Refusal("", status)
            entered: dict[str, str] = json.loads(sink.read_text(encoding="utf-8"))
        finally:
            sink.unlink(missing_ok=True)
        return {name: value for name, value in entered.items() if name not in _SHELL_OWN}

    def executables(self, prefix: Path) -> list[Path]:
        return [prefix / "bin"]


class Recorded(Entering):
    """Windows: nothing can source a shell hook, so pixi's recorded activation is applied."""

    def ready(self, shard: Path, script: Path) -> bool:
        return Pixi(shard).windows_activation_cache.is_file()

    def activated(
        self, base: Mapping[str, str], *, shard: Path, script: Path, env: str, cwd: str
    ) -> dict[str, str]:
        return Pixi(shard).recorded_environment(env, base)

    def executables(self, prefix: Path) -> list[Path]:
        return [prefix, prefix / "Scripts", prefix / "Library" / "bin"]


def entering() -> Entering:
    """How this machine enters an environment: from its record on Windows, a shell elsewhere."""
    return Recorded() if platform.system() == "Windows" else Sourcing()
