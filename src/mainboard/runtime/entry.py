# How a job enters its environment on the machine it landed on.
#
# The activation itself is written when the environment is built: on POSIX the `activate.sh` that
# pixi's shell hook and the module stack make up, on Windows the activation pixi recorded as JSON.
# Entering is reading that back into a plain mapping the command is then started with. POSIX has
# to ask a shell once, since that is the only thing that can run a shell hook, and Windows reads
# the record directly. Which activation counts, when a prefix is refused and what a machine with
# nothing to enter says are decided by the job's activation record, the same way for both.

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

# What a shell adds to its own environment that describes the shell rather than the activation,
# and would otherwise be handed to a command started somewhere else.
_SHELL_OWN = frozenset({"_", "PWD", "OLDPWD", "SHLVL"})

# The interpreter line a sourcing shell ends with, writing the environment it activated to the
# file named after it. A file rather than stdout, since activation scripts print.
_DUMP = (
    "import json, os, sys\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as sink:\n"
    "    json.dump(dict(os.environ), sink)\n"
)


class Refusal(Exception):
    """An environment that could not be entered, with the status the job ends on.

    message: what the job says about it, empty when the activation already said it.
    status: the exit status the job reports.
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
    """Windows: the activation pixi recorded when the environment was built, applied directly.

    Nothing on Windows can source a shell hook, so the record is the activation.
    """

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
