import os
import tomllib
from contextlib import contextmanager
from functools import cached_property
from typing import TYPE_CHECKING

from plumbum import local

from ....core import MissionError
from .engine import PixiEngine
from .process import Process
from .repair import EnvironmentAudit
from .tool import Tool

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from plumbum.commands.base import BaseCommand

    from .result import CommandResult

# What pixi writes into a prefix's `conda-meta/` once an installation has finished.
_FINGERPRINT = ".pixi-environment-fingerprint"


class Pixi(Tool):
    """The one seam to the pixi binary, pinned to the `pixi.toml` it owns in a workspace env dir.

    Every command that provisions or queries an environment goes through this class on
    purpose, so the lock rules and the drift diagnosis are stated once. That is why so much of
    the compile pipeline depends on it and why it depends on so little: the argv building comes
    from :class:`Tool`, running a child belongs to :class:`Process`, and what comes back is a
    :class:`CommandResult`.

    Finding the executable (and installing it the first time) is a different job from owning a
    workspace manifest, so :class:`PixiEngine` is held rather than inherited, and only its
    resolved command is used.
    """

    name = "pixi"
    filename = "pixi.toml"

    def __init__(self, out: Path) -> None:
        self.engine = PixiEngine()
        self.manifest = out / self.filename

    @cached_property
    def command(self) -> BaseCommand:
        """The pixi executable, resolved (and bootstrapped when absent) by the engine."""
        return self.engine.command

    @property
    def lock(self) -> Path:
        """The lock file paired with the compiled Pixi manifest."""
        return self.manifest.with_suffix(".lock")

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Prepend the provisioned env's `bin/` to PATH for the duration of the block.

        The env may not exist yet (a dry call before install), in which case PATH is left
        untouched.
        """
        binary = self.env_prefix(env) / "bin"
        path = local.env["PATH"]
        with local.env(PATH=f"{binary}{os.pathsep}{path}" if binary.is_dir() else path):
            yield

    def env_prefix(self, env: str) -> Path:
        """The provisioned pixi environment prefix for ``env``."""
        return self.manifest.parent / ".pixi" / "envs" / env

    def environment_result(self, verb: str, *args: str, resolve: bool = False) -> CommandResult:
        """Run an environment verb and retain its streamed native output."""
        if not resolve and not self.lock.exists():
            raise MissionError(
                "pixi.lock is missing. Run `Provisioner.provision(resolve=True)` on a "
                "solve-capable machine to create and verify the generated manifest/lock pair."
            )
        return self.within_cwd(
            Process.stream,
            verb,
            *args,
            locked=not resolve and not self._has_editable_paths(),
            frozen=not resolve and self._has_editable_paths(),
        )

    def _has_editable_paths(self) -> bool:
        """Whether the generated manifest carries a mutable editable Python source."""
        try:
            manifest = tomllib.loads(self.manifest.read_text())
        except FileNotFoundError:
            return False

        pending = [manifest]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if isinstance(value.get("path"), str) and value.get("editable") is True:
                    return True
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        return False

    def install(self, env: str, *, resolve: bool = False) -> None:
        """Install ``env`` locked by default and verify every explicitly resolved lock."""
        locked = not resolve
        result = self.environment_result("install", "-e", env, resolve=resolve)
        self._raise_on_lock_drift(result, locked=locked)
        if result.returncode:
            raise MissionError("`pixi install` failed (see its output above)")
        # Known wart carried over from chefe: a resolve re-installs once more from the
        # now-verified lock, so `install(env, resolve=True)` runs `pixi install` twice. The
        # repair pass rides that second, locked call, so an environment is audited once and
        # always against a lock pixi has already verified.
        if resolve:
            self.install(env)
        else:
            self.repair(env)

    def ready(self, env: str) -> bool:
        """Whether pixi ever finished installing ``env``.

        pixi stamps the environment fingerprint only once an installation completes, so an
        existing prefix directory is not enough. An interrupted install leaves one behind.
        """
        return (self.env_prefix(env) / "conda-meta" / _FINGERPRINT).is_file()

    def repair(self, env: str) -> None:
        """Reinstall whatever ``env`` still holds that can no longer be trusted to import.

        Only a finished installation is audited, because a prefix an interrupted install
        abandoned half-written reads as damaged everywhere and would turn one broken package
        into a whole-environment reinstall. A reinstall that leaves a wheel still missing every
        import root it declares is a failure rather than a repair, so it raises here instead of
        handing back an environment whose first import is the one that fails.
        """
        if not self.ready(env):
            return
        audit = EnvironmentAudit(self.env_prefix(env))
        packages = audit.suspect()
        if not packages:
            return
        if self.environment_result("reinstall", "-e", env, *packages).returncode:
            raise MissionError("`pixi reinstall` failed while repairing Python packages")
        if remaining := audit.damaged():
            raise MissionError(f"{', '.join(remaining)} stayed incomplete after `pixi reinstall`")

    def scope(self) -> tuple[str, ...]:
        return ("--manifest-path", str(self.manifest))

    def shell_hook(self, env: str = "default", *, shell: str = "bash") -> str:
        """The activation script for ``env`` as a sourceable ``shell`` snippet.

        It carries the env vars, PATH, and `[activation] scripts` pixi sets when entering the
        env, the exact activation :meth:`activated` performs, captured as text so a generated
        `activate.sh` can reproduce the whole pixi env without invoking pixi at job time.
        """
        command = self.command["shell-hook", "-s", shell, "-e", env, *self.scope()]
        return Process.output(command, "pixi shell-hook")

    @staticmethod
    def _raise_on_lock_drift(result: CommandResult, *, locked: bool) -> None:
        """Turn Pixi's pre-task lock rejection into an actionable recovery message."""
        failure = f"{result.stdout}\n{result.stderr}".lower().replace("-", " ")
        task_started = "pixi task (" in failure
        if (
            result.returncode
            and locked
            and not task_started
            and "lock file" in failure
            and "not up to date" in failure
        ):
            raise MissionError(
                "the manifest drifted from pixi.lock. Run `Provisioner.provision(resolve=True)` "
                "on a solve-capable machine and reship `.mainboard/pixi.toml` with "
                "`.mainboard/pixi.lock`."
            )
