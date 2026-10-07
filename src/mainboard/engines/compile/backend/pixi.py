import json
import os
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plumbum import local
from plumbum.commands.base import BoundEnvCommand

from ....core import MissionError, Project
from ....core.host import current_platform
from .engine import PixiEngine
from .process import Process
from .repair import EnvironmentAudit
from .tool import Tool

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from plumbum.commands.base import BaseCommand

    from .result import CommandResult

# What pixi writes into a prefix's `conda-meta/` once an installation has finished.
_FINGERPRINT = ".pixi-environment-fingerprint"

# The env var vouching each virtual-package floor, so a machine that cannot present the package
# (a login node with no GPU driver) still installs the frozen lock its jobs run under.
_FLOOR_OVERRIDES = {
    "archspec": "CONDA_OVERRIDE_ARCHSPEC",
    "cuda": "CONDA_OVERRIDE_CUDA",
    "glibc": "CONDA_OVERRIDE_GLIBC",
    "linux": "CONDA_OVERRIDE_LINUX",
    "macos": "CONDA_OVERRIDE_OSX",
    "osx": "CONDA_OVERRIDE_OSX",
}


class Pixi(Tool):
    """The one seam to the pixi binary, pinned to the `pixi.toml` it owns in a workspace env dir.

    Every provisioning or query command goes through here, so the lock rules and drift diagnosis
    are stated once. `PixiEngine` finds the executable and is held rather than inherited.
    """

    name = "pixi"
    filename = "pixi.toml"

    def __init__(self, out: Path) -> None:
        self.engine = PixiEngine()
        self.manifest = out / self.filename

    def version(self) -> str:
        return self.engine.version()

    @property
    def command(self) -> BaseCommand:
        """The engine's pixi with the workspace's `overrides` bound over its own environment.

        Read per invocation, since the compiler may have just rewritten the floors. Bound
        outright rather than through `with_env`, which returns a bare command when empty, so
        callers see one shape.
        """
        engine = self.engine.command
        environment = dict(engine.env or {}) | self.overrides
        executable = Path(engine.formulate()[0])
        return BoundEnvCommand(local[str(executable)], env=environment)

    @property
    def executable(self) -> Path:
        """The resolved binary alone, for a caller replacing this process rather than spawning."""
        return Path(self.engine.command.formulate()[0])

    @property
    def lock(self) -> Path:
        return self.manifest.with_suffix(".lock")

    @property
    def overrides(self) -> dict[str, str]:
        return Pixi._floor_overrides(self.manifest)

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Prepend the environment's `bin`, once installed, to PATH for the block."""
        binaries = self.env_prefix(env) / "bin"
        leading = [str(binaries)] if binaries.is_dir() else []
        with local.env(PATH=os.pathsep.join([*leading, str(local.env["PATH"])])):
            yield

    def env_prefix(self, env: str) -> Path:
        return self.manifest.parent / ".pixi" / "envs" / env

    def environment_result(self, verb: str, *args: str) -> CommandResult:
        """Run an environment verb over the lock as it stands, retaining its streamed output.

        `--locked` unless an editable source is declared, whose metadata pixi re-reads and so
        would call stale on every edit; such a lock is installed `--frozen`, the caller having
        already asked whether it answers the manifest.
        """
        if not self.lock.exists():
            project = Project()
            raise MissionError(
                f"pixi.lock is missing. Run `{project.name} lock` on a "
                "solve-capable machine to create and verify the generated manifest/lock pair, "
                f"and commit the {project.locks[0]} it writes."
            )
        editable = self._has_editable_paths()
        return self.within_cwd(Process.stream, verb, *args, locked=not editable, frozen=editable)

    def solve(self) -> None:
        """Solve the lock for every declared platform, installing nothing, even ones not this."""
        result = self.within_cwd(Process.stream, "lock")
        if result.returncode:
            raise MissionError("`pixi lock` failed (see its output above)")

    def runs_here(self) -> bool:
        """Whether the compiled manifest declares this platform, or none, or is not compiled."""
        try:
            parsed = tomllib.loads(self.manifest.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return True
        declared = parsed.get("workspace", {}).get("platforms", [])
        names = {
            entry if isinstance(entry, str) else str(entry.get("platform", ""))
            for entry in declared
        }
        return not names or current_platform() in names

    def install(self, env: str) -> None:
        """Install `env` from its lock, then repair whatever it holds that is incomplete."""
        result = self.environment_result("install", "-e", env)
        self._raise_on_lock_drift(result)
        if result.returncode:
            raise MissionError("`pixi install` failed (see its output above)")
        self.repair(env)

    def sync(self, env: str) -> None:
        """Bring the installed prefix in line with the lock, frozen, raising what pixi said.

        The update `pixi run` does as a side effect, taken deliberately: jobs of one wave sharing
        a prefix raced it and the loser saw it mid-write (`Failed to update PyPI packages ... No
        such file or directory`, or a vanished editable; miyabi-g, 2026-09-05).
        """
        result = self.within_cwd(Process.capture, "install", "--frozen", "-e", env)
        if result.returncode:
            raise MissionError(
                f"could not bring environment {env!r} in line with its lock: "
                f"{(result.stderr or result.stdout).strip()[-400:]}"
            )

    def locked(self, env: str) -> dict[str, str]:
        """Every package the lock pins for `env`, name to version, empty when it has none.

        `--frozen` reads the lock as it sits, so readings before and after an edit show exactly
        what the solve moved.
        """
        if not self.lock.exists():
            return {}
        command = self.command["list", "--json", "--frozen", "-e", env, *self.scope()]
        result = Process.capture(command)
        if not result.succeeded:
            return {}
        packages = cast("list[dict[str, str]]", json.loads(result.stdout))
        return {str(package["name"]): str(package["version"]) for package in packages}

    def ready(self, env: str) -> bool:
        """Whether pixi finished installing `env`: its fingerprint, not a mere prefix, exists."""
        return (self.env_prefix(env) / "conda-meta" / _FINGERPRINT).is_file()

    def run(self, command: Sequence[str], env: str = "default") -> int:
        """Run a task or command argv through Pixi, each token a distinct argument, no shell.

        command: a task name and arguments, or an ad-hoc argv.
        """
        return self.within_cwd(Process.passthrough, "run", "--frozen", "-e", env, *command)

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """`run`, capturing output under `timeout` seconds."""
        return self.within_cwd(
            lambda argv: Process.capture(argv, timeout=timeout),
            "run",
            "--frozen",
            "-e",
            env,
            *command,
        )

    def repair(self, env: str) -> None:
        """Reinstall whatever `env` holds that is missing files or can no longer import.

        Only a finished install is audited, since a half-written prefix reads damaged everywhere.
        A package still missing files after the reinstall raises.
        """
        if not self.ready(env):
            return
        audit = EnvironmentAudit(self.env_prefix(env))
        packages = audit.suspect()
        if not packages:
            return
        sys.stderr.write(
            f"{Project().name}: reinstalling {', '.join(packages)} in {env!r}: files they "
            "installed are missing, or an editable's build is behind its sources\n"
        )
        if self.environment_result("reinstall", "-e", env, *packages).returncode:
            raise MissionError("`pixi reinstall` failed while repairing the environment")
        if remaining := audit.damaged():
            raise MissionError(f"{', '.join(remaining)} stayed incomplete after `pixi reinstall`")

    def scope(self) -> tuple[str, ...]:
        return ("--manifest-path", str(self.manifest))

    @staticmethod
    def _floor_overrides(manifest: Path) -> dict[str, str]:
        """The `CONDA_OVERRIDE_*` values for the floors in `manifest`'s platform descriptors.

        An override the process already exports is left to stand, so the caller has the last word.
        """
        try:
            parsed = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        entries = parsed.get("workspace", {}).get("platforms", [])
        return {
            _FLOOR_OVERRIDES[key]: str(value)
            for entry in entries
            if isinstance(entry, dict)
            for key, value in entry.items()
            if key in _FLOOR_OVERRIDES and _FLOOR_OVERRIDES[key] not in os.environ
        }

    def shell_hook(self, env: str = "default", *, shell: str = "bash") -> str:
        """Pixi's full activation of `env` as a sourceable `shell` snippet, for `activate.sh`."""
        # Frozen: unfrozen, a lock pixi reads as stale is re-solved for every platform, which is
        # how a host came to build another platform's sdist (2026-09-11).
        command = self.command["shell-hook", "--frozen", "-s", shell, "-e", env, *self.scope()]
        return Process.output(command, "pixi shell-hook")

    def update(self, env: str, names: Sequence[str] = ()) -> None:
        """Move `env`'s lock to the newest releases the manifest allows, `names` alone when given,
        installing nothing: the caller installs once the lock is committed."""
        if self.within_cwd(Process.stream, "update", "--no-install", "-e", env, *names).returncode:
            raise MissionError("`pixi update` failed (see its output above)")

    @staticmethod
    def _raise_on_lock_drift(result: CommandResult) -> None:
        """Turn Pixi's pre-task lock rejection into an actionable recovery message."""
        failure = f"{result.stdout}\n{result.stderr}".lower().replace("-", " ")
        if (
            result.returncode
            and "pixi task (" not in failure
            and "lock file" in failure
            and "not up to date" in failure
        ):
            raise MissionError(
                f"the manifest drifted from pixi.lock. Run `{Project().name} lock` on a "
                "solve-capable machine, which is also what a host is then sent."
            )

    def _has_editable_paths(self) -> bool:
        """Whether the generated manifest carries a mutable editable Python source."""
        try:
            manifest = tomllib.loads(self.manifest.read_text(encoding="utf-8"))
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
