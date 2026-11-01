from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenOpenModel
from plumbum import local
from plumbum.commands.base import BoundEnvCommand
from pydantic import TypeAdapter

from .....core import Disk, MissionError, Project
from .....log import logger
from ..process import Process
from ..tool import Tool
from .compiled import CompiledManifest
from .engine import PixiEngine
from .prefix import PixiPrefix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plumbum.commands.base import BaseCommand

    from ..result import CommandResult


class _Package(FrozenOpenModel):
    """One row of `pixi list`: an editable workspace member is listed with no version."""

    name: str
    version: str | None


_PACKAGES = TypeAdapter(list[_Package])


class Pixi(Tool):
    """The one seam to the pixi binary, pinned to the `pixi.toml` it owns in a workspace env dir.

    Every provisioning or query command goes through here, so the lock rules and drift diagnosis
    are stated once. `PixiEngine` finds the executable and is held rather than inherited.
    """

    name = "pixi"
    filename = "pixi.toml"

    @property
    def command(self) -> BaseCommand:
        """The engine's pixi with the workspace's `overrides` bound over its own environment.

        Read per invocation, since the compiler may have just rewritten the floors. Bound
        outright rather than through `with_env`, which returns a bare command when empty, so
        callers see one shape.
        """
        engine = self.engine.command
        environment = dict(engine.env or {}) | self.declared.overrides
        executable = Path(engine.formulate()[0])
        return BoundEnvCommand(local[str(executable)], env=environment)

    @property
    def declared(self) -> CompiledManifest:
        """What the compiled manifest declares, read afresh each time it is asked."""
        return CompiledManifest(path=self.manifest)

    @property
    def executable(self) -> Path:
        """The resolved binary alone, for a caller replacing this process rather than spawning."""
        return Path(self.engine.command.formulate()[0])

    @property
    def lock(self) -> Path:
        return self.manifest.with_suffix(".lock")

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

    def __init__(self, out: Path) -> None:
        self.engine = PixiEngine()
        self.manifest = out / self.filename

    def _environment_result(self, verb: str, *args: str) -> CommandResult:
        """Run an environment verb over the lock as it stands, retaining its streamed output.

        `--locked` unless an editable source is declared, whose metadata pixi re-reads and so
        would call stale on every edit; such a lock is installed `--frozen`, the caller having
        already asked whether it answers the manifest.
        """
        if not self.lock.exists():
            project = Project()
            raise MissionError(
                f"pixi.lock is missing. Run `{project.name} dep lock` on a "
                "solve-capable machine to create and verify the generated manifest/lock pair, "
                f"and commit the {project.locks[0]} it writes."
            )
        editable = self.declared.editable
        return self.within_cwd(Process.stream, verb, *args, locked=not editable, frozen=editable)

    def _repair(self, env: str) -> None:
        """Reinstall whatever `env` holds that is missing files or can no longer import.

        Only a finished install is audited, since a half-written prefix reads damaged everywhere.
        A package still missing files after the reinstall raises.
        """
        prefix = self.prefix(env)
        if not prefix.ready:
            return
        audit = prefix.audit
        packages = audit.suspect()
        if not packages:
            return
        logger.warning(
            f"{Project().name}: reinstalling {', '.join(packages)} in {env!r}: files they "
            "installed are missing, or an editable's build is behind its sources"
        )
        if self._environment_result("reinstall", "-e", env, *packages).returncode:
            raise MissionError("`pixi reinstall` failed while repairing the environment")
        if remaining := audit.damaged():
            raise MissionError(f"{', '.join(remaining)} stayed incomplete after `pixi reinstall`")

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

    def install(self, env: str) -> None:
        """Install `env` from its lock, then repair whatever it holds that is incomplete.

        An environment runs to gigabytes, so a disk already below its floor refuses first
        (`Disk.require`).
        """
        Disk(path=self.prefix(env).path).require(doing=f"installing {env}")
        result = self._environment_result("install", "-e", env)
        self._raise_on_lock_drift(result)
        if result.returncode:
            raise MissionError("`pixi install` failed (see its output above)")
        self._repair(env)

    def locked(self, env: str) -> dict[str, str]:
        """Every package the lock pins for `env`, name to version, empty when it has none.

        `--frozen` reads the lock as it sits, so readings before and after an edit show exactly
        what the solve moved. An editable member of the workspace has the empty version.
        """
        if not self.lock.exists():
            return {}
        command = self.command["list", "--json", "--frozen", "-e", env, *self.scope()]
        result = Process.capture(command)
        if not result.succeeded:
            return {}
        return {
            package.name: package.version or ""
            for package in _PACKAGES.validate_json(result.stdout)
        }

    def prefix(self, env: str) -> PixiPrefix:
        """Where pixi installs `env`, whether or not it has yet."""
        return PixiPrefix(path=self.manifest.parent / ".pixi" / "envs" / env)

    def run(self, command: Sequence[str], env: str = "default") -> int:
        """Run a task or command argv through Pixi, each token a distinct argument, no shell.

        command: a task name and arguments, or an ad-hoc argv.
        """
        return self.within_cwd(Process.passthrough, "run", "--frozen", "-e", env, *command)

    def scope(self) -> tuple[str, ...]:
        return ("--manifest-path", str(self.manifest))

    def shell_hook(self, env: str = "default", *, shell: str = "bash") -> str:
        """Pixi's full activation of `env` as a sourceable `shell` snippet, for `activate.sh`."""
        # Frozen: unfrozen, a lock pixi reads as stale is re-solved for every platform, which is
        # how a host came to build another platform's sdist (2026-09-11).
        command = self.command["shell-hook", "--frozen", "-s", shell, "-e", env, *self.scope()]
        return Process.output(command, "pixi shell-hook")

    def solve(self) -> None:
        """Solve the lock for every declared platform, installing nothing, even ones not this."""
        result = self.within_cwd(Process.stream, "lock")
        if result.returncode:
            raise MissionError("`pixi lock` failed (see its output above)")

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

    def update(self, env: str, names: Sequence[str] = ()) -> None:
        """Move `env`'s lock to the newest releases the manifest allows, `names` alone when given.

        Install nothing when the lock carries no editable source; with one, pixi must build the
        environment to read that source's metadata, so `--no-install` fails the solve
        ("installation of conda environment is required to solve PyPI source dependencies") and is
        dropped, exactly as `_environment_result` installs such a lock `--frozen`.
        """
        flags = () if self.declared.editable else ("--no-install",)
        if self.within_cwd(Process.stream, "update", *flags, "-e", env, *names).returncode:
            raise MissionError("`pixi update` failed (see its output above)")
