import json
import os
import platform
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plumbum import local

from ....core import MissionError, Project
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

# The env var conda tooling reads to vouch each virtual-package floor a platform descriptor can
# carry. A machine that cannot present the package itself still installs the frozen lock its
# jobs will run under, the cluster login node with no GPU driver being the canonical case.
_FLOOR_OVERRIDES = {
    "archspec": "CONDA_OVERRIDE_ARCHSPEC",
    "cuda": "CONDA_OVERRIDE_CUDA",
    "glibc": "CONDA_OVERRIDE_GLIBC",
    "linux": "CONDA_OVERRIDE_LINUX",
    "macos": "CONDA_OVERRIDE_OSX",
    "osx": "CONDA_OVERRIDE_OSX",
    "windows": "CONDA_OVERRIDE_WIN",
}


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

    @property
    def command(self) -> BaseCommand:
        """The pixi executable, resolved by the engine, vouching the workspace's floors.

        Every declared virtual-package floor rides along as its `CONDA_OVERRIDE_*` variable, so
        a frozen install succeeds on a machine that cannot present the package itself, and a
        value the caller already exported always wins. Read per invocation rather than cached,
        since the floors live in the generated manifest the compiler may write moments earlier.
        """
        command = self.engine.command
        if overrides := self.overrides:
            command = command.with_env(**overrides)
        return command

    @property
    def executable(self) -> Path:
        """The resolved pixi binary itself, without the environment `command` binds onto it.

        A caller that replaces this process rather than spawning one needs the path and the
        environment separately, since binding them together is a convenience only a child
        process inherits.
        """
        return Path(self.engine.command.executable)

    @property
    def lock(self) -> Path:
        """The lock file paired with the compiled Pixi manifest."""
        return self.manifest.with_suffix(".lock")

    @property
    def overrides(self) -> dict[str, str]:
        """The `CONDA_OVERRIDE_*` variables this workspace's declared floors vouch for."""
        return Pixi._floor_overrides(self.manifest)

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Prepend the provisioned environment's executable directories for the block.

        Pixi's Windows prefixes expose commands from the root, ``Scripts`` and ``Library/bin``;
        POSIX prefixes use ``bin``. The env may not exist yet, in which case PATH is untouched.
        """
        prefix = self.env_prefix(env)
        candidates = (
            (prefix, prefix / "Scripts", prefix / "Library" / "bin")
            if platform.system() == "Windows"
            else (prefix / "bin",)
        )
        binaries = [str(candidate) for candidate in candidates if candidate.is_dir()]
        path = local.env["PATH"]
        with local.env(PATH=os.pathsep.join([*binaries, str(path)])):
            yield

    def env_prefix(self, env: str) -> Path:
        """The provisioned pixi environment prefix for ``env``."""
        return self.manifest.parent / ".pixi" / "envs" / env

    def environment_result(self, verb: str, *args: str, resolve: bool = False) -> CommandResult:
        """Run an environment verb and retain its streamed native output."""
        if not resolve and not self.lock.exists():
            raise MissionError(
                f"pixi.lock is missing. Run `{Project().name} install --resolve` on a "
                "solve-capable machine to create and verify the generated manifest/lock pair."
            )
        return self.within_cwd(
            Process.stream,
            verb,
            *args,
            locked=not resolve and not self._has_editable_paths(),
            frozen=not resolve and self._has_editable_paths(),
        )

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

    def locked(self, env: str) -> dict[str, str]:
        """Every package the lock pins for ``env``, by name and version, without solving.

        ``--frozen`` reads the lock exactly as it sits instead of checking it against the
        manifest, which is what lets a caller take one reading before an edit and one after and
        report only what the solve actually moved. An environment the lock does not carry
        answers with nothing, since a snapshot of what is not there yet is empty rather than an
        error.
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
        """Whether pixi ever finished installing ``env``.

        pixi stamps the environment fingerprint only once an installation completes, so an
        existing prefix directory is not enough. An interrupted install leaves one behind.
        """
        return (self.env_prefix(env) / "conda-meta" / _FINGERPRINT).is_file()

    def run(self, command: Sequence[str], env: str = "default") -> int:
        """Run exact task or command argv through Pixi's cross-platform runner.

        Pixi owns environment activation while each caller-owned token remains a distinct
        process argument. No intermediate shell reparses quoting or operators.

        command: task name and arguments, or an ad-hoc command argv.
        env: generated environment in which to execute it.
        """
        return self.within_cwd(Process.passthrough, "run", "--frozen", "-e", env, *command)

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """Run through Pixi while capturing output under ``timeout`` seconds."""
        return self.within_cwd(
            lambda argv: Process.capture(argv, timeout=timeout),
            "run",
            "--frozen",
            "-e",
            env,
            *command,
        )

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

    @staticmethod
    def _floor_overrides(manifest: Path) -> dict[str, str]:
        """The `CONDA_OVERRIDE_*` values for every floor the generated `manifest` declares.

        Floors are read from the workspace platform descriptors the compiler wrote, the one place
        they already live, and an override the process env already carries is left to stand so a
        caller keeps the last word.

        manifest: path to the generated pixi manifest.
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
        """The activation script for ``env`` as a sourceable ``shell`` snippet.

        It carries the env vars, PATH, and `[activation] scripts` pixi sets when entering the
        env, the exact activation :meth:`activated` performs, captured as text so a generated
        `activate.sh` can reproduce the whole pixi env without invoking pixi at job time.
        """
        command = self.command["shell-hook", "-s", shell, "-e", env, *self.scope()]
        return Process.output(command, "pixi shell-hook")

    def update(self, env: str) -> None:
        """Move ``env``'s lock to the newest releases the manifest still allows.

        The one verb that re-reads the indexes inside the declared bounds. `install` keeps
        whatever the lock already pins as long as it satisfies the manifest, which is the right
        default and the reason asking for newer releases has to be its own request.
        """
        if self.within_cwd(Process.stream, "update", "-e", env).returncode:
            raise MissionError("`pixi update` failed (see its output above)")

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
                f"the manifest drifted from pixi.lock. Run `{Project().name} install --resolve` "
                "on a solve-capable machine, which is also what a host is then sent."
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
