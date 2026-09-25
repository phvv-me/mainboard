import json
import os
import platform
import tomllib
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import TYPE_CHECKING, cast

from plumbum import local
from plumbum.commands.base import BoundEnvCommand

from ....core import MissionError, Project
from ....core.host import current_platform
from ....runtime.activation import prepended
from .engine import PixiEngine
from .process import Process
from .repair import EnvironmentAudit
from .tool import Tool
from .windows_task import WindowsTaskRunner

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from plumbum.commands.base import BaseCommand

    from .result import CommandResult

# What pixi writes into a prefix's `conda-meta/` once an installation has finished.
_FINGERPRINT = ".pixi-environment-fingerprint"

# Pixi's complete activation, conda package hooks included, recorded after a Windows provision
# (while Pixi can still initialize its auth store) for commands run inside a restricted sandbox.
_WINDOWS_ACTIVATION = "activation-windows.json"

# The env var vouching each virtual-package floor, so a machine that cannot present the package
# (a login node with no GPU driver) still installs the frozen lock its jobs run under.
_FLOOR_OVERRIDES = {
    "archspec": "CONDA_OVERRIDE_ARCHSPEC",
    "cuda": "CONDA_OVERRIDE_CUDA",
    "glibc": "CONDA_OVERRIDE_GLIBC",
    "linux": "CONDA_OVERRIDE_LINUX",
    "macos": "CONDA_OVERRIDE_OSX",
    "osx": "CONDA_OVERRIDE_OSX",
    "windows": "CONDA_OVERRIDE_WIN",
}


def _executable_dirs(prefix: Path, *, windows: bool) -> list[str]:
    """The prefix's existing command directories: root, `Scripts`, `Library/bin` on Windows."""
    candidates = (
        (prefix, prefix / "Scripts", prefix / "Library" / "bin") if windows else (prefix / "bin",)
    )
    return [str(candidate) for candidate in candidates if candidate.is_dir()]


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

        Read per invocation, since the compiler may have just rewritten the floors. Bound in one
        overlay so a floor cannot discard Windows' HOME binding, and outright rather than through
        `with_env`, which returns a bare command when empty, so callers see one shape.
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
        """Prepend the environment's existing executable directories to PATH for the block."""
        windows = platform.system() == "Windows"
        binaries = _executable_dirs(self.env_prefix(env), windows=windows)
        with local.env(PATH=os.pathsep.join([*binaries, str(local.env["PATH"])])):
            yield

    @contextmanager
    def direct_windows_environment(self, env: str) -> Generator[None]:
        """Activate a Windows prefix with an accessible profile and temporary storage.

        The temporary directory lives outside the workspace, since provenance sampled inside this
        context would otherwise see a dirty tree.
        """
        exported, cleared = self._windows_activation()
        with (
            TemporaryDirectory(prefix="mainboard-run-", ignore_cleanup_errors=True) as temporary,
            local.env(
                **exported,
                HOME=str(Path.home()),
                TEMP=temporary,
                TMP=temporary,
            ),
            self.activated(env),
        ):
            for name in cleared:
                if name in local.env:
                    del local.env[name]
            yield

    def env_prefix(self, env: str) -> Path:
        return self.manifest.parent / ".pixi" / "envs" / env

    def environment_result(self, verb: str, *args: str, resolve: bool = False) -> CommandResult:
        """Run an environment verb and retain its streamed native output."""
        if not resolve and not self.lock.exists():
            raise MissionError(
                f"pixi.lock is missing. Run `{Project().name} install --resolve` on a "
                "solve-capable machine to create and verify the generated manifest/lock pair."
            )
        editable = self._has_editable_paths()
        return self.within_cwd(
            Process.stream,
            verb,
            *args,
            locked=not resolve and not editable,
            frozen=not resolve and editable,
        )

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

    def install(self, env: str, *, resolve: bool = False) -> None:
        """Install `env` locked by default and verify every explicitly resolved lock."""
        result = self.environment_result("install", "-e", env, resolve=resolve)
        self._raise_on_lock_drift(result, locked=not resolve)
        self._raise_on_inaccessible_windows_home(result, env=env, resolve=resolve)
        if result.returncode:
            raise MissionError("`pixi install` failed (see its output above)")
        # Known wart from chefe: a resolve installs twice, the repair riding the second, locked
        # call, so an environment is audited once against a verified lock.
        if resolve:
            self.install(env)
        else:
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

    def run(
        self,
        command: Sequence[str],
        env: str = "default",
        *,
        exports: dict[str, str] | None = None,
    ) -> int:
        """Run a task or command argv through Pixi, each token a distinct argument, no shell.

        Under a restricted Windows sandbox, where Pixi 0.78's auth store cannot resolve the
        profile even with `HOME` and `USERPROFILE` right, an argv runs straight from the prefix
        and a declared task through `WindowsTaskRunner`, both under the cached activation and
        without starting Pixi.

        command: a task name and arguments, or an ad-hoc argv.
        """
        if self._restricted_windows_command(command):
            runner = self._restricted_runner(env)
            with self.direct_windows_environment(env), local.env(**(exports or {})):
                if command[0] in runner.tasks:
                    return runner.run(command, Process.stream).returncode
                return Process.passthrough(local[command[0]][command[1:]])
        if exports:
            command = ["env", *(f"{name}={value}" for name, value in exports.items()), *command]
        return self.within_cwd(Process.passthrough, "run", "--frozen", "-e", env, *command)

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """`run`, capturing output under `timeout` seconds."""
        if self._restricted_windows_command(command):
            runner = self._restricted_runner(env)
            with self.direct_windows_environment(env):
                if command[0] in runner.tasks:
                    deadline = None if timeout is None else monotonic() + timeout

                    def capture_task(argv: BaseCommand) -> CommandResult:
                        remaining = None if deadline is None else max(deadline - monotonic(), 0.0)
                        return Process.capture(argv, timeout=remaining)

                    return runner.run(command, capture_task)
                return Process.capture(local[command[0]][command[1:]], timeout=timeout)
        return self.within_cwd(
            lambda argv: Process.capture(argv, timeout=timeout),
            "run",
            "--frozen",
            "-e",
            env,
            *command,
        )

    def _restricted_runner(self, env: str) -> WindowsTaskRunner:
        """The sandbox task runner for an installed `env`."""
        if not self.ready(env):
            raise MissionError(
                f"environment {env!r} is not installed; run `{Project().name} install {env}`"
            )
        return WindowsTaskRunner(self.manifest, env)

    def _restricted_windows_command(self, command: Sequence[str]) -> bool:
        """Whether a Windows command can use cached activation and explicit auth storage."""
        return (
            platform.system() == "Windows"
            and bool(command)
            and self.windows_activation_cache.is_file()
        )

    @property
    def windows_activation_cache(self) -> Path:
        return self.manifest.parent / _WINDOWS_ACTIVATION

    def cache_windows_activation(self, env: str, binaries: Sequence[Path]) -> None:
        """Persist Pixi's full Windows activation, including every conda package hook.

        binaries: the second-stage executable directories, recorded leading the activated `PATH`
            the way `activate.sh` puts them, since every Windows entry reads this record instead.
        """
        if platform.system() != "Windows":
            return
        text = self.within_cwd(
            lambda command: Process.output(command, "pixi shell-hook --json"),
            "shell-hook",
            "--frozen",
            "--json",
            "-e",
            env,
        )
        recorded = json.loads(text)
        variables = recorded["environment_variables"]
        path = next((name for name in variables if name.upper() == "PATH"), "PATH")
        prepended(variables, path, binaries)
        self.windows_activation_cache.write_text(json.dumps(recorded), encoding="utf-8")

    def recorded_environment(self, env: str, base: Mapping[str, str]) -> dict[str, str]:
        """`base` entered into `env` the way a restricted command enters it, as a plain mapping.

        The recorded variables over `base`, the declared clears taken out, and the prefix's
        executable directories leading `PATH`.
        """
        exported, cleared = self._windows_activation()
        entered = {
            name: value for name, value in {**base, **exported}.items() if name not in cleared
        }
        binaries = _executable_dirs(self.env_prefix(env), windows=True)
        entered["PATH"] = os.pathsep.join([*binaries, entered.get("PATH", "")])
        return entered

    def _windows_activation(self) -> tuple[dict[str, str], set[str]]:
        """The recorded activation's exports and clears, its generated scripts applied."""
        exported, scripts = self._cached_windows_activation()
        cleared: set[str] = set()
        for script in scripts:
            self._apply_generated_activation(script, exported, cleared)
        return exported, cleared

    def _cached_windows_activation(self) -> tuple[dict[str, str], list[Path]]:
        """Load the complete activation Pixi recorded when this prefix was provisioned."""
        cache = self.windows_activation_cache
        try:
            if cache.stat().st_mtime_ns < self.manifest.stat().st_mtime_ns:
                raise MissionError(
                    f"Windows activation changed; run `{Project().name} install` outside the "
                    "application sandbox"
                )
            decoded = json.loads(cache.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise MissionError(
                f"Windows activation is not cached; run `{Project().name} install` outside the "
                "application sandbox"
            ) from error
        variables = decoded.get("environment_variables", {})
        scripts = decoded.get("activation_scripts", [])
        if not isinstance(variables, dict) or not isinstance(scripts, list):
            raise MissionError(f"Windows activation cache is invalid: {cache}")
        exported = {
            name.upper(): value
            for name, value in variables.items()
            if isinstance(name, str) and isinstance(value, str)
        }
        return exported, [Path(script) for script in scripts if isinstance(script, str)]

    @staticmethod
    def _apply_generated_activation(
        script: Path, exported: dict[str, str], cleared: set[str]
    ) -> None:
        """Apply Mainboard's generated dotenv/unset batch scripts, refusing arbitrary ones.

        A POSIX `.sh` is skipped: pixi cannot run it on Windows either.
        """
        if script.suffix == ".sh":
            return
        try:
            text = script.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise MissionError(f"Windows activation script does not exist: {script}") from error
        if "Generated by mainboard's Provisioner" in text and script.name == "dotenv.bat":
            location = next(
                line.partition('"')[2].partition('"')[0]
                for line in text.splitlines()
                if line.strip().lower().startswith('if not exist "')
            )
            dotenv = script.parent / location
            try:
                lines = dotenv.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                return
            for line in lines:
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                # Upper-cased like the cached activation, since the batch `if not defined` is
                # case-blind; `foo` beside a cached `FOO` put both in one CreateProcess block.
                key = name.upper()
                if key and key not in os.environ and key not in exported:
                    exported[key] = value
            return
        if "Generated by mainboard from the [env] table" in text and script.name == "unset.bat":
            cleared.update(
                line.removeprefix("set ").removesuffix("=")
                for line in text.splitlines()
                if line.lower().startswith("set ") and line.endswith("=")
            )
            return
        raise MissionError(
            f"restricted Windows execution cannot reproduce activation script {script}; "
            "run this command outside the application sandbox"
        )

    def repair(self, env: str) -> None:
        """Reinstall whatever `env` holds that can no longer be trusted to import.

        Only a finished install is audited, since a half-written prefix reads damaged everywhere.
        A wheel still missing every import root after the reinstall raises.
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
        # how a Windows host came to build an osx-arm64 sdist (2026-09-11).
        command = self.command["shell-hook", "--frozen", "-s", shell, "-e", env, *self.scope()]
        return Process.output(command, "pixi shell-hook")

    def update(self, env: str) -> None:
        """Move `env`'s lock to the newest releases the manifest allows, which `install` keeps."""
        if self.within_cwd(Process.stream, "update", "-e", env).returncode:
            raise MissionError("`pixi update` failed (see its output above)")

    @staticmethod
    def _raise_on_lock_drift(result: CommandResult, *, locked: bool) -> None:
        """Turn Pixi's pre-task lock rejection into an actionable recovery message."""
        failure = f"{result.stdout}\n{result.stderr}".lower().replace("-", " ")
        if (
            result.returncode
            and locked
            and "pixi task (" not in failure
            and "lock file" in failure
            and "not up to date" in failure
        ):
            raise MissionError(
                f"the manifest drifted from pixi.lock. Run `{Project().name} install --resolve` "
                "on a solve-capable machine, which is also what a host is then sent."
            )

    @staticmethod
    def _raise_on_inaccessible_windows_home(
        result: CommandResult, *, env: str, resolve: bool
    ) -> None:
        """Explain the one Pixi provisioning failure caused by a restricted Windows profile."""
        failure = f"{result.stdout}\n{result.stderr}".casefold()
        if (
            result.returncode
            and platform.system() == "Windows"
            and "filestorageerror" in failure
            and "could not determine the home directory" in failure
        ):
            environment = "" if env == "default" else f" {env}"
            resolution = " --resolve" if resolve else ""
            raise MissionError(
                "Pixi could not access the Windows home/profile required for provisioning. "
                f"Run `{Project().name} install{environment}{resolution}` from a regular "
                "terminal outside the restricted application sandbox."
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
