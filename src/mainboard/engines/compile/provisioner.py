import os
import re
from contextlib import contextmanager, suppress
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from plumbum import local

from ...core import MissionError, Project
from ...manifest.schema.environment import Env
from ...runtime.activation import Runtime
from .backend import Pixi
from .compiler import Compiler
from .ecosystems import SecondStage
from .generated import ActivationScript, GeneratedFiles
from .pixi_manifest import selected_manifest
from .state import SyncState
from .vendor import Vendor

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from pathlib import Path

    from ...manifest import Manifest
    from .backend import CommandResult

_ENVIRONMENT_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def environment_segment(environment: str) -> str:
    """Validate a logical environment name as a generated path segment, portably.

    What Linux accepts must stay a valid, unaliased directory on Windows, so device names are
    refused even with an extension (`con.txt`).
    """
    stem = environment.partition(".")[0].upper()
    if (
        not _ENVIRONMENT_SEGMENT.fullmatch(environment)
        or environment.endswith((".", " "))
        or stem in _WINDOWS_DEVICES
    ):
        raise MissionError(
            f"environment {environment!r} cannot name a generated directory; use letters, "
            "digits, dots, underscores or hyphens, starting with a letter or digit, and avoid "
            "Windows device names"
        )
    return environment


# The lock and resolver inputs a shard's prefix was last synced with, so a sync happens once.
_SYNCED = ".mainboard-synced"

# Where each environment's shard lives; its depth is what compiled paths are written against.
_ENVS = "envs"


def environment_shard(environment: str) -> PurePosixPath:
    """The workspace-relative directory `environment`'s artifact is generated into, by name."""
    return PurePosixPath(Project().out_dir) / _ENVS / environment_segment(environment)


def validate_environment_roster(manifest: Manifest) -> None:
    """Refuse logical names that alias the same portable shard directory."""
    seen: dict[str, str] = {}
    for environment in ("default", *manifest.envs):
        segment = environment_segment(environment)
        folded = segment.casefold()
        if folded in seen and seen[folded] != segment:
            raise MissionError(
                f"environments {seen[folded]!r} and {segment!r} name the same directory on "
                "a case-insensitive filesystem"
            )
        seen[folded] = segment


def task_line(manifest: Manifest, command: str, *, env: str) -> str:
    """`command` routed through `pixi run` when its first word names a task for `env`.

    Anything else is returned as written. The manifest path is relative, since every wrapped
    command already runs from the workspace root.
    """
    validate_environment_roster(manifest)
    env = environment_segment(env)
    declared = {*manifest.tasks, *manifest.envs.get(env, Env()).tasks}
    if command.partition(" ")[0] not in declared:
        return command
    generated = environment_shard(env) / Pixi.filename
    # Frozen, or a task could re-solve and overwrite the lock a workstation shipped.
    return f"pixi run --manifest-path {generated.as_posix()} --frozen -e {env} {command}"


class _EnvironmentShard:
    """The compiler and installers bound to one generated environment directory.

    All read this environment's projection, except `Vendor`, which is the workspace's.
    """

    def __init__(self, root: Path, manifest: Manifest, directory: Path, environment: str) -> None:
        projected = selected_manifest(manifest, environment)
        self.directory = directory
        self.pixi = Pixi(directory)
        self.stage = SecondStage(root, projected, directory, self.pixi)
        self.compiler = Compiler(
            root,
            projected,
            directory,
            self.pixi,
            self.stage,
            Vendor(root, manifest),
            environment=environment,
        )


class Provisioner:
    """Compiles a manifest into a pixi workspace and keeps it installed and activatable.

    Every entry point recompiles under one lock once the manifest moved on, so nothing is served
    from a stale `.mainboard/`.
    """

    def __init__(self, root: Path, manifest: Manifest) -> None:
        validate_environment_roster(manifest)
        self.root = root
        self.manifest = manifest
        self.out = root / Project().out_dir
        self._shards: dict[str, _EnvironmentShard] = {}

    def _shard(self, environment: str = "default") -> _EnvironmentShard:
        """The cached compile stack for one logical environment."""
        environment = environment_segment(environment)
        self.manifest.environment(environment)
        if environment not in self._shards:
            directory = self.root / environment_shard(environment)
            self._shards[environment] = _EnvironmentShard(
                self.root, self.manifest, directory, environment
            )
        return self._shards[environment]

    def environment_dir(self, environment: str = "default") -> Path:
        return self._shard(environment).directory

    def pixi_for(self, environment: str = "default") -> Pixi:
        return self._shard(environment).pixi

    def compiler_for(self, environment: str = "default") -> Compiler:
        return self._shard(environment).compiler

    def solver_version(self) -> str:
        """The pixi that solves here, empty on a machine that has none."""
        return self._shard("default").pixi.version()

    @property
    def pixi(self) -> Pixi:
        return self.pixi_for()

    @property
    def stage(self) -> SecondStage:
        return self._shard().stage

    @property
    def compiler(self) -> Compiler:
        return self.compiler_for()

    @property
    def artifact(self) -> tuple[str, ...]:
        """The default compiled artifact a host installs from, workspace-relative.

        Shipped whole so a host installs frozen: solving there would build sdists with the host's
        compiler, the last thing that belongs in a lock's dependency path.
        """
        return self.artifact_for("default")

    def artifact_for(self, environment: str) -> tuple[str, ...]:
        """Every defining generated input and lock, second-stage locks required locally."""
        shard = self._shard(environment)
        paths = (
            shard.pixi.manifest,
            shard.pixi.lock,
            SyncState.path(shard.directory),
            *GeneratedFiles(directory=shard.directory).inputs,
            *shard.stage.frozen_inputs(environment),
        )
        return tuple(dict.fromkeys(path.relative_to(self.root).as_posix() for path in paths))

    def activate(self, env: str = "default", *, modules: Mapping[str, str] = {}) -> Path:
        """Write `env`'s own `activate.sh` for this host (see `ActivationScript`), returning it.

        modules: this host's Lmod stack, name to version.
        """
        self.out.mkdir(exist_ok=True)
        shard = self._shard(env)
        path = self.root / Project().activation(env)
        hook = shard.pixi.shell_hook(env)
        return ActivationScript(path, hook, self.binaries(env)).write(modules)

    def runs_here(self, env: str = "default") -> bool:
        """Whether `env` declares this machine's platform, so it can be installed here."""
        return self._shard(env).pixi.runs_here()

    def recompiled(self, env: str = "default") -> None:
        """Bring `env`'s generated artifact in line with the manifest, and touch nothing else.

        All a dispatch needs, without `refreshed`'s local sync: an older compile once moved the
        workstation's address away from the host's and killed a wave at environment prime.
        Unconditional, since `stale` reads nothing compiled as fresh and a no-op write is free.
        """
        with GeneratedFiles(directory=self.out).locked() as files:
            self._shard(env).compiler.write(files)

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Recompile `env` if stale, then put everything it installed on PATH for the block.

        Second-stage directories lead pixi's `bin/`, the order `activate.sh` writes.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            if shard.compiler.stale():
                shard.compiler.write(files)
        with shard.pixi.activated(env):
            installed = [str(directory) for directory in self.binaries(env)]
            with local.env(PATH=os.pathsep.join([*installed, str(local.env["PATH"])])):
                yield

    def run(
        self,
        command: Sequence[str],
        env: str = "default",
        *,
        exports: dict[str, str] | None = None,
    ) -> int:
        """Compile stale generated files, then let Pixi's cross-platform runner run `command`."""
        shard = self.refreshed(env)
        with local.cwd(str(self.root)), self.runtime(shard, env):
            if exports:
                return shard.pixi.run(command, env, exports=exports)
            return shard.pixi.run(command, env)

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """Compile stale files, then capture a bounded command through Pixi."""
        shard = self.refreshed(env)
        with local.cwd(str(self.root)), self.runtime(shard, env):
            return shard.pixi.capture(command, env, timeout=timeout)

    @staticmethod
    @contextmanager
    def runtime(shard: _EnvironmentShard, env: str) -> Generator[None]:
        """Hand Pixi an environment the runtime step already added its facts to.

        pixi's activation leaves these variables alone, so a local run matches a dispatched one.
        """
        added = Runtime(shard.pixi.env_prefix(env)).changes(dict(local.env))
        with local.env(**added):
            yield

    def refreshed(self, env: str) -> _EnvironmentShard:
        """`env`'s compile stack with its generated files current, ready to run a command in.

        A recompile makes the Windows activation cache read stale, so an installed prefix has
        it retaken here rather than refusing every command until a reinstall.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            recompiled = shard.compiler.stale()
            if recompiled:
                shard.compiler.write(files)
            self.synchronized(shard, env)
        if recompiled and shard.pixi.ready(env):
            shard.pixi.cache_windows_activation(env)
        return shard

    def synchronized(self, shard: _EnvironmentShard, env: str) -> None:
        """Bring `env`'s prefix in line with its lock once, under the lock the caller holds.

        pixi's own per-command sync races when a wave shares a prefix (see `Pixi.sync`). The
        stamp names the lock revision and resolver inputs, so tasks and activation never force a
        reinstall; tool configuration and editable native sources still need an explicit one.
        An uninstalled or unlocked environment is left for activation to refuse by name.
        """
        if not shard.pixi.ready(env) or not shard.pixi.lock.is_file():
            return
        stamp = shard.directory / _SYNCED
        current = f"{shard.compiler.resolution_digest()}:{shard.pixi.lock.stat().st_mtime_ns}"
        with suppress(OSError):
            if stamp.read_text(encoding="utf-8") == current:
                return
        shard.pixi.sync(env)
        with suppress(OSError):
            stamp.write_text(current, encoding="utf-8")

    def binaries(self, env: str) -> list[Path]:
        """The existing second-stage binary directories in PATH order, never a dead entry."""
        return [
            directory
            for directory in self._shard(env).stage.binary_dirs(env)
            if directory.is_dir()
        ]

    def provision(
        self, env: str = "default", *, resolve: bool = False, refresh: bool = False
    ) -> None:
        """Compile `env` unconditionally, then install it, all under the workspace lock.

        The whole run holds the lock so no agent rewrites the manifest mid-solve; the second
        stage runs last since its managers are conda packages pixi just installed.

        refresh: take the newest releases the manifest allows first, a solve implying `resolve`.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            shard.compiler.write(files)
            if refresh:
                shard.pixi.update(env)
            shard.compiler.install_locked(files, resolve=resolve or refresh)
            if not shard.pixi.runs_here():
                # Solved for platforms this machine cannot run: the lock ships with `setup`,
                # and the host that runs it installs the second stage and its activation.
                return
            shard.stage.install(env, resolve=resolve or refresh)
            if shard.pixi.ready(env):
                shard.pixi.cache_windows_activation(env)
