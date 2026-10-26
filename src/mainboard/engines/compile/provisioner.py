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
from .generated import GeneratedFiles
from .generated.activation import write
from .inputs.vendor import Vendor
from .pixi.manifest import selected_manifest
from .pixi.paths import local_sources
from .pixi.policy import LockPolicy
from .state import SyncState

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from pathlib import Path

    from ...manifest import Manifest
    from .backend import CommandResult

_ENVIRONMENT_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _environment_segment(environment: str) -> str:
    """Validate a logical environment name as a generated path segment."""
    if not _ENVIRONMENT_SEGMENT.fullmatch(environment) or environment.endswith("."):
        raise MissionError(
            f"environment {environment!r} cannot name a generated directory; use letters, "
            "digits, dots, underscores or hyphens, starting with a letter or digit"
        )
    return environment


# The lock and resolver inputs a shard's prefix was last synced with, so a sync happens once:
# the suffix of a marker file (`Project.marker`), written under the legacy name, read under any.
_SYNCED = "synced"

# Where each environment's shard lives; its depth is what compiled paths are written against.
_ENVS = "envs"


def environment_shard(environment: str, root: Path | None = None) -> PurePosixPath:
    """The directory `environment`'s artifact is generated into, by name.

    It is relative to workspace `root`, the cwd's workspace by default.
    """
    return PurePosixPath(Project().out_dir(root)) / _ENVS / _environment_segment(environment)


def _validate_environment_roster(manifest: Manifest) -> None:
    """Refuse logical names that alias the same portable shard directory."""
    seen: dict[str, str] = {}
    for environment in ("default", *manifest.envs):
        segment = _environment_segment(environment)
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
    _validate_environment_roster(manifest)
    env = _environment_segment(env)
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
        self.compiler = Compiler(self.stage, Vendor(root, manifest), environment=environment)


class Provisioner:
    """Compiles a manifest into a pixi workspace and keeps it installed and activatable.

    Every entry point recompiles under one lock once the manifest moved on, so nothing is served
    from a stale state directory.
    """

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Recompile `env` if stale, then put everything it installed on PATH for the block.

        Second-stage directories lead pixi's `bin/`, the order `activate.sh` writes.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            if shard.compiler.stale():
                shard.compiler.write(files)
        with shard.pixi.activated(env), self.leading(env):
            yield

    @contextmanager
    def leading(self, env: str) -> Generator[None]:
        """Put `env`'s second-stage binary directories ahead of PATH for the block.

        `pixi run` then puts the prefix's own directories ahead of them.
        """
        installed = [str(directory) for directory in self.binaries(env)]
        with local.env(PATH=os.pathsep.join([*installed, local.env["PATH"]])):
            yield

    @property
    def artifact(self) -> tuple[str, ...]:
        """The default compiled artifact a host installs from, workspace-relative.

        Shipped whole so a host installs frozen: solving there would build sdists with the host's
        compiler, the last thing that belongs in a lock's dependency path.
        """
        return self.artifact_for("default")

    @property
    def compiler(self) -> Compiler:
        return self.compiler_for()

    @property
    def pixi(self) -> Pixi:
        return self.pixi_for()

    @property
    def stage(self) -> SecondStage:
        return self._shard().stage

    @staticmethod
    @contextmanager
    def runtime(shard: _EnvironmentShard, env: str) -> Generator[None]:
        """Hand Pixi an environment the runtime step already added its facts to.

        pixi's activation leaves these variables alone, so a local run matches a dispatched one.
        """
        added = Runtime(shard.pixi.env_prefix(env)).changes(dict(local.env))
        with local.env(**added):
            yield

    @staticmethod
    def _stamped(directory: Path, current: str) -> bool:
        """Whether `directory` carries the sync stamp `current`, false when it has none."""
        with suppress(OSError):
            return Project().marked(directory, _SYNCED).read_text(encoding="utf-8") == current
        return False

    def __init__(self, root: Path, manifest: Manifest) -> None:
        _validate_environment_roster(manifest)
        self.root = root
        self.manifest = manifest
        self.out = Project().out(root)
        self._shards: dict[str, _EnvironmentShard] = {}

    def _shard(self, environment: str = "default") -> _EnvironmentShard:
        """The cached compile stack for one logical environment."""
        environment = _environment_segment(environment)
        self.manifest.environment(environment)
        if environment not in self._shards:
            directory = self.root / environment_shard(environment, self.root)
            self._shards[environment] = _EnvironmentShard(
                self.root, self.manifest, directory, environment
            )
        return self._shards[environment]

    def activate(self, env: str = "default", *, modules: Mapping[str, str] = {}) -> Path:
        """Write `env`'s activation into its own directory for this host, returning it.

        modules: this host's Lmod stack, name to version.
        """
        shard = self._shard(env)
        path = self.root / Project().activation(env)
        path.parent.mkdir(parents=True, exist_ok=True)
        return write(
            path, shard.pixi.shell_hook(env), modules=modules, binaries=self.binaries(env)
        )

    def artifact_for(self, environment: str) -> tuple[str, ...]:
        """Every defining generated input and lock, second-stage locks required locally.

        The committed lock rides along once there is one, whatever the host's sync scope says,
        so a host on this release installs from it; the shard's materialized copy and its state
        still ship for a host on an older one.
        """
        shard = self._shard(environment)
        committed = Project().lock(self.root)
        paths = (
            shard.pixi.manifest,
            shard.pixi.lock,
            SyncState.path(shard.directory),
            *GeneratedFiles(directory=shard.directory).inputs,
            *shard.stage.frozen_inputs(environment),
            *(
                self.root / path
                for path in local_sources(
                    shard.pixi.manifest.read_text(encoding="utf-8"),
                    generated_dir=shard.directory.relative_to(self.root),
                )
                if (self.root / path).is_file()
            ),
            *((committed,) if committed.is_file() else ()),
        )
        return tuple(dict.fromkeys(path.relative_to(self.root).as_posix() for path in paths))

    def binaries(self, env: str) -> list[Path]:
        """The existing second-stage binary directories in PATH order, never a dead entry."""
        return [
            directory
            for directory in self._shard(env).stage.binary_dirs(env)
            if directory.is_dir()
        ]

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """Compile stale files, then capture a bounded command through Pixi."""
        shard = self.refreshed(env)
        with local.cwd(str(self.root)), self.runtime(shard, env), self.leading(env):
            return shard.pixi.capture(command, env, timeout=timeout)

    def compiler_for(self, environment: str = "default") -> Compiler:
        return self._shard(environment).compiler

    def environment_dir(self, environment: str = "default") -> Path:
        return self._shard(environment).directory

    def pixi_for(self, environment: str = "default") -> Pixi:
        return self._shard(environment).pixi

    def provision(
        self,
        env: str = "default",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        update: Sequence[str] | None = None,
        install: bool = True,
    ) -> None:
        """Compile `env`, bring its lock to `policy` and install it, as `pixi install` does.

        All of it runs under the workspace lock.

        The whole run holds the lock so no agent rewrites the manifest mid-solve; the second
        stage runs last since its managers are conda packages pixi just installed, and re-solves
        its own locks when the pixi lock moved or the policy lets this install solve.

        update: take the newest releases the manifest allows first (`pixi update`), only these
            packages when any are named.
        install: install what the lock pins; off, only the lock moves (`pixi lock`).
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            shard.compiler.write(files)
            # A solve starts from the committed lock, so it moves only what the manifest moved.
            shard.compiler.materialize(files)
            moved = shard.compiler.locked(files, policy=policy, update=update)
            if not install or not shard.pixi.runs_here():
                # Solved for platforms this machine cannot run: the lock ships with `setup`,
                # and the host that runs it installs the second stage and its activation.
                return
            shard.pixi.install(env)
            # Solved whenever this policy may solve, not only when the pixi lock moved in this
            # call: an `upgrade --no-install` moved it and raised `package.json` earlier, and the
            # frozen pnpm install that followed refused the stale lock (2026-10-23). A manager
            # resolving a lock that already matches changes nothing.
            shard.stage.install(env, resolve=moved or policy is LockPolicy.UPDATE)

    def recompiled(self, env: str = "default") -> None:
        """Bring `env`'s generated artifact and lock in line, touching nothing else.

        The artifact follows the manifest and the lock follows the committed one.

        All a dispatch needs, without `refreshed`'s local sync: an older compile once moved the
        workstation's address away from the host's and killed a wave at environment prime.
        Unconditional, since `stale` reads nothing compiled as fresh and a no-op write is free.
        """
        with GeneratedFiles(directory=self.out).locked() as files:
            compiler = self._shard(env).compiler
            compiler.write(files)
            compiler.materialize(files)

    def refreshed(
        self, env: str, *, policy: LockPolicy = LockPolicy.FROZEN, install: bool = True
    ) -> _EnvironmentShard:
        """`env`'s compile stack, ready to run a command in.

        Its generated files are current, its lock brought to `policy` and, unless `install` is off,
        its prefix in line with that lock.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            if shard.compiler.stale() or not shard.pixi.manifest.exists():
                shard.compiler.write(files)
            if policy is not LockPolicy.FROZEN:
                shard.compiler.materialize(files)
                shard.compiler.locked(files, policy=policy)
            if install:
                self.synchronized(shard, env)
        return shard

    def run(
        self,
        command: Sequence[str],
        env: str = "default",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        install: bool = True,
    ) -> int:
        """Bring `env` to `policy` as `pixi run` does, then let Pixi's runner run `command`."""
        shard = self.refreshed(env, policy=policy, install=install)
        with local.cwd(str(self.root)), self.runtime(shard, env), self.leading(env):
            return shard.pixi.run(command, env)

    def runs_here(self, env: str = "default") -> bool:
        """Whether `env` declares this machine's platform, so it can be installed here."""
        return self._shard(env).pixi.runs_here()

    def solver_version(self) -> str:
        """The pixi that solves here, empty on a machine that has none."""
        return self._shard("default").pixi.version()

    def stage_for(self, environment: str = "default") -> SecondStage:
        return self._shard(environment).stage

    def synchronized(self, shard: _EnvironmentShard, env: str) -> None:
        """Bring `env`'s prefix in line with its lock once, under the lock the caller holds.

        pixi's own per-command sync races when a wave shares a prefix (see `Pixi.sync`). The
        stamp names the lock revision and resolver inputs, so tasks and activation never force a
        reinstall; tool configuration and editable native sources still need an explicit one.
        An environment never installed is installed whole, second stage included, as `pixi run`
        installs one; an unlocked one is left for activation to refuse by name.
        """
        if not shard.pixi.lock.is_file():
            return
        current = f"{shard.compiler.resolution_digest()}:{shard.pixi.lock.stat().st_mtime_ns}"
        if shard.pixi.ready(env):
            if self._stamped(shard.directory, current):
                return
            shard.pixi.sync(env)
        elif shard.pixi.runs_here():
            shard.pixi.install(env)
            shard.stage.install(env)
        with suppress(OSError):
            (shard.directory / Project().marker(_SYNCED)).write_text(
                current, encoding="utf-8", newline="\n"
            )
