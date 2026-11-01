import os
import re
from contextlib import contextmanager, suppress
from functools import cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from plumbum import local

from ...core import Marker, MissionError, Project
from ...enums import LockPolicy
from ...log import logger
from ...manifest.schema.environment import Env
from ...runtime.activation import Runtime
from .backend import Pixi
from .compiler import Compiler
from .ecosystems import SecondStage
from .generated import GeneratedFiles, SyncState
from .generated.activation import write
from .inputs.vendor import Vendor
from .pixi.manifest import selected_manifest
from .pixi.paths import local_sources

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from ...manifest import Manifest
    from .backend import CommandResult
    from .generated import Writer

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


@cache
def _shadowing(task: str, executable: Path) -> None:
    """Warn that `task` stands in for `executable`, once per process however often asked."""
    logger.warning(
        "task {} shadows {}: `{} run {} ...` runs the task with the arguments appended, never "
        "the executable; rename the task",
        task,
        executable,
        Project().name,
        task,
    )


def _declared_tasks(manifest: Manifest, env: str) -> set[str]:
    """The names of the tasks `env` runs: the workspace's own and the environment's."""
    return {*manifest.tasks, *manifest.envs.get(env, Env()).tasks}


def task_line(manifest: Manifest, command: str, *, env: str) -> str:
    """`command` routed through `pixi run` when its first word names a task for `env`.

    Anything else is returned as written. The manifest path is relative, since every wrapped
    command already runs from the workspace root.
    """
    _validate_environment_roster(manifest)
    env = _environment_segment(env)
    if command.partition(" ")[0] not in _declared_tasks(manifest, env):
        return command
    generated = environment_shard(env) / Pixi.filename
    # Frozen, or a task could re-solve and overwrite the lock a workstation shipped.
    return f"pixi run --manifest-path {generated.as_posix()} --frozen -e {env} {command}"


class Shard:
    """The compiler and installers bound to one generated environment directory.

    All read this environment's projection, except `Vendor`, which is the workspace's.

    environment: the logical environment name.
    directory: where its pixi workspace is generated.
    """

    @contextmanager
    def leading(self) -> Generator[None]:
        """Put the second-stage binary directories ahead of PATH for the block.

        `pixi run` then puts the prefix's own directories ahead of them.
        """
        installed = [str(directory) for directory in self.binaries()]
        with local.env(PATH=os.pathsep.join([*installed, local.env["PATH"]])):
            yield

    @contextmanager
    def runtime(self) -> Generator[None]:
        """Hand Pixi an environment the runtime step already added its facts to.

        pixi's activation leaves these variables alone, so a local run matches a dispatched one.
        """
        added = Runtime(self.pixi.prefix(self.environment).path).changes(dict(local.env))
        with local.env(**added):
            yield

    def __init__(self, root: Path, manifest: Manifest, directory: Path, environment: str) -> None:
        projected = selected_manifest(manifest, environment)
        self.environment = environment
        self.directory = directory
        self.tasks = _declared_tasks(manifest, environment)
        self.pixi = Pixi(directory)
        self.stage = SecondStage(root, projected, directory, self.pixi)
        self.compiler = Compiler(self.stage, Vendor(root, manifest), environment=environment)

    def _stamped(self, current: str) -> bool:
        """Whether the directory carries the sync stamp `current`, false when it has none."""
        with suppress(OSError):
            return Marker(suffix=_SYNCED).at(self.directory).read_text(encoding="utf-8") == current
        return False

    def binaries(self) -> list[Path]:
        """The existing second-stage binary directories in PATH order, never a dead entry."""
        return [
            directory
            for directory in self.stage.binary_dirs(self.environment)
            if directory.is_dir()
        ]

    def generate(self, files: Writer) -> None:
        """Write this environment's generated files, warning of each task that shadows a program.

        Pixi runs a name it knows as a task before it looks for a program, so `run ruff format x`
        with a task `ruff` ran the task with `format x` appended. The programs are those pixi has
        installed in the prefix, so an environment not installed yet has none to shadow.
        """
        self.compiler.write(files)
        for task in sorted(self.tasks):
            if (executable := self.pixi.prefix(self.environment).path / "bin" / task).exists():
                _shadowing(task, executable)

    def synchronized(self) -> None:
        """Bring the prefix in line with its lock once, under the lock the caller holds.

        pixi's own per-command sync races when a wave shares a prefix (see `Pixi.sync`). The
        stamp names the lock revision and resolver inputs, so tasks and activation never force a
        reinstall; tool configuration and editable native sources still need an explicit one.
        An environment never installed is installed whole, second stage included, as `pixi run`
        installs one; an unlocked one is left for activation to refuse by name.
        """
        if not self.pixi.lock.is_file():
            return
        current = f"{self.compiler.resolution_digest()}:{self.pixi.lock.stat().st_mtime_ns}"
        if self.pixi.prefix(self.environment).ready:
            if self._stamped(current):
                return
            self.pixi.sync(self.environment)
        elif self.pixi.declared.runs_here:
            self.pixi.install(self.environment)
            self.stage.install(self.environment)
        with suppress(OSError):
            (self.directory / Project().marker(_SYNCED)).write_text(
                current, encoding="utf-8", newline="\n"
            )


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
        shard = self.shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            if shard.compiler.stale():
                shard.generate(files)
        with shard.pixi.prefix(env).activated(), shard.leading():
            yield

    def __init__(self, root: Path, manifest: Manifest) -> None:
        _validate_environment_roster(manifest)
        self.root = root
        self.manifest = manifest
        self.out = Project().out(root)
        self._shards: dict[str, Shard] = {}

    def _placed(self, command: Sequence[str], env: str) -> None:
        """Say where a bare `command` runs when that is not the directory it was typed in."""
        here = Path.cwd()
        task = bool(command) and command[0] in _declared_tasks(self.manifest, env)
        if not task and here.resolve() != self.root.resolve():
            logger.info("running in {}, the workspace root, not in {}", self.root, here)

    def activate(self, env: str = "default", *, modules: Mapping[str, str] = {}) -> Path:
        """Write `env`'s activation into its own directory for this host, returning it.

        modules: this host's Lmod stack, name to version.
        """
        shard = self.shard(env)
        path = self.root / Project().activation(env)
        path.parent.mkdir(parents=True, exist_ok=True)
        return write(path, shard.pixi.shell_hook(env), modules=modules, binaries=shard.binaries())

    def artifact_for(self, environment: str) -> tuple[str, ...]:
        """Every defining generated input and lock, second-stage locks required locally.

        The committed lock rides along once there is one, whatever the host's sync scope says,
        so a host on this release installs from it; the shard's materialized copy and its state
        still ship for a host on an older one.

        Shipped whole so a host installs frozen: solving there would build sdists with the host's
        compiler, the last thing that belongs in a lock's dependency path.
        """
        shard = self.shard(environment)
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

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """Compile stale files, then capture a bounded command through Pixi."""
        shard = self.refreshed(env)
        with local.cwd(str(self.root)), shard.runtime(), shard.leading():
            return shard.pixi.capture(command, env, timeout=timeout)

    def environment_dir(self, environment: str = "default") -> Path:
        return self.shard(environment).directory

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
        shard = self.shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            shard.generate(files)
            # A solve starts from the committed lock, so it moves only what the manifest moved.
            shard.compiler.materialize(files)
            moved = shard.compiler.locked(files, policy=policy, update=update)
            if not install or not shard.pixi.declared.runs_here:
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
            compiler = self.shard(env).compiler
            compiler.write(files)
            compiler.materialize(files)

    def refreshed(
        self, env: str, *, policy: LockPolicy = LockPolicy.FROZEN, install: bool = True
    ) -> Shard:
        """`env`'s compile stack, ready to run a command in.

        Its generated files are current, its lock brought to `policy` and, unless `install` is off,
        its prefix in line with that lock.
        """
        shard = self.shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            if shard.compiler.stale() or not shard.pixi.manifest.exists():
                shard.generate(files)
            if policy is not LockPolicy.FROZEN:
                shard.compiler.materialize(files)
                shard.compiler.locked(files, policy=policy)
            if install:
                shard.synchronized()
        return shard

    def run(
        self,
        command: Sequence[str],
        env: str = "default",
        *,
        policy: LockPolicy = LockPolicy.UPDATE,
        install: bool = True,
    ) -> int:
        """Bring `env` to `policy` as `pixi run` does, then let Pixi's runner run `command`.

        It runs at the workspace root wherever it was typed, so a relative path in a bare command
        resolves against the root, and says so when that is not the current directory. A declared
        task runs from its own `dir`.
        """
        shard = self.refreshed(env, policy=policy, install=install)
        self._placed(command, env)
        with local.cwd(str(self.root)), shard.runtime(), shard.leading():
            return shard.pixi.run(command, env)

    def shard(self, environment: str = "default") -> Shard:
        """The cached compile stack for one logical environment."""
        environment = _environment_segment(environment)
        self.manifest.environment(environment)
        if environment not in self._shards:
            directory = self.root / environment_shard(environment, self.root)
            self._shards[environment] = Shard(self.root, self.manifest, directory, environment)
        return self._shards[environment]
