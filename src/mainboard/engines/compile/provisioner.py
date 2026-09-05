import os
import re
from contextlib import contextmanager, suppress
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from plumbum import local

from ...core import MissionError, Project
from ...manifest.schema.environment import Env
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
    """Validate a logical environment name before using it as a generated path segment.

    The contract is deliberately portable rather than host-dependent: a manifest accepted on
    Linux must not become an invalid or aliased directory when the same lock reaches Windows.
    Windows device names remain reserved even with an extension, so ``con.txt`` is rejected
    beside the obvious traversal and separator spellings.
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


# Where a generated environment records the lock and manifest its prefix was last brought in
# line with, so the update pixi would otherwise perform inside every command happens once.
_SYNCED = ".mainboard-synced"

# The directory each logical environment's generated shard lives under, inside the generated
# tree. One spelling, because its depth is also what every workspace-relative path inside a
# compiled artifact is written against (see `pixi_manifest.rerooted`).
_ENVS = "envs"


def environment_shard(environment: str) -> PurePosixPath:
    """The workspace-relative directory `environment`'s compiled artifact is generated into.

    Path arithmetic alone, so a caller that has only a name, and no workspace to ask, still
    knows where that environment's manifest and lock are written and how deep in the tree they
    sit.
    """
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
    """``command`` handed back to pixi when its first word names a task compiled for ``env``.

    A declared task reaches an environment through the generated `pixi.toml`, and pixi is the
    runner that resolves one, so a task name goes to pixi rather than to a shell that knows no
    such command. Everything else is returned exactly as written, so an ordinary command line
    still runs as the line it is. The generated manifest is named relatively because every
    wrapped command has already changed into the workspace root, on this machine or a remote
    one.

    manifest: the workspace manifest declaring the tasks.
    command: the command line as the caller wrote it.
    env: the environment the command runs in, whose own tasks join the workspace-wide ones.
    """
    validate_environment_roster(manifest)
    env = environment_segment(env)
    declared = {*manifest.tasks, *manifest.envs.get(env, Env()).tasks}
    if command.partition(" ")[0] not in declared:
        return command
    generated = environment_shard(env) / Pixi.filename
    # Frozen, or every task invocation could silently re-solve and rewrite the lock, which
    # on a remote host would overwrite the pair the workstation shipped. Locks change only
    # through an explicit resolve.
    return f"pixi run --manifest-path {generated.as_posix()} --frozen -e {env} {command}"


class _EnvironmentShard:
    """The compiler and installers bound to one generated environment directory.

    Everything here reads this environment's projection of the manifest, except the vendored
    path dependencies, which are the workspace's rather than one environment's: they live beside
    the shards, several environments may declare the same distribution, and a compile that saw
    only its own projection would retire what the environment beside it depends on.
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

    Every entry point recompiles under one lock when the manifest has moved on, so a caller is
    never served env vars or dependencies from a `.mainboard/` that no longer matches
    ``manifest``. pixi installs conda and Python, and the second stage then installs every
    other ecosystem the manifest declares into that same environment.
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
        """The generated directory owned by one logical environment."""
        return self._shard(environment).directory

    def pixi_for(self, environment: str = "default") -> Pixi:
        """The Pixi backend whose manifest and prefix belong to ``environment``."""
        return self._shard(environment).pixi

    def compiler_for(self, environment: str = "default") -> Compiler:
        """The compiler whose projection and state belong to ``environment``."""
        return self._shard(environment).compiler

    def solver_version(self) -> str:
        """The version of the pixi that solves here, empty on a machine that has none."""
        return self._shard("default").pixi.version()

    @property
    def pixi(self) -> Pixi:
        """The default shard's Pixi backend, retained for default-environment callers."""
        return self.pixi_for()

    @property
    def stage(self) -> SecondStage:
        """The default shard's second-stage compiler."""
        return self._shard().stage

    @property
    def compiler(self) -> Compiler:
        """The default shard's compiler."""
        return self.compiler_for()

    @property
    def artifact(self) -> tuple[str, ...]:
        """The compiled dependency artifact a host installs from, workspace-relative.

        The generated pixi manifest, the lock solved from it, and the state naming which
        resolution that lock was solved from. Shipping the three together is what lets a host
        install frozen instead of solving on its own toolchain, which is the whole point: a
        solve reads dependency metadata, reading metadata builds source distributions, and a
        host's compiler is the last thing that belongs in a lock's dependency path.
        """
        return self.artifact_for("default")

    def artifact_for(self, environment: str) -> tuple[str, ...]:
        """The manifest, lock and state belonging to ``environment`` alone."""
        shard = self._shard(environment)
        paths = (
            shard.pixi.manifest,
            shard.pixi.lock,
            SyncState.path(shard.directory),
        )
        return tuple(path.relative_to(self.root).as_posix() for path in paths)

    def activate(self, env: str = "default", *, modules: Mapping[str, str] = {}) -> Path:
        """Write ``env``'s generated activation script for this host and return its path.

        Formats ``modules`` (name -> version, a per-host map since Lmod stacks differ machine
        to machine) as guarded `module purge` + `module load` lines, followed by pixi's own
        activation and the second stage's own binary directories, so a job or interactive
        shell that `source`s it reaches everything this workspace installed, not only what pixi
        did. Each environment writes its own script, so installing one never overwrites the
        activation another environment's commands still source.
        """
        self.out.mkdir(exist_ok=True)
        shard = self._shard(env)
        path = self.root / Project().activation(env)
        hook = shard.pixi.shell_hook(env)
        return ActivationScript(path, hook, self.binaries(env)).write(modules)

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Recompile ``env`` if stale, then expose everything it installed on PATH for the block.

        pixi's own `bin/` comes first, and the directories the second-stage toolchains link
        into go ahead of it, so a tool installed by npm is reachable by name exactly like a
        conda one, the same order the generated `activate.sh` writes.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            if shard.compiler.stale():
                shard.compiler.write(files)
        with shard.pixi.activated(env):
            installed = [str(directory) for directory in self.binaries(env)]
            with local.env(PATH=os.pathsep.join([*installed, str(local.env["PATH"])])):
                yield

    def run(self, command: Sequence[str], env: str = "default") -> int:
        """Compile stale generated files, then let Pixi activate and run ``command``.

        Local execution deliberately goes through Pixi instead of a host shell. Pixi already
        owns the environment and a cross-platform task shell, so Windows and POSIX machines
        execute the same manifest without mainboard maintaining a second command grammar.
        """
        shard = self.refreshed(env)
        with local.cwd(str(self.root)):
            return shard.pixi.run(command, env)

    def capture(
        self, command: Sequence[str], env: str = "default", *, timeout: float | None = None
    ) -> CommandResult:
        """Compile stale files, then capture a bounded command through Pixi."""
        shard = self.refreshed(env)
        with local.cwd(str(self.root)):
            return shard.pixi.capture(command, env, timeout=timeout)

    def refreshed(self, env: str) -> _EnvironmentShard:
        """``env``'s compile stack with its generated files current, ready to run a command in.

        A recompile rewrites the generated manifest, and on Windows the cached activation is
        read as stale the moment that file is newer than it, so a manifest edit followed by an
        ordinary `run` used to recompile, invalidate the cache and then refuse every command
        until someone reinstalled the whole environment. The cache is Pixi's own answer about a
        prefix that is already installed, so it is retaken here beside the recompile that
        invalidated it, and the refusal is kept for the one case it was written for: a prefix
        that is genuinely not installed.

        env: the environment whose generated files are being brought up to date.
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

        pixi does this on the way into every command it runs, which is a race when nine jobs
        start together out of one pinned tree: they share the prefix, each decides for itself
        that it needs updating, and the loser meets the environment mid-write. Doing it here
        makes it one process at a time, and stamping what was synced makes it happen once
        rather than once per command: the stamp names the lock and the manifest the prefix was
        brought in line with, so nothing runs again until one of them moves.

        An environment nothing has installed is left alone, and so is one with no lock to be in
        line with. A command in either is refused by the activation with the one line that names
        the install to run, and syncing there would answer that question with pixi's words
        instead.

        shard: the environment's compile stack.
        env: the environment being entered.
        """
        if not shard.pixi.ready(env) or not shard.pixi.lock.is_file():
            return
        stamp = shard.directory / _SYNCED
        current = f"{shard.compiler.digest()}:{shard.pixi.lock.stat().st_mtime_ns}"
        with suppress(OSError):
            if stamp.read_text(encoding="utf-8") == current:
                return
        shard.pixi.sync(env)
        with suppress(OSError):
            stamp.write_text(current, encoding="utf-8")

    def binaries(self, env: str) -> list[Path]:
        """The second-stage binary directories that exist, in the order PATH should carry them.

        A directory nothing has installed into yet is left out rather than exported as a dead
        PATH entry, so an environment provisioned without a `[nodejs]` table exports nothing.
        """
        return [
            directory
            for directory in self._shard(env).stage.binary_dirs(env)
            if directory.is_dir()
        ]

    def provision(
        self, env: str = "default", *, resolve: bool = False, refresh: bool = False
    ) -> None:
        """Compile ``env``, then install it under one lock.

        Unlike :meth:`activated`, this always compiles rather than gating on `Compiler.stale`,
        since `stale` reads "nothing compiled yet" as fresh (first provisioning is exactly this
        method's job, not `activated`'s), and the writer it goes through is itself a no-op once
        the generated file already matches, so an unconditional compile costs nothing extra on
        an already-fresh env. The whole provisioning runs under the workspace lock, not just the
        compile, so two agents sharing this checkout never let one rewrite the manifest while
        the other is still solving against it. The second stage runs last, inside that same
        lock, because every manager it drives ships as a conda package pixi has just installed.

        ``refresh`` asks the indexes for the newest releases the manifest still allows before
        installing, which is a solve by definition and so implies ``resolve``. Without it a
        provision keeps whatever the lock already pins, since satisfying the manifest and being
        current are different questions and only the caller knows which one was asked.
        """
        shard = self._shard(env)
        with GeneratedFiles(directory=self.out).locked() as files:
            shard.compiler.write(files)
            if refresh:
                shard.pixi.update(env)
            shard.compiler.install_locked(files, resolve=resolve or refresh)
            shard.stage.install(env)
            if shard.pixi.ready(env):
                shard.pixi.cache_windows_activation(env)
