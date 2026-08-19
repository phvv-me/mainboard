import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from plumbum import local

from ...core import Project
from ...manifest.schema.environment import Env
from .backend import Pixi
from .compiler import Compiler
from .ecosystems import SecondStage
from .generated import ActivationScript, GeneratedFiles
from .state import SyncState

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

    from ...manifest import Manifest


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
    declared = {*manifest.tasks, *manifest.envs.get(env, Env()).tasks}
    if command.partition(" ")[0] not in declared:
        return command
    generated = f"{Project().out_dir}/{Pixi.filename}"
    # Frozen, or every task invocation could silently re-solve and rewrite the lock, which
    # on a remote host would overwrite the pair the workstation shipped. Locks change only
    # through an explicit resolve.
    return f"pixi run --manifest-path {generated} --frozen -e {env} {command}"


class Provisioner:
    """Compiles a manifest into a pixi workspace and keeps it installed and activatable.

    Every entry point recompiles under one lock when the manifest has moved on, so a caller is
    never served env vars or dependencies from a `.mainboard/` that no longer matches
    ``manifest``. pixi installs conda and Python, and the second stage then installs every
    other ecosystem the manifest declares into that same environment.
    """

    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest
        self.out = root / Project().out_dir
        self.pixi = Pixi(self.out)
        self.stage = SecondStage(root, manifest, self.out, self.pixi)
        self.compiler = Compiler(root, manifest, self.out, self.pixi, self.stage)

    @property
    def artifact(self) -> tuple[str, ...]:
        """The compiled dependency artifact a host installs from, workspace-relative.

        The generated pixi manifest, the lock solved from it, and the state naming which
        resolution that lock was solved from. Shipping the three together is what lets a host
        install frozen instead of solving on its own toolchain, which is the whole point: a
        solve reads dependency metadata, reading metadata builds source distributions, and a
        host's compiler is the last thing that belongs in a lock's dependency path.
        """
        paths = (self.pixi.manifest, self.pixi.lock, SyncState.path(self.out))
        return tuple(str(path.relative_to(self.root)) for path in paths)

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
        path = self.root / Project().activation(env)
        hook = self.pixi.shell_hook(env)
        return ActivationScript(path, hook, self.binaries(env)).write(modules)

    @contextmanager
    def activated(self, env: str = "default") -> Generator[None]:
        """Recompile ``env`` if stale, then expose everything it installed on PATH for the block.

        pixi's own `bin/` comes first, and the directories the second-stage toolchains link
        into go ahead of it, so a tool installed by npm is reachable by name exactly like a
        conda one, the same order the generated `activate.sh` writes.
        """
        with GeneratedFiles(directory=self.out).locked() as files:
            if self.compiler.stale(env):
                self.compiler.write(files, env)
        with self.pixi.activated(env):
            installed = [str(directory) for directory in self.binaries(env)]
            with local.env(PATH=os.pathsep.join([*installed, str(local.env["PATH"])])):
                yield

    def binaries(self, env: str) -> list[Path]:
        """The second-stage binary directories that exist, in the order PATH should carry them.

        A directory nothing has installed into yet is left out rather than exported as a dead
        PATH entry, so an environment provisioned without a `[nodejs]` table exports nothing.
        """
        return [directory for directory in self.stage.binary_dirs(env) if directory.is_dir()]

    def provision(self, env: str = "default", *, resolve: bool = False) -> None:
        """Compile ``env``, then install it under one lock.

        Unlike :meth:`activated`, this always compiles rather than gating on `Compiler.stale`,
        since `stale` reads "nothing compiled yet" as fresh (first provisioning is exactly this
        method's job, not `activated`'s), and the writer it goes through is itself a no-op once
        the generated file already matches, so an unconditional compile costs nothing extra on
        an already-fresh env. The whole provisioning runs under the workspace lock, not just the
        compile, so two agents sharing this checkout never let one rewrite the manifest while
        the other is still solving against it. The second stage runs last, inside that same
        lock, because every manager it drives ships as a conda package pixi has just installed.
        """
        with GeneratedFiles(directory=self.out).locked() as files:
            self.compiler.write(files, env)
            self.compiler.install_locked(files, env, resolve=resolve)
            self.stage.install(env)
