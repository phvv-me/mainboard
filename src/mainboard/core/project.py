import os
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from .errors import MissionError
from .membership import Membership

if TYPE_CHECKING:
    from collections.abc import Mapping

_PACKAGE = __name__.split(".")[0]


def _names() -> tuple[str, ...]:
    """The console scripts the installed distribution points at its CLI, shortest first.

    The package name alone when no metadata is installed, as when the source is imported off a
    path rather than installed.
    """
    try:
        scripts = distribution(_PACKAGE).entry_points.select(group="console_scripts")
    except PackageNotFoundError:
        return (_PACKAGE,)
    target = f"{_PACKAGE}.cli:main"
    found = {script.name for script in scripts if script.value == target}
    return tuple(sorted(found, key=lambda name: (len(name), name))) or (_PACKAGE,)


class Variable(FrozenModel):
    """One environment variable the tool trades with the processes it starts, under every name.

    Both ends may run different releases, so an export sets every name (an older reader knows
    only the legacy one) and a read takes the first name set, primary first.
    """

    names: tuple[str, ...]

    def read(self, environ: Mapping[str, str] | None = None) -> str:
        """The value under the first name `environ` (the process environment) sets, else empty."""
        source = os.environ if environ is None else environ
        return next((source[name] for name in self.names if name in source), "")

    def present(self, environ: Mapping[str, str] | None = None) -> bool:
        """Whether `environ` (the process environment) sets the variable under any name."""
        source = os.environ if environ is None else environ
        return any(name in source for name in self.names)

    def exported(self, value: str) -> dict[str, str]:
        """`value` under every name, the mapping a child environment is updated with."""
        return dict.fromkeys(self.names, value)


class Project(FrozenModel):
    """Every name the tool answers to, derived from the console scripts pyproject.toml declares.

    `[project.scripts]` is the one place a name is spelled: the shortest script pointing at the
    CLI is the primary name every new file, folder and message takes, and the others are legacy
    names read indefinitely, so a workspace, host or job generated under an older name keeps
    working and is never renamed behind its owner.

    package: the import and distribution name, and the one stem every release has answered to,
        so a command line or marker file another machine's release must read spells it.
    names: every script name, primary first.
    """

    package: str = _PACKAGE
    names: tuple[str, ...] = _names()

    @property
    def name(self) -> str:
        """The primary name: the command a user types and the stem of everything new."""
        return self.names[0]

    @property
    def manifests(self) -> tuple[str, ...]:
        """The workspace manifest filenames, primary first."""
        return tuple(f"{name}.toml" for name in self.names)

    @property
    def out_dirs(self) -> tuple[str, ...]:
        """The generated-state directory names a workspace root may hold, primary first."""
        return tuple(f".{name}" for name in self.names)

    @property
    def locks(self) -> tuple[str, ...]:
        """The committed lock filenames a workspace root may hold, primary first."""
        return tuple(f"{name}.lock" for name in self.names)

    @property
    def jobs_roots(self) -> tuple[str, ...]:
        """Where a dispatch target keeps the tool's code and state unless its profile says
        otherwise, primary first: one dedicated folder under the login home, never a human
        checkout. `dispatch.targets.resolve` decides which one a host uses."""
        return tuple(f"~/.{name}-jobs" for name in self.names)

    @property
    def plugin_groups(self) -> tuple[str, ...]:
        """The entry-point groups third-party providers advertise under, primary first."""
        return tuple(f"{name}.providers" for name in self.names)

    def table[T](self, tool: Mapping[str, T]) -> T | None:
        """This tool's table in a pyproject's `[tool]`: the primary name's, else a legacy one's."""
        return next((tool[name] for name in self.names if name in tool), None)

    def variable(self, key: str) -> Variable:
        """The environment variable `<NAME>_<key>` under every name, `MB_SOURCE` first."""
        return Variable(names=tuple(f"{name.upper()}_{key}" for name in self.names))

    def marker(self, suffix: str) -> str:
        """The name a cross-machine marker file `.<name>-<suffix>` is written under.

        Deliberately the legacy stem: center and hosts may run different releases, and a marker
        an older release cannot find reads as missing (an unpinned tree, an unbuilt prefix), so
        writers keep the name every release reads until no host runs one without `markers`.
        """
        return f".{self.package}-{suffix}"

    def markers(self, suffix: str) -> tuple[str, ...]:
        """Every name a marker file `.<name>-<suffix>` is read under, primary first."""
        return tuple(f".{name}-{suffix}" for name in self.names)

    def marked(self, directory: Path, suffix: str) -> Path:
        """The marker file `.<name>-<suffix>` in `directory` under the first name it exists by,
        else where `marker` writes it."""
        found = (directory / name for name in self.markers(suffix))
        return next((path for path in found if path.exists()), directory / self.marker(suffix))

    def manifest(self, directory: Path) -> Path:
        """The manifest `directory` holds, or where a new one goes (the primary name).

        Raises MissionError when `directory` holds more than one, since which is meant is the
        owner's call.
        """
        return self._declared(directory) or directory / self.manifests[0]

    def lock(self, root: Path | None = None) -> Path:
        """The committed lock at workspace `root` (the cwd's workspace), or where a new one goes.

        The first name one already exists under, so a lock is never renamed behind its owner;
        with none, the primary name even beside a legacy manifest. The state directory follows
        its manifest only because it is ignored, so a clone or mirror must derive its name from
        a file that travels; the lock is tracked and travels itself, and no older release reads
        it under any name, so nothing argues for the legacy one.

        Raises MissionError when `root` holds more than one.
        """
        here = root or self.workspace()
        found = self._one(here, self.locks, "record one workspace's solves twice")
        return found or here / self.locks[0]

    def out_dir(self, root: Path | None = None) -> str:
        """The generated-state directory's name at workspace `root` (the cwd's workspace).

        The first of the tool's names whose directory already exists there, so a workspace
        generated under a legacy name keeps it and nothing is renamed. With none yet, the name
        its manifest is spelled with, so a clone or a host's mirror of a legacy workspace lands
        where its origin keeps state and a workspace born under the primary name gets that.

        Workspace-relative, so the same string names it on the center and on a host's mirror. A
        path on a host is spelled with the cwd's answer, the workspace this process serves and
        so the one the host mirrors; only a path on this machine passes its own `root`.
        """
        here = root or self.workspace()
        for name in self.out_dirs:
            if (here / name).is_dir():
                return name
        declared = self._declared(here)
        return f".{declared.stem}" if declared else self.out_dirs[0]

    def out(self, root: Path | None = None) -> Path:
        """The generated-state directory itself, under `root` or the cwd's workspace."""
        here = root or self.workspace()
        return here / self.out_dir(here)

    def activation(self, env: str = "default", root: Path | None = None) -> str:
        """The activation script for `env`, relative to workspace `root` (the cwd's workspace).

        One per environment, since a shared file would activate whichever was provisioned last.
        The default keeps the bare `activate.sh` that onboarded hosts and hand-written job
        scripts already source.
        """
        suffix = "" if env == "default" else f"-{env}"
        return f"{self.out_dir(root)}/activate{suffix}.sh"

    def find_root(self, start: Path) -> Path:
        """The workspace `start` lies in: the nearest manifest upward, or the one composing it.

        Inside a member, the ancestor whose `[workspace] members` claims that member is the
        workspace, the way cargo finds its workspace root, so a member's tasks are reachable
        from its own directory; cloned alone, the member's manifest is the nearest and only one.
        """
        nearest = self._nearest(start)
        for ancestor in nearest.parents:
            declared = self._declared(ancestor)
            if declared and Membership.declared(declared, self.manifests).claims(nearest):
                return ancestor
        return nearest

    def workspace(self, start: Path | None = None) -> Path:
        """The workspace `start` (default the cwd) belongs to, or `start` itself outside any.

        Generated state belongs to the workspace, not the directory a command was typed in, and
        a scratch tree under no manifest keeps its own rather than raising.
        """
        here = start or Path.cwd()
        try:
            return self.find_root(here)
        except FileNotFoundError:
            return here

    def _nearest(self, start: Path) -> Path:
        """The nearest directory at or above `start` holding a manifest."""
        for directory in (start, *start.parents):
            if self._declared(directory):
                return directory
        raise FileNotFoundError(
            f"no {' or '.join(self.manifests)} found from {start} upward; run inside a workspace"
        )

    def _declared(self, directory: Path) -> Path | None:
        """The one manifest `directory` holds under any name, None when it holds none."""
        return self._one(directory, self.manifests, "declare one workspace twice")

    @staticmethod
    def _one(directory: Path, names: tuple[str, ...], why: str) -> Path | None:
        """The one file `directory` holds under any of `names`, None when it holds none.

        why: what holding several would mean, for the refusal.
        """
        found = [directory / name for name in names if (directory / name).is_file()]
        if len(found) > 1:
            raise MissionError(
                f"{directory} holds both {' and '.join(path.name for path in found)}, which "
                f"{why}; keep {found[0].name} and delete the other"
            )
        return found[0] if found else None
