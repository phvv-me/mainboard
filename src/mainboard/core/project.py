from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from patos import FrozenModel

from .environ import Variable
from .errors import MissionError, NoWorkspace
from .membership import Membership

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


class Marker(FrozenModel):
    """A cross-machine marker file `.<name>-<suffix>`, written under one name and read under all.

    suffix: what the file marks, the part of its name after the tool's.
    """

    suffix: str

    @property
    def names(self) -> tuple[str, ...]:
        """Every name the marker file is read under, primary first."""
        return tuple(f".{name}-{self.suffix}" for name in Project().names)

    @property
    def written(self) -> str:
        """The name the marker file is written under.

        Deliberately the legacy stem: center and hosts may run different releases, and a marker
        an older release cannot find reads as missing (an unpinned tree, an unbuilt prefix), so
        writers keep the name every release reads until no host runs one without `names`.
        """
        return f".{Project().package}-{self.suffix}"

    def at(self, directory: Path) -> Path:
        """The marker file in `directory`, under the first name it exists by.

        Otherwise it is where `written` names it.
        """
        found = (directory / name for name in self.names)
        return next((path for path in found if path.exists()), directory / self.written)


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
    def jobs_roots(self) -> tuple[str, ...]:
        """Where a dispatch target keeps the tool's code and state, primary first.

        That is one dedicated folder under the login home, never a human checkout, unless its
        profile says otherwise. `dispatch.targets.resolve` decides which one a host uses.
        """
        return tuple(f"~/.{name}-jobs" for name in self.names)

    @property
    def locks(self) -> tuple[str, ...]:
        """The committed lock filenames a workspace root may hold, primary first."""
        return tuple(f"{name}.lock" for name in self.names)

    @property
    def manifests(self) -> tuple[str, ...]:
        """The workspace manifest filenames, primary first."""
        return tuple(f"{name}.toml" for name in self.names)

    @property
    def name(self) -> str:
        """The primary name: the command a user types and the stem of everything new."""
        return self.names[0]

    @property
    def out_dirs(self) -> tuple[str, ...]:
        """The generated-state directory names a workspace root may hold, primary first."""
        return tuple(f".{name}" for name in self.names)

    @staticmethod
    def _one(directory: Path, names: Iterable[str], why: str) -> Path | None:
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

    def _declared(self, directory: Path) -> Path | None:
        """The one manifest `directory` holds under any name, None when it holds none."""
        return self._one(directory, self.manifests, "declare one workspace twice")

    def _nearest(self, start: Path) -> Path:
        """The nearest directory at or above `start` holding a manifest."""
        for directory in (start, *start.parents):
            if self._declared(directory):
                return directory
        raise NoWorkspace(
            f"no {' or '.join(self.manifests)} found from {start} upward; run inside a workspace"
        )

    def activation(self, env: str = "default", root: Path | None = None) -> str:
        """`env`'s activation script, relative to workspace `root` (the cwd's workspace).

        It lives in the environment's own directory beside the prefix it activates, as a
        content-addressed prefix keeps its own, so the state directory holds none;
        `mb shell-hook` names it for a shell to source.
        """
        return f"{self.out_dir(root)}/envs/{env}/activate.sh"

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

    def manifest(self, directory: Path) -> Path:
        """The manifest `directory` holds, or where a new one goes (the primary name).

        Raises MissionError when `directory` holds more than one, since which is meant is the
        owner's call.
        """
        return self._declared(directory) or directory / self.manifests[0]

    def marker(self, suffix: str) -> str:
        """The name a cross-machine marker file `.<name>-<suffix>` is written under (`Marker`)."""
        return Marker(suffix=suffix).written

    def out(self, root: Path | None = None) -> Path:
        """The generated-state directory itself, under `root` or the cwd's workspace."""
        here = root or self.workspace()
        return here / self.out_dir(here)

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

    def variable(self, key: str) -> Variable:
        """The environment variable `<NAME>_<key>` under every name, `MB_SOURCE` first."""
        return Variable(names=tuple(f"{name.upper()}_{key}" for name in self.names))

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
