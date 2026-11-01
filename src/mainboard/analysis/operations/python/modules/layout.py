# Moving modules: files renamed to new paths, every import of them in the project following.

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import ConfigDict, ValidationError, field_validator

from ....edits import Create, Edit, Remove, Rename
from ...base import Operation
from .relocation import ModuleRelocation

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

# What a module's file name adds to its dotted name: the suffix, and `__init__` for a package.
_SUFFIX = re.compile(r"(/__init__)?\.pyi?$")


class ModuleMove(FrozenModel):
    """One file moving from `path` to `target`."""

    path: str
    target: str

    @property
    def python(self) -> bool:
        """Whether the file is a module, or the stub of one."""
        return self.path.endswith((".py", ".pyi"))

    def landing(self, module: str) -> str | None:
        """The dotted name the file takes at `target`, given `module` as its name at `path`.

        None when `target` lies outside the source root `module` is named from.
        """
        old, new = _SUFFIX.sub("", self.path), _SUFFIX.sub("", self.target)
        named = module.replace(".", "/")
        if old != named and not old.endswith(f"/{named}"):
            return None
        root = old.removesuffix(named)
        return new.removeprefix(root).replace("/", ".") if new.startswith(root) else None


class ModuleMoveRow(FrozenModel):
    """The columns of a `move-modules` fix row; the others the engine adds are left alone.

    moves: the files that move.
    files: the modules a move touches (`module_readers`), each a record of `path`, `module` and
        `content`.
    held: the paths files already stand at in the folders they land in.
    inits: the `__init__.py` paths of the folders the moved modules leave.
    mentions: the files that name a moved module or path: no Python, or an import the move
        leaves dangling.
    removed: the paths that go with the move, such as the folder it leaves holding nothing
        but an empty `__init__.py`.
    """

    model_config = ConfigDict(extra="ignore")

    moves: list[ModuleMove]
    files: list[dict[str, str]]
    held: list[str] = []
    inits: list[str] = []
    mentions: list[str] = []
    removed: list[str] = []

    @field_validator("held", "inits", "mentions", "removed", mode="before")
    @classmethod
    def _listed(cls, value: list[str] | None) -> list[str]:
        """A list SQL aggregated over no row is NULL; read it as empty."""
        return value or []


class MoveModules(Operation):
    """Move modules to new paths, every import of them following.

    The imports of a moved module are respelled in the module itself, in each module importing it
    and in each reading it as `a.b.NAME` through a plain `import a`. A folder a module lands in
    gets the `__init__.py` its old folder had. The move is whole or refused, with the reason: a
    file landing on another, a name that is no module of the project, an import the move cannot
    follow, or a file that is no Python naming a moved module or path. The paths the row
    removes go with it, after the files have moved.
    """

    @staticmethod
    def _clashes(moves: Sequence[ModuleMove], held: Collection[str]) -> bool:
        """Whether two moves share a target, or one lands where a file stands or another starts."""
        targets = [move.target for move in moves]
        taken = {*held, *(move.path for move in moves)}
        return len(set(targets)) < len(targets) or bool(taken.intersection(targets))

    @staticmethod
    def _created(row: ModuleMoveRow) -> list[Edit]:
        """The `__init__.py` each folder a module lands in gets when the folder it left had one."""
        standing = {*row.held, *(move.target for move in row.moves)}
        initializer = {
            str(PurePosixPath(move.target).with_name("__init__.py")): str(
                PurePosixPath(move.path).with_name("__init__.py")
            )
            for move in row.moves
            if move.python
        }
        return [
            Create(path=new, text="")
            for new, old in initializer.items()
            if old in row.inits and old != new and new not in standing
        ]

    def _asked(self, row: Mapping) -> ModuleMoveRow | None:
        """The row's columns, None when it is malformed or a non-Python file names a move."""
        try:
            asked = ModuleMoveRow(**row)
        except ValidationError as error:
            self.refused("the row's " + ", ".join(str(each["loc"][0]) for each in error.errors()))
            return None
        if asked.mentions:
            self.refused(f"`{asked.mentions[0]}` names a moved module or path")
            return None
        return asked

    def _modules(self, asked: ModuleMoveRow) -> dict[str, str] | None:
        """Each moved module's old dotted name mapped to its new one, None when a move clashes.

        A move clashes when two land on one path, one lands where a file stands or another move
        starts, or a module is none the facts hold or lands outside its source root.
        """
        if MoveModules._clashes(asked.moves, asked.held):
            self.refused("a file would land on a path where another stands, starts or lands")
            return None
        named = {file["path"]: file["module"] for file in asked.files}
        modules: dict[str, str] = {}
        for move in (each for each in asked.moves if each.python):
            old = named.get(move.path, "")
            if not old or (new := move.landing(old)) is None:
                self.refused(
                    f"`{move.path}` is no module of the project or leaves its source root"
                )
                return None
            modules[old] = new
        return modules

    def edits(self, row: Mapping) -> list[Edit] | None:
        if (asked := self._asked(row)) is None or (modules := self._modules(asked)) is None:
            return None
        if (respelled := ModuleRelocation.respelled(asked.files, modules)) is None:
            self.refused(
                "an import of a moved module cannot be respelled, or a file does not parse"
            )
            return None
        renames = [Rename(path=move.path, target=move.target) for move in asked.moves]
        removals = [Remove(path=each) for each in asked.removed]
        return [*respelled, *renames, *removals, *MoveModules._created(asked)]
