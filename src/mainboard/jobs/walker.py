"""The static import walk over a job's first-party roots, read and never run."""

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError
from .targets.target import dotted

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# An initializer, which makes its directory a regular package.
_INIT = "__init__.py"

# The dynamic import spellings whose literal argument the walk still follows.
_DYNAMIC = ("import_module", "__import__")


class Module(FrozenModel):
    """One first-party module file the walk reached, and its import root, workspace-relative."""

    path: str
    root: str


class Walker:
    """The static import walk over a job's first-party roots.

    home: the job's own import root, workspace-relative, searched first.
    distributions: the roots the manifest installs editable, workspace-relative, in order.
    """

    @staticmethod
    def __regular(package: Path) -> bool:
        return (package / _INIT).is_file() or package.with_suffix(".py").is_file()

    @staticmethod
    def _absolute(package: str, *, level: int, name: str) -> str | None:
        """`name` imported from module `package` with `level` leading dots.

        `name` is empty for `from . import x`, and the result is None past the top package.
        """
        if not level:
            return name
        parts = package.split(".")[:-1]
        if level - 1 >= len(parts):
            return None
        base = parts[: len(parts) - (level - 1)]
        return ".".join([*base, name] if name else base)

    @staticmethod
    def _dynamic(call: ast.Call) -> str:
        """The literal module `import_module("a.b")` or `__import__("a.b")` names, else empty."""
        func = call.func
        named = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if named not in _DYNAMIC or not call.args:
            return ""
        literal = call.args[0]
        if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
            return ""
        return "" if literal.value.startswith(".") else literal.value

    @staticmethod
    def _named(node: ast.AST, *, package: str) -> Iterator[str]:
        """The module names one syntax node of module `package` imports.

        An import names its modules, a from-import its base and each name under it, and a
        dynamic import its literal argument.
        """
        match node:
            case ast.Import(names=names):
                yield from (alias.name for alias in names)
            case ast.ImportFrom(module=module, level=level, names=names):
                base = Walker._absolute(package, level=level, name=module or "")
                if base is not None:
                    yield base
                    yield from (f"{base}.{alias.name}" for alias in names if alias.name != "*")
            case ast.Call() if literal := Walker._dynamic(node):
                yield literal

    def __chain(self, base: Path, parts: Sequence[str]) -> list[Path] | None:
        """The package files `parts` walks through under `base`, None when the chain breaks."""
        found: list[Path] = []
        for depth in range(1, len(parts) + 1):
            package = base.joinpath(*parts[:depth])
            if (package / _INIT).is_file():
                found.append(package / _INIT)
            elif depth == len(parts) and package.with_suffix(".py").is_file():
                found.append(package.with_suffix(".py"))
            elif not package.is_dir():
                return None
        return found

    def __init__(self, root: Path, *, home: str, distributions: Sequence[str]) -> None:
        self.root = root
        self.home = home
        self.roots = list(dict.fromkeys([home, *distributions]))

    def imported(self, module: Module) -> Iterator[str]:
        """Every module name `module` imports, relative ones spelled out, in syntax order."""
        tree = ast.parse((self.root / module.path).read_text(encoding="utf-8"))
        package = dotted(self.root / module.path, home=self.root / module.root)
        # `_absolute` drops the importing module's last component, but an initializer imports
        # relative to itself, so it keeps one to drop.
        if Path(module.path).name == _INIT:
            package = f"{package}.__init__"
        for node in ast.walk(tree):
            yield from Walker._named(node, package=package)

    def reach(self, file: str) -> list[Module]:
        """Every first-party module `file` imports, transitively, `file` itself first."""
        first = Module(path=file, root=self.home)
        reached: dict[str, Module] = {file: first}
        pending = [first]
        while pending:
            fresh = {
                found.path: found
                for name in self.imported(pending.pop())
                for found in self.resolve(name)
                if found.path not in reached
            }
            reached.update(fresh)
            pending.extend(fresh.values())
        return list(reached.values())

    def resolve(self, name: str) -> list[Module]:
        """The files `name` imports, parent packages included, under the one root holding it.

        Two roots holding one top-level package is refused rather than settled by order (two
        campaigns each keep an `experiments` package, and importing the other's would run code
        the receipts never named). Namespace portions, which roots legitimately share, are walked
        wherever they are.
        """
        parts = name.split(".")
        holders = [root for root in self.roots if self.__regular(self.root / root / parts[0])]
        if len(holders) > 1:
            raise MissionError(
                f"`{parts[0]}` is a package under more than one import root "
                f"({', '.join(holders)}); a job's imports have to name one, so keep the package "
                "in one root or import it by the root that owns it"
            )
        for root in self.roots:
            chain = self.__chain(self.root / root, parts)
            if chain is not None:
                return [
                    Module(path=file.relative_to(self.root).as_posix(), root=root)
                    for file in chain
                ]
        return []
