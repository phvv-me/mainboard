from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, Field

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..manifest import Manifest, Scope

# The default resolver, whose requirements sit directly under a scope rather than a named table.
_CONDA = "conda"
_DEPS = "deps"
_DEV = "dev"


class Slot(FrozenModel):
    """One dependency table in the manifest, addressed by the key path that reaches it.

    path: `("dev", "python", "deps")` for `[dev.python.deps]`.
    """

    path: tuple[str, ...]
    ecosystem: str

    @property
    def table(self) -> str:
        return f"[{'.'.join(self.path)}]"


def declared(manifest: Manifest) -> dict[Slot, tuple[str, ...]]:
    """Every requirement the manifest declares, by table; an empty table is left out."""
    tables = (table for path, scope in _scopes(manifest) for table in _tables(path, scope))
    return {slot: names for slot, names in tables if names}


def _scopes(manifest: Manifest) -> Iterator[tuple[tuple[str, ...], Scope]]:
    """Every dependency-carrying scope in the manifest, with the key path that reaches it."""
    yield (), manifest
    yield (_DEV,), manifest.dev
    for platform, overlay in manifest.on.items():
        yield ("on", platform), overlay
    for name, env in manifest.envs.items():
        yield ("envs", name), env
        for platform, overlay in env.on.items():
            yield ("envs", name, "on", platform), overlay


def _tables(path: tuple[str, ...], scope: Scope) -> Iterator[tuple[Slot, tuple[str, ...]]]:
    """Each requirement table `scope` carries at `path`, the empty ones too."""
    yield Slot(path=(*path, _DEPS), ecosystem=_CONDA), tuple(scope.deps)
    for ecosystem, chain in scope.toolchains().items():
        yield Slot(path=(*path, ecosystem, _DEPS), ecosystem=ecosystem), tuple(chain.deps)
        yield Slot(path=(*path, ecosystem, _DEV), ecosystem=ecosystem), tuple(chain.dev)


@Parameter(name="*")
class Tables(FrozenModel):
    """Which manifest tables a requirement edit names or searches; all of them unless narrowed."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    ecosystem: Annotated[str, Parameter(name=["--lang", "-l"])] = ""
    """only this ecosystem's tables (`python`, `conda`, `nodejs`, ...); a new requirement is
    declared for conda when empty."""
    env: str = ""
    """only this environment's tables, the workspace-wide ones when empty."""
    dev: bool = False
    """only the development-only tables."""

    @property
    def is_narrowed(self) -> bool:
        """Whether a flag narrows the search to some tables."""
        return bool(self.ecosystem or self.env or self.dev)

    def _conda_dev(self) -> Slot:
        """The one conda development table, which no environment has."""
        if self.env:
            raise MissionError(
                f"environment {self.env!r} has no conda development table. Declare the "
                f"requirement in [envs.{self.env}.deps], or drop --env to reach the "
                "workspace-wide [dev.deps]."
            )
        return Slot(path=(_DEV, _DEPS), ecosystem=_CONDA)

    def candidates(self) -> tuple[Slot, ...]:
        """Every table these flags name, the preferred one first.

        Dev requirements have two house spellings (`[dev.python.deps]`, `[nodejs.dev]`); the
        caller takes the first the manifest already carries, so an edit lands beside its
        neighbours.
        """
        ecosystem = self.ecosystem or _CONDA
        scope = ("envs", self.env) if self.env else ()
        resolver = () if ecosystem == _CONDA else (ecosystem,)
        if not self.dev:
            return (Slot(path=(*scope, *resolver, _DEPS), ecosystem=ecosystem),)
        if ecosystem == _CONDA:
            return (self._conda_dev(),)
        if self.env:
            return (Slot(path=(*scope, ecosystem, _DEV), ecosystem=ecosystem),)
        return (
            Slot(path=(_DEV, ecosystem, _DEPS), ecosystem=ecosystem),
            Slot(path=(ecosystem, _DEV), ecosystem=ecosystem),
        )


@Parameter(name="*")
class Relock(FrozenModel):
    """What a requirement edit does to the lock: solve and install, solve alone, or leave it."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    no_install: bool = False
    """re-lock without installing."""
    frozen: bool = False
    """edit the manifest alone, leaving the lock as it stands, to solve several edits at once."""


@Parameter(name="*")
class Edit(FrozenModel):
    """One requirement edit's reach: which tables it touches, and what it then does to the lock."""

    tables: Tables = Field(default_factory=Tables)
    relock: Relock = Field(default_factory=Relock)
