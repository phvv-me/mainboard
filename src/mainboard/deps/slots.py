from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..manifest import Manifest, Scope

# The default resolver's own name. Its requirements sit directly under a scope, while every
# other ecosystem hides behind a table named after the runtime package that installs it.
_CONDA = "conda"

# The two table names a scope reaches its requirements through, runtime first.
_DEPS = "deps"
_DEV = "dev"


class Slot(FrozenModel):
    """One dependency table in the manifest, addressed by the key path that reaches it.

    path: the table's key path, `("dev", "python", "deps")` for `[dev.python.deps]`.
    ecosystem: whose resolver reads the table, `conda` for the manifest's default one.
    """

    path: tuple[str, ...]
    ecosystem: str

    @property
    def table(self) -> str:
        """The table heading a reader of the manifest would look for."""
        return f"[{'.'.join(self.path)}]"


def candidates(*, ecosystem: str, env: str, dev: bool) -> tuple[Slot, ...]:
    """Every table a requirement of this shape may live in, the preferred one first.

    A manifest spells a development-only requirement two ways and both are declared house
    style, `[dev.python.deps]` beside the conda `[dev.deps]` and `[nodejs.dev]` beside the
    runtime table it belongs to. Rather than pick one and rewrite the other, the caller takes
    the first of these that the manifest already carries and falls back to the first listed, so
    an edit lands where its neighbours already are.

    ecosystem: the resolver the requirement belongs to.
    env: an environment name, the whole manifest when empty.
    dev: whether the requirement is development-only.
    """
    base = ("envs", env) if env else ()
    if ecosystem == _CONDA:
        if not dev:
            return (Slot(path=(*base, _DEPS), ecosystem=ecosystem),)
        if env:
            raise MissionError(
                f"environment {env!r} has no conda development table. Declare the requirement "
                f"in [envs.{env}.deps], or drop --env to reach the workspace-wide [dev.deps]."
            )
        return (Slot(path=(_DEV, _DEPS), ecosystem=ecosystem),)
    if not dev:
        return (Slot(path=(*base, ecosystem, _DEPS), ecosystem=ecosystem),)
    if env:
        return (Slot(path=(*base, ecosystem, _DEV), ecosystem=ecosystem),)
    return (
        Slot(path=(_DEV, ecosystem, _DEPS), ecosystem=ecosystem),
        Slot(path=(ecosystem, _DEV), ecosystem=ecosystem),
    )


def declared(manifest: Manifest) -> dict[Slot, tuple[str, ...]]:
    """Every requirement the manifest declares, by the table declaring it.

    The schema already discovers which tables are dependency tables, `deps` on a scope and the
    runtime-named ecosystems riding in its extras, so the walk here only says which scopes
    exist and lets each one name its own. A table declaring nothing is left out, which is what
    makes membership here mean the manifest really carries that table.

    manifest: the validated workspace manifest.
    """
    found: dict[Slot, tuple[str, ...]] = {}
    for path, scope in _scopes(manifest):
        if scope.deps:
            found[Slot(path=(*path, _DEPS), ecosystem=_CONDA)] = tuple(scope.deps)
        for ecosystem, chain in scope.toolchains().items():
            if chain.deps:
                found[Slot(path=(*path, ecosystem, _DEPS), ecosystem=ecosystem)] = tuple(
                    chain.deps
                )
            if chain.dev:
                found[Slot(path=(*path, ecosystem, _DEV), ecosystem=ecosystem)] = tuple(chain.dev)
    return found


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
