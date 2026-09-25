from typing import TYPE_CHECKING

from patos import FrozenModel

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


def candidates(*, ecosystem: str, env: str, dev: bool) -> tuple[Slot, ...]:
    """Every table a requirement of this shape may live in, the preferred one first.

    Dev requirements have two house spellings (`[dev.python.deps]`, `[nodejs.dev]`); the caller
    takes the first the manifest already carries, so an edit lands beside its neighbours.

    env: an environment name, the whole manifest when empty.
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
    """Every requirement the manifest declares, by table; an empty table is left out."""
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
