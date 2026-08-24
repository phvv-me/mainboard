from typing import Self

from patos import FlexModel
from pydantic import ConfigDict

from .spec import Spec
from .toolchain import Toolchain


class Scope(FlexModel):
    """A dependency-carrying unit: the root manifest, an overlay, or an env.

    `deps` is the conda table; every other table riding in the extras whose
    value parses as a `Toolchain` is an ecosystem keyed by its runtime package
    name (`[python.deps]`, `[nodejs.deps]`), discovered rather than enumerated
    so a new ecosystem never edits this schema.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    deps: dict[str, Spec] = {}

    def merged(self, over: Self) -> Self:
        """This scope layered over `over`: conda deps and each ecosystem merge.

        over: the lower-precedence scope being overlaid.
        """
        deps = dict(over.deps)
        for name, spec in self.deps.items():
            deps[name] = spec.merged(deps[name]) if name in deps else spec
        chains = over.toolchains()
        merged_chains: dict[str, Toolchain] = dict(chains)
        for name, chain in self.toolchains().items():
            merged_chains[name] = chain.merged(chains[name]) if name in chains else chain
        landed = {
            name: chain.model_dump(exclude_defaults=True) for name, chain in merged_chains.items()
        }
        plain = {
            key: value
            for key, value in {**(over.model_extra or {}), **(self.model_extra or {})}.items()
            if key not in merged_chains
        }
        return type(self).model_validate({"deps": deps, **landed, **plain})

    def path_deps(self) -> dict[str, Spec]:
        """Every local path requirement across conda and all ecosystems."""
        found = {name: spec for name, spec in self.deps.items() if spec.is_path}
        for chain in self.toolchains().values():
            found |= {name: spec for name, spec in chain.all_deps().items() if spec.is_path}
        return found

    def toolchains(self) -> dict[str, Toolchain]:
        """Every ecosystem table this scope carries, by runtime name."""
        found: dict[str, Toolchain] = {}
        for name, value in (self.model_extra or {}).items():
            if isinstance(value, dict) and ("deps" in value or "dev" in value):
                found[name] = Toolchain.model_validate(value)
        return found
