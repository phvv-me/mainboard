from typing import Self

from patos import FlexModel
from pydantic import ConfigDict, Field, model_validator

from .spec import (  # noqa: TC001  reason=Spec is a pydantic field type, resolved at class-build time, not annotation-only since=2026-08-17
    Json,
    Spec,
)


class Toolchain(FlexModel):
    """One ecosystem's dependency table, keyed by its runtime package name.

    `[python.deps]`, `[nodejs.deps]`, `[rust.deps]` and friends all validate
    here. Unknown keys pass through to the ecosystem's solver options untyped
    (`dependency-overrides`, `manager`, index configuration).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    deps: dict[str, Spec] = {}
    dev: dict[str, Spec] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def tables_only(cls, value: Json) -> Json:
        """Reject a bare string where an ecosystem table is required."""
        if isinstance(value, str):
            raise ValueError(
                "an ecosystem entry must be a table with a deps key, not a version string"
            )
        return value

    def all_deps(self) -> dict[str, Spec]:
        """Runtime and dev requirements as one view, dev winning on a name clash."""
        return {**self.deps, **self.dev}

    def merged(self, over: Self) -> Self:
        """This toolchain layered over `over`: deps merge per-name, extras override.

        over: the lower-precedence toolchain being overlaid.
        """
        deps = dict(over.deps)
        for name, spec in self.deps.items():
            deps[name] = spec.merged(deps[name]) if name in deps else spec
        dev = {**over.dev, **self.dev}
        extras = {**(over.model_extra or {}), **(self.model_extra or {})}
        return type(self).model_validate({"deps": deps, "dev": dev, **extras})
