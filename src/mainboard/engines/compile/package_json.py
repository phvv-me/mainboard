from typing import TYPE_CHECKING, Self

from patos import FlexModel
from pydantic import ConfigDict

from ...core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...manifest.schema.spec import Json, Spec


class PackageJson(FlexModel):
    """The compiled `package.json`, extra keys riding through from `[nodejs.package]`."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    private: bool = True
    dependencies: dict[str, str] = {}

    @staticmethod
    def requirement(name: str, spec: Spec) -> str:
        """The npm version string for `spec`.

        A `path`, `git` or `url` spec is refused, since as a bare `*` it would install the
        registry package of the same name.
        """
        extras = sorted(spec.model_extra or {})
        if extras:
            raise MissionError(
                f"[nodejs] dep `{name}` carries {', '.join(extras)}, which cannot be spelled "
                "in package.json. Pin a version here and put source overrides under "
                "[nodejs.package] instead."
            )
        return spec.version

    @classmethod
    def compiled(
        cls,
        *,
        name: str,
        deps: Mapping[str, Spec],
        dev: Mapping[str, Spec],
        fields: Mapping[str, Json],
    ) -> Self:
        """Build the manifest for one Node.js toolchain table.

        name: required by npm even for a private manifest.
        dev: becomes `devDependencies`, omitted entirely when empty.
        fields: `[nodejs.package]` entries merged verbatim over the generated ones.
        """
        body: dict[str, Json] = {
            "name": name,
            "dependencies": {pkg: cls.requirement(pkg, spec) for pkg, spec in deps.items()},
            **fields,
        }
        if dev:
            body["devDependencies"] = {
                pkg: cls.requirement(pkg, spec) for pkg, spec in dev.items()
            }
        return cls.model_validate(body)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"
