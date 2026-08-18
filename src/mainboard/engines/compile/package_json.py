from typing import TYPE_CHECKING, Self

from patos import FlexModel
from pydantic import ConfigDict

from ...core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...manifest.schema.spec import Json, Spec


class PackageJson(FlexModel):
    """The compiled `package.json` a workspace's Node.js manager installs from.

    Extra keys ride through from `[nodejs.package]` (`type`, `engines`, `pnpm`), so an
    application controls its own manifest fields without mainboard hardcoding a framework.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    private: bool = True
    dependencies: dict[str, str] = {}

    @staticmethod
    def requirement(name: str, spec: Spec) -> str:
        """The npm version string for `spec`, refusing the source forms npm would misread.

        A `path`, `git` or `url` spec compiled to a bare `*` would install the registry
        package of the same name, so anything beyond a version fails fast instead.

        name: the package the requirement belongs to, named in the failure.
        spec: the declared requirement.
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

        name: the `name` field, which npm requires even for a private manifest.
        deps: requirements becoming `dependencies`.
        dev: requirements becoming `devDependencies`, omitted entirely when empty.
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
        """Render to `package.json` text."""
        return self.model_dump_json(indent=2) + "\n"
