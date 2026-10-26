"""The direct dependencies a project declares, and the version each resolves to here."""

import tomllib
from collections.abc import Iterable
from functools import cached_property
from importlib import metadata
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from patos import FrozenModel

from ....log import logger
from .models import Declaration


class Manifests(FrozenModel):
    """The direct dependencies a project declares and the version each resolves to locally."""

    root: Path

    @cached_property
    def locked(self) -> dict[str, list[Version]]:
        """The registry versions `uv.lock` holds, by canonical name."""
        found: dict[str, list[Version]] = {}
        for package in self._read("uv.lock").get("package", []):
            if "registry" in package.get("source", {}):
                found.setdefault(canonicalize_name(package["name"]), []).append(
                    Version(package["version"])
                )
        return found

    @staticmethod
    def _constraint(value: str | dict) -> str:
        constraint = value if isinstance(value, str) else value.get("version", "")
        return "" if constraint == "*" else constraint

    @staticmethod
    def _declared(requirements: Iterable[str], *, development: bool) -> list[Declaration]:
        found = []
        for text in requirements:
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                logger.warning("skipped an unreadable dependency", requirement=text)
                continue
            found.append(
                Declaration(
                    name=canonicalize_name(requirement.name),
                    requirement=str(requirement),
                    is_development=development,
                )
            )
        return found

    def _is_self(self, value: str | dict) -> bool:
        path = value.get("path") if isinstance(value, dict) else None
        return isinstance(path, str) and (self.root / path).resolve() == self.root.resolve()

    def _pyproject(self) -> list[Declaration]:
        project = self._read("pyproject.toml")
        runtime = project.get("project", {})
        return [
            *self._declared(runtime.get("dependencies", []), development=False),
            *[
                d
                for group in runtime.get("optional-dependencies", {}).values()
                for d in self._declared(group, development=True)
            ],
            *self._declared(project.get("build-system", {}).get("requires", []), development=True),
        ]

    def _read(self, name: str) -> dict:
        try:
            return tomllib.loads((self.root / name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _workspace(self) -> list[Declaration]:
        """The Python tables of `mainboard.toml`, minus entries installing this project itself."""
        manifest = self._read("mainboard.toml")
        tables = [
            (manifest.get("python", {}).get("deps", {}), False),
            (manifest.get("dev", {}).get("python", {}).get("deps", {}), True),
        ]
        return [
            declared
            for table, development in tables
            for declared in self._declared(
                [
                    f"{name}{self._constraint(value)}"
                    for name, value in table.items()
                    if not self._is_self(value)
                ],
                development=development,
            )
        ]

    def declarations(self) -> list[Declaration]:
        """Unique direct requirements by name, a runtime one winning over a development one."""
        found: dict[str, Declaration] = {}
        for declaration in [*self._pyproject(), *self._workspace()]:
            if declaration.name not in found or found[declaration.name].is_development:
                found[declaration.name] = declaration
        return [found[name] for name in sorted(found)]

    def resolve(self, declaration: Declaration) -> Version:
        """The greatest locked registry version the requirement allows, else the installed one.

        Raises LookupError with the reason when neither exists.
        """
        specifier = Requirement(declaration.requirement).specifier
        locked = self.locked.get(declaration.name, [])
        if allowed := [v for v in locked if specifier.contains(v, prereleases=True)]:
            return max(allowed)
        try:
            installed = Version(metadata.version(declaration.name))
        except metadata.PackageNotFoundError:
            raise LookupError("dependency is not installed") from None
        except InvalidVersion:
            raise LookupError("installed version is not valid PEP 440") from None
        if not specifier.contains(installed, prereleases=True):
            raise LookupError("installed version does not satisfy declaration")
        return installed
