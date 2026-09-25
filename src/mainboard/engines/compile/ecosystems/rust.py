import tomllib
from typing import TYPE_CHECKING, ClassVar

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .base import Ecosystem

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest.schema.spec import Spec

# The cargo root, in the environment's generated directory rather than its prefix: conda packages
# built with cargo (dust) ship their own `.crates.toml` into the prefix, so a record kept there is
# shared, and diffing it once uninstalled a conda package's binary (2026-09-25).
_ROOT = "cargo"

# cargo's own record of what it installed under its root, crate names and versions.
_RECORD = ".crates.toml"

# Source keys a spec may carry, each spelled as the cargo flag of the same name.
_SOURCES = ("git", "path", "branch", "tag", "rev")


class Rust(Ecosystem):
    """The Rust toolchain: crates installed into a cargo root only this environment's installs use.

    Every crate that root's record names is one mainboard installed, so a dropped crate can be
    uninstalled without touching a file a conda package owns. cargo runs as `pixi run cargo`: it
    lives inside the environment, and a crate linking a conda library needs the environment's
    compiler and pkg-config settings to build.
    """

    toolchain: ClassVar[str] = "rust"

    @property
    def install_root(self) -> Path:
        return self.out / _ROOT

    @staticmethod
    def install_args(spec: Spec) -> list[str]:
        """The `cargo install` flags for `spec`: a version pin, source keys and `--locked`."""
        extra = spec.model_extra or {}
        args = [] if spec.version == "*" else ["--version", spec.version]
        for key in _SOURCES:
            if value := extra.get(key):
                args += [f"--{key}", str(value)]
        if extra.get("locked"):
            args.append("--locked")
        return args

    @staticmethod
    def satisfied(constraint: str, *, installed: str) -> bool:
        """Whether the version cargo recorded meets `constraint`.

        What `packaging` cannot read (a caret spelling, a git or path source) counts as
        satisfied, since reinstalling on every sync is worse than trusting cargo's record.
        """
        if constraint == "*":
            return True
        try:
            return Version(installed) in SpecifierSet(constraint)
        except InvalidVersion, InvalidSpecifier:
            return True

    def cargo(self, verb: str, *args: str) -> None:
        self.pixi(
            "run", "cargo", verb, "--root", str(self.install_root), *args, environment=self.env
        )

    def binary_dirs(self) -> tuple[Path, ...]:
        return (self.install_root / "bin",)

    def installed(self) -> dict[str, str]:
        """Every crate cargo recorded under the root, name to version."""
        try:
            record = (self.install_root / _RECORD).read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        # A key reads `"name version (source)"`; one missing the version is skipped.
        return {
            parts[0]: parts[1]
            for key in tomllib.loads(record).get("v1", {})
            if len(parts := key.split()) >= 2
        }

    def frozen_inputs(self) -> tuple[Path, ...]:
        """Exact registry releases with `--locked` install from their packaged Cargo.lock."""
        for spec in self.deps.values():
            extra = spec.model_extra or {}
            if not extra.get("locked") or any(extra.get(key) for key in _SOURCES):
                return super().frozen_inputs()
            try:
                version = Version(spec.version)
            except InvalidVersion:
                return super().frozen_inputs()
            if len(version.release) != 3:
                return super().frozen_inputs()
        return ()

    def sync(self, *, resolve: bool = False) -> None:
        """Install what is missing or drifted, and uninstall what the table no longer declares."""
        if not resolve:
            self.frozen_inputs()
        installed = self.installed()
        for name in sorted(installed.keys() - self.deps.keys()):
            self.cargo("uninstall", name)
        for name, spec in self.deps.items():
            current = installed.get(name)
            if (
                self.satisfied(spec.version, installed=current)
                if resolve and current is not None
                else current == spec.version
            ):
                continue
            self.cargo("install", *self.install_args(spec), name)
