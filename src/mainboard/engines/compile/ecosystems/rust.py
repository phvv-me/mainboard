import tomllib
from typing import TYPE_CHECKING, ClassVar

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .base import Ecosystem

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest.schema.spec import Spec

# cargo's own record of what it installed under a `--root`, the only place an installed crate
# and its version are written down.
_RECORD = ".crates.toml"

# Source keys a spec may carry, each spelled as the cargo flag of the same name.
_SOURCES = ("git", "path", "branch", "tag", "rev")


class Rust(Ecosystem):
    """The Rust toolchain: crates installed into the pixi environment prefix.

    Installing under the prefix means the crates share the environment's own `bin/`, so
    activation exports them with everything else and no extra directory reaches PATH. Every
    command runs as `pixi run cargo` pinned to the environment being synced, because cargo
    lives inside that environment rather than on the outer PATH, and a crate that links a
    conda library needs the environment's compiler and pkg-config settings to build at all.
    """

    toolchain: ClassVar[str] = "rust"

    @property
    def prefix(self) -> Path:
        """The environment prefix crates install under."""
        return self.pixi.env_prefix(self.env)

    @staticmethod
    def install_args(spec: Spec) -> list[str]:
        """The `cargo install` flags expressing `spec`: a version pin plus its source extras.

        `git`, `path`, `branch`, `tag` and `rev` ride through as the cargo flags of the same
        name, and `locked` as `--locked`, so `{ version = ">=0.1", locked = true }` installs
        exactly as declared.

        spec: one declared crate requirement.
        """
        extra = spec.model_extra or {}
        args = [] if spec.version == "*" else ["--version", spec.version]
        for key in _SOURCES:
            if value := extra.get(key):
                args += [f"--{key}", str(value)]
        if extra.get("locked"):
            args.append("--locked")
        return args

    @staticmethod
    def satisfied(constraint: str, installed: str) -> bool:
        """Whether an already installed crate version meets `constraint`.

        Anything cargo accepts but `packaging` cannot read (a caret spelling, a git or path
        source whose recorded version answers to no constraint) counts as satisfied, since
        reinstalling on every sync is worse than trusting cargo's own record.

        constraint: the declared version requirement.
        installed: the version cargo recorded for the installed crate.
        """
        if constraint == "*":
            return True
        try:
            return Version(installed) in SpecifierSet(constraint)
        except InvalidVersion, InvalidSpecifier:
            return True

    def cargo(self, verb: str, *args: str) -> None:
        """Run one cargo verb against the environment prefix, through the environment's cargo."""
        self.pixi("run", "cargo", verb, "--root", str(self.prefix), *args, environment=self.env)

    def installed(self) -> dict[str, str]:
        """Every crate cargo recorded under the prefix, by name and installed version."""
        try:
            record = (self.prefix / _RECORD).read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        # A `.crates.toml` key reads `"name version (source)"`, and an entry missing the
        # version is no usable install record, so it is skipped rather than indexed past.
        return {
            parts[0]: parts[1]
            for key in tomllib.loads(record).get("v1", {})
            if len(parts := key.split()) >= 2
        }

    def sync(self) -> None:
        """Install what is missing or drifted, and uninstall what the table no longer declares."""
        installed = self.installed()
        for name in sorted(installed.keys() - self.deps.keys()):
            self.cargo("uninstall", name)
        for name, spec in self.deps.items():
            current = installed.get(name)
            if current is not None and self.satisfied(spec.version, current):
                continue
            reinstall = ("--force",) if current is not None else ()
            self.cargo("install", *self.install_args(spec), *reinstall, name)
