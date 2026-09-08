import re
from typing import TYPE_CHECKING, ClassVar

from plumbum import local

from ....core.errors import MissionError
from .base import Ecosystem

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest.schema.spec import Spec

# Where `go install` links the executables this workspace declares, kept inside the generated
# directory so the workspace owns every binary in it and may prune the ones it stops declaring.
_GOBIN = ("go", "bin")

# A major-version suffix on a module path (`.../v2`), which names no executable of its own.
_MAJOR = re.compile(r"v[0-9]+")

# Characters that only appear in a version range, which Go resolves for nothing.
_RANGE = frozenset("<>=!~^,")
_PIN = re.compile(r"v?\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+incompatible)?|[0-9a-f]{40}")


class Go(Ecosystem):
    """The Go toolchain: modules installed as executables into a `GOBIN` this workspace owns.

    `go install module@version` builds in module-independent mode and drops one executable per
    module into `GOBIN`, so the whole backend is that command plus the directory it points at.
    Go resolves an exact version, a branch, a commit or `latest` and understands no ranges, so
    a range is refused where it is written rather than passed on as a reference no module proxy
    can resolve.
    """

    toolchain: ClassVar[str] = "go"
    # `GOBIN` is one directory inside the generated tree, shared by every environment, and
    # `sync` prunes whatever it no longer finds declared.
    shared: ClassVar[bool] = True

    @property
    def gobin(self) -> Path:
        """The generated directory `go install` links executables into."""
        return self.out.joinpath(*_GOBIN)

    @staticmethod
    def executable(module: str) -> str:
        """The executable name `go install` gives `module`.

        The last element of the module path, or the one before it when that element is a major
        version suffix, since `example.com/tool/v2` still installs as `tool`.

        module: the declared module path.
        """
        elements = module.rstrip("/").split("/")
        if len(elements) > 1 and _MAJOR.fullmatch(elements[-1]):
            return elements[-2]
        return elements[-1]

    @staticmethod
    def reference(module: str, spec: Spec) -> str:
        """The `module@version` argument `go install` takes for one declared requirement.

        An unconstrained requirement resolves as `latest`, a bare semver gains the `v` prefix
        Go requires, and anything else (a branch, a tag, a commit) rides through as written.

        module: the declared module path.
        spec: its declared requirement.
        """
        version = spec.version
        if version == "*":
            return f"{module}@latest"
        if _RANGE & set(version):
            raise MissionError(
                f"[go] dep `{module}` declares `{version}`, and go install resolves one exact "
                "version, branch, commit or `*` for latest, never a range."
            )
        numbered = version[0].isdigit() and not re.fullmatch(r"[0-9a-f]{40}", version)
        return f"{module}@{f'v{version}' if numbered else version}"

    def binary_dirs(self) -> tuple[Path, ...]:
        """Where installed modules land, since Go links them outside the environment prefix."""
        return (self.gobin,)

    def frozen_inputs(self) -> tuple[Path, ...]:
        """Exact module versions/revisions carry the published module's dependency graph."""
        if any(not _PIN.fullmatch(spec.version) for spec in self.deps.values()):
            return super().frozen_inputs()
        return ()

    def sync(self, *, resolve: bool = False) -> None:
        """Install every declared module, and unlink an executable the table no longer declares.

        Go writes no install record the way cargo does, so every declared module is installed
        on each sync rather than diffed first. Reading the version back out of each executable
        would cost one `go version` process per binary to skip an install the module cache
        already makes cheap.
        """
        if not resolve:
            self.frozen_inputs()
        declared = {self.executable(module) for module in self.deps}
        for installed in sorted(self.gobin.glob("*")):
            if installed.name not in declared:
                installed.unlink()
        if not self.deps:
            return
        self.gobin.mkdir(parents=True, exist_ok=True)
        with local.env(GOBIN=str(self.gobin)):
            for module, spec in self.deps.items():
                self.pixi(
                    "run", "go", "install", self.reference(module, spec), environment=self.env
                )
