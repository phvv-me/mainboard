from typing import TYPE_CHECKING, ClassVar

from pydantic import model_validator

from ...core.errors import MissionError
from .admission import Admission
from .ci import Ci
from .container import Container
from .environment import Env, Task
from .figures.figure import FigureSpec
from .gate import Gate
from .git import GitPolicy
from .host import HostProfile
from .lint import Lint
from .paper import Paper
from .plot import PlotStyle
from .scope import PlatformScope, Scope
from .template import Template
from .tracking import Tracking
from .workspace import Header

if TYPE_CHECKING:
    from collections.abc import Mapping

_DEFAULTS_KEY = "defaults"
_RESERVED_ENVS = frozenset({"default", "dev"})


class Manifest(Scope):
    """The validated workspace manifest, itself the root `Scope` (`[deps]`, ecosystem tables).

    Beside it: `[vars]` feed interpolation, `[containers.*]` declare base images, `[hosts.*]`
    are execution profiles inheriting `[hosts.defaults]`, `[gates.*]` are what `doctor` asks,
    `[templates.*]` what `new` renders, `[tracking]` where receipts are mirrored, `[papers.*]`
    what `paper` builds, `[plots.*]` how charts look, `[admission.<card>]` how idle a card must
    be before a trial, `[git]` whose repositories `git` may write and what never commits,
    `[lint]` what `lint` runs and leaves alone, `[ci]` the hosts `ci --matrix` runs on.

    `[env]` sets a variable with a string and clears one with `false`, which an empty string
    (still defined for `${VAR:-default}` and `[ -n "$VAR" ]`) cannot do; `true` is refused.
    """

    # The exact complement of what `PixiManifest.from_manifest` and the second stage translate
    # (`[vars]` is already folded into every string quoting it), so editing one never stales an
    # environment. `tests/engines/compile/test_compiler.py` proves the split table by table, so
    # a new schema table is refused until it is classified.
    uncompiled: ClassVar[frozenset[str]] = frozenset(
        {
            "admission",
            "ci",
            "containers",
            "figures",
            "gates",
            "git",
            "hosts",
            "lint",
            "papers",
            "plots",
            "templates",
            "tracking",
            "vars",
        }
    )

    workspace: Header
    vars: dict[str, str] = {}
    system: dict[str, str] = {}
    on: dict[str, PlatformScope] = {}
    dev: Scope = Scope()
    envs: dict[str, Env] = {}
    env: dict[str, str | bool] = {}
    tasks: dict[str, Task] = {}
    gates: dict[str, Gate] = {}
    templates: dict[str, Template] = {}
    tracking: Tracking = Tracking()
    containers: dict[str, Container] = {}
    hosts: dict[str, HostProfile] = {}
    admission: dict[str, Admission] = {}
    papers: dict[str, Paper] = {}
    plots: dict[str, PlotStyle] = {}
    figures: dict[str, FigureSpec] = {}
    git: GitPolicy = GitPolicy()
    lint: Lint = Lint()
    ci: Ci = Ci()

    @model_validator(mode="after")
    def env_values_set_or_clear(self) -> Manifest:
        """Refuse `true` in `[env]`, a typo rather than a third meaning."""
        wrong = sorted(name for name, value in self.env.items() if value is True)
        if wrong:
            raise ValueError(
                f"[env] {wrong[0]!r} is `true`, which says nothing; write a string to set it "
                "or `false` to clear whatever the machine inherited"
            )
        return self

    def environment(self, name: str) -> Env:
        """The named environment table (`default` always), refusing others with the roster."""
        if name == "default":
            return Env()
        try:
            return self.envs[name]
        except KeyError:
            raise MissionError(
                f"no environment {name!r}; declared environments are {sorted(self.envs)}"
            ) from None

    @model_validator(mode="after")
    def names_resolve(self) -> Manifest:
        """Reserved env names stay free, and every host or lint tool names a real table."""
        taken = _RESERVED_ENVS & self.envs.keys()
        if taken:
            raise ValueError(f"reserved environment names declared: {sorted(taken)}")
        for alias, profile in self.profiles().items():
            self._resolves(f"host {alias!r}", profile.container, profile.env)
        for name, tool in self.lint.tools.items():
            self._resolves(f"lint tool {name!r}", "", tool.env)
        return self

    def _resolves(self, subject: str, container: str, env: str) -> None:
        """Refuse `subject`'s container or environment when neither declared table holds it."""
        if container and container not in self.containers:
            raise ValueError(
                f"{subject} names container {container!r}, declared containers are "
                f"{sorted(self.containers)}"
            )
        if env != "default" and env not in self.envs:
            raise ValueError(
                f"{subject} names environment {env!r}, declared environments are "
                f"{sorted(self.envs)}"
            )

    def holding(self, held: Mapping[str, HostProfile]) -> Manifest:
        """This manifest with the held machines' ssh profiles laid over `[hosts]`."""
        return self.model_copy(update={"hosts": {**self.hosts, **held}}) if held else self

    @property
    def defaults(self) -> HostProfile:
        """`[hosts.defaults]`, what every host inherits and an undeclared one is."""
        return self.hosts.get(_DEFAULTS_KEY, HostProfile())

    def profile(self, alias: str) -> HostProfile:
        """The resolved profile for `alias` (ssh alias or `local`), defaults if undeclared."""
        return self.profiles().get(alias) or HostProfile().inheriting(self.defaults)

    def profiles(self) -> dict[str, HostProfile]:
        """Every concrete host profile with `[hosts.defaults]` already inherited."""
        return {
            alias: profile.inheriting(self.defaults)
            for alias, profile in self.hosts.items()
            if alias != _DEFAULTS_KEY
        }
