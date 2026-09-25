from typing import TYPE_CHECKING, ClassVar

from pydantic import model_validator

from ...core.errors import MissionError
from .admission import Admission
from .container import Container
from .environment import Env, Task
from .figures.figure import FigureSpec
from .gate import Gate
from .host import HostProfile
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
    """The validated workspace manifest: deps, envs, containers, hosts, one file.

    The root is itself a `Scope`, so `[deps]` and the ecosystem tables sit at
    the top level exactly as before, with `[vars]` feeding interpolation,
    `[containers.*]` declaring base images, and `[hosts.*]` carrying per-host
    execution profiles that inherit `[hosts.defaults]`.

    These tables carry no dependency at all and let a workspace hand
    its own decisions to the verbs that would otherwise have to guess them:
    `[gates.*]` names the commands `doctor` asks for a verdict, `[templates.*]`
    names the project templates `new` renders, and `[tracking]` names where a
    batch's receipts are mirrored beyond this workspace's own files.
    `[papers.*]` names the manuscripts `paper` builds and checks against their page rule.
    `[plots.*]` names palette, theme, and output settings for result charts.
    `[admission.<card>]` says how idle a named card must be before a trial measures on it.

    `[env]` sets a variable to a string and clears one with `false`. Clearing
    is not the same as setting an empty string, which is what the table could
    say before: an empty `OMP_NUM_THREADS` is still defined, so `${VAR:-default}`
    and `[ -n "$VAR" ]` both behave differently from an unset one, and declaring
    a genuinely clean environment was therefore impossible from here and needed
    an `env -u` workaround downstream. `true` is refused rather than guessed at,
    since a variable is either given a value or taken away and there is no third
    thing it could mean.
    """

    # The tables no compile reads, the exact complement of what `PixiManifest.from_manifest`
    # and the second stage translate. `[gates]` is what `doctor` asks, `[templates]` is what
    # `new` renders, `[tracking]` is where a batch's receipts are mirrored, `[containers]` and
    # `[hosts]` are how a job reaches a machine, `[papers]` is what `paper` builds, `[plots]` is
    # how results are drawn, and `[vars]`
    # has already been folded into every string that quotes it by the time a manifest
    # validates, so a var a compiled table really uses moves the digest through that table's own
    # rendered value. None of them reaches a generated file, so editing one must not make every
    # installed environment stale. The classification is proved table by table against the
    # compiler's own output in `tests/engines/compile/test_compiler.py`, so a table added to
    # the schema is refused until somebody decides which side of this line it sits on.
    uncompiled: ClassVar[frozenset[str]] = frozenset(
        {
            "admission",
            "containers",
            "figures",
            "gates",
            "hosts",
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

    @model_validator(mode="after")
    def env_values_set_or_clear(self) -> Manifest:
        """Refuse `true` in `[env]`, since only `false` means anything there.

        A variable is either given a value or taken away, so a boolean has exactly one useful
        reading and guessing at the other one would hide a typo behind a plausible default.
        """
        wrong = sorted(name for name, value in self.env.items() if value is True)
        if wrong:
            raise ValueError(
                f"[env] {wrong[0]!r} is `true`, which says nothing; write a string to set it "
                "or `false` to clear whatever the machine inherited"
            )
        return self

    def environment(self, name: str) -> Env:
        """The named environment table, refusing unknown names with the roster.

        name: the environment name, `default` always allowed.
        """
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
        """Reserved env names stay free, and every host names a table that exists."""
        taken = _RESERVED_ENVS & self.envs.keys()
        if taken:
            raise ValueError(f"reserved environment names declared: {sorted(taken)}")
        for alias, profile in self.profiles().items():
            self._resolves(f"host {alias!r}", profile.container, profile.env)
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
        """This manifest with the machines the workspace is holding laid over `[hosts]`.

        held: the held machines' ssh profiles by alias.
        """
        return self.model_copy(update={"hosts": {**self.hosts, **held}}) if held else self

    def profile(self, alias: str) -> HostProfile:
        """The resolved profile for `alias`, defaults-only when undeclared.

        alias: the host name, an ssh-config alias or `local`.
        """
        profiles = self.profiles()
        if alias in profiles:
            return profiles[alias]
        return HostProfile().inheriting(self.hosts.get(_DEFAULTS_KEY, HostProfile()))

    def profiles(self) -> dict[str, HostProfile]:
        """Every concrete host profile with `[hosts.defaults]` already inherited."""
        base = self.hosts.get(_DEFAULTS_KEY, HostProfile())
        return {
            alias: profile.inheriting(base)
            for alias, profile in self.hosts.items()
            if alias != _DEFAULTS_KEY
        }
