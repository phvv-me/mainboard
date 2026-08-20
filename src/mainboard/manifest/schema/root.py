from typing import ClassVar

from pydantic import model_validator

from ...core.errors import MissionError
from .container import Container
from .environment import Env, Task
from .gate import Gate
from .host import HostProfile
from .scope import Scope
from .template import Template
from .workspace import Header

_DEFAULTS_KEY = "defaults"
_RESERVED_ENVS = frozenset({"default", "dev"})


class Manifest(Scope):
    """The validated workspace manifest: deps, envs, containers, hosts, one file.

    The root is itself a `Scope`, so `[deps]` and the ecosystem tables sit at
    the top level exactly as before, with `[vars]` feeding interpolation,
    `[containers.*]` declaring base images, and `[hosts.*]` carrying per-host
    execution profiles that inherit `[hosts.defaults]`.

    Two tables carry no dependency at all and exist so a workspace can hand its
    own decisions to the verbs that would otherwise have to guess them:
    `[gates.*]` names the commands `doctor` asks for a verdict, and
    `[templates.*]` names the project templates `new` renders.
    """

    # The tables no compile reads. `[gates]` is what `doctor` asks and `[templates]` is what
    # `new` renders, so neither reaches a generated file and editing one must not make every
    # installed environment stale.
    uncompiled: ClassVar[frozenset[str]] = frozenset({"gates", "templates"})

    workspace: Header
    vars: dict[str, str] = {}
    system: dict[str, str] = {}
    on: dict[str, Scope] = {}
    dev: Scope = Scope()
    envs: dict[str, Env] = {}
    env: dict[str, str] = {}
    tasks: dict[str, Task] = {}
    gates: dict[str, Gate] = {}
    templates: dict[str, Template] = {}
    containers: dict[str, Container] = {}
    hosts: dict[str, HostProfile] = {}

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
        """Reserved env names stay free, and host references point at real tables."""
        taken = _RESERVED_ENVS & self.envs.keys()
        if taken:
            raise ValueError(f"reserved environment names declared: {sorted(taken)}")
        for alias, profile in self.profiles().items():
            if profile.container and profile.container not in self.containers:
                raise ValueError(
                    f"host {alias!r} names container {profile.container!r}, "
                    f"declared containers are {sorted(self.containers)}"
                )
            if profile.env != "default" and profile.env not in self.envs:
                raise ValueError(
                    f"host {alias!r} names environment {profile.env!r}, "
                    f"declared environments are {sorted(self.envs)}"
                )
        return self

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
