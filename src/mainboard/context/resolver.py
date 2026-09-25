from ..core.errors import MissionError
from ..manifest.schema.container import Container
from ..manifest.schema.root import Manifest
from .plan import ExecutionPlan


class Resolver:
    """Turns the manifest plus a host alias into one concrete `ExecutionPlan`."""

    def __init__(self, manifest: Manifest) -> None:
        """manifest: the loaded workspace manifest plans resolve against."""
        self.manifest = manifest

    def plan(self, host: str = "local", *, env: str = "", container: str = "") -> ExecutionPlan:
        """The execution plan for `host` (`local` for this machine), overrides beating the profile.

        container: a container name, `none` forcing bare.
        """
        profile = self.manifest.profile(host)
        chosen_env = env or profile.env
        self.manifest.environment(chosen_env)
        return ExecutionPlan(
            host=host,
            profile=profile,
            env=chosen_env,
            container=self._container(
                "" if container == "none" else container or profile.container
            ),
            vars={**self.manifest.vars, **profile.vars},
            exports=profile.exports,
        )

    def _container(self, name: str) -> Container | None:
        if not name:
            return None
        try:
            return self.manifest.containers[name]
        except KeyError:
            declared = sorted(self.manifest.containers)
            raise MissionError(
                f"no container {name!r}; declared containers are {declared}"
            ) from None
