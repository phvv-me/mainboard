from ..core.errors import MissionError
from ..manifest.schema.root import Manifest
from .plan import ExecutionPlan


class Resolver:
    """Turns the manifest plus a host alias into one concrete `ExecutionPlan`."""

    def __init__(self, manifest: Manifest) -> None:
        """manifest: the loaded workspace manifest plans resolve against."""
        self.manifest = manifest

    def plan(self, host: str = "local", *, env: str = "", container: str = "") -> ExecutionPlan:
        """The execution plan for `host`, overrides winning over the profile.

        host: the host alias, `local` for this machine.
        env: an environment name overriding the profile's choice.
        container: a container name overriding the profile's, `none` forcing bare.
        """
        profile = self.manifest.profile(host)
        chosen_env = env or profile.env
        self.manifest.environment(chosen_env)
        chosen_container = container or profile.container
        if container == "none":
            chosen_container = ""
        base = None
        if chosen_container:
            try:
                base = self.manifest.containers[chosen_container]
            except KeyError:
                raise MissionError(
                    f"no container {chosen_container!r}; declared containers are "
                    f"{sorted(self.manifest.containers)}"
                ) from None
        return ExecutionPlan(
            host=host,
            profile=profile,
            env=chosen_env,
            container=base,
            vars={**self.manifest.vars, **profile.vars},
            exports=profile.exports,
        )
