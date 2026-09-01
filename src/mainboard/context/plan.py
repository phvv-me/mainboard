from patos import FrozenModel

from ..manifest.schema.container import Container
from ..manifest.schema.host import HostProfile


class ExecutionPlan(FrozenModel):
    """The resolved answer to where and how one command runs.

    The object the three predecessor tools each held a third of: which host
    alias, which resolved profile, which environment, which container base
    (None when bare), and the variables in force. Engines materialize it into
    argv, dispatch ships it, probe validates it against reality. `stable_id`
    from the base is the cache key for anything derived from a plan.
    """

    host: str
    profile: HostProfile
    env: str
    container: Container | None = None
    vars: dict[str, str] = {}
    exports: dict[str, str] = {}

    @property
    def containerized(self) -> bool:
        """Whether the command runs inside a container base image."""
        return self.container is not None

    def prefix(self, root: str) -> str:
        """The environment prefix path under `root`, always a bound host path.

        The prefix lives outside any image, on the workspace's generated
        directory locally and on the synced root remotely, which is what lets
        a fixed off-the-shelf image serve every dependency change.

        root: the workspace root path on the executing machine.
        """
        return f"{root}/.mainboard/envs/{self.env}/.pixi/envs/{self.env}"
