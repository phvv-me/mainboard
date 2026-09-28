from patos import FrozenModel

from ..core.project import Project
from ..manifest.schema.container import Container
from ..manifest.schema.host import HostProfile


class ExecutionPlan(FrozenModel):
    """The resolved answer to where and how one command runs (`container` None when bare).

    Engines materialize it into argv, dispatch ships it, probe validates it against reality, and
    the base's `stable_id` keys any cache derived from a plan.
    """

    host: str
    profile: HostProfile
    env: str
    container: Container | None = None
    vars: dict[str, str] = {}
    exports: dict[str, str] = {}

    @property
    def containerized(self) -> bool:
        return self.container is not None

    def prefix(self, root: str) -> str:
        """The environment prefix under the executing machine's workspace `root`.

        Always a bound host path outside any image, which lets a fixed off-the-shelf image serve
        every dependency change. Its state directory is the name this workspace keeps, which a
        host's mirror of it keeps too.
        """
        return f"{root}/{Project().out_dir()}/envs/{self.env}/.pixi/envs/{self.env}"
