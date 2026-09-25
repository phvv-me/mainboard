import abc
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ....core import MissionError

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest.schema.spec import Spec
    from ....manifest.schema.toolchain import Toolchain
    from ..backend import Pixi
    from ..generated import Writer


class Ecosystem(Registry, abc.ABC):
    """One toolchain (`[nodejs]`, `[rust]`, `[go]`) filled in after pixi installs the environment.

    Its package manager ships as a conda package, so it runs only once the environment exists:
    a second stage, not more pixi tables. Implementations enroll keyed by their `toolchain` table.
    """

    toolchain: ClassVar[str] = ""

    # Whether the install tree belongs to the workspace (the generated directory or root) rather
    # than one environment's prefix. A shared toolchain binds to the tables merged across the
    # whole manifest, since one environment's view would let installing it rewrite or delete what
    # another environment installed from.
    shared: ClassVar[bool] = False

    def __init__(
        self,
        table: Toolchain,
        *,
        env: str,
        project: str,
        workspace: Path,
        out: Path,
        pixi: Pixi,
    ) -> None:
        """Bind one merged table to the environment and workspace it installs into.

        table: already merged across every scope that applies.
        project: the workspace's declared name.
        out: the generated directory.
        """
        self.table = table
        self.env = env
        self.project = project
        self.workspace = workspace
        self.out = out
        self.pixi = pixi

    @property
    def deps(self) -> dict[str, Spec]:
        """Runtime and dev requirements together."""
        return self.table.all_deps()

    @abc.abstractmethod
    def binary_dirs(self) -> tuple[Path, ...]:
        """Directories this toolchain links executables into, beyond the prefix's own `bin/`."""

    def generate(self, files: Writer) -> None:
        """Write what the installer reads; `files` is valid only under the sync lock."""

    def frozen_inputs(self) -> tuple[Path, ...]:
        """Inputs that permit installation without resolving versions, or a clear refusal."""
        if self.deps:
            raise MissionError(
                f"[{self.toolchain}] has no frozen installation contract; "
                "pin a supported native locked mode before remote setup or dispatch"
            )
        return ()

    @abc.abstractmethod
    def sync(self, *, resolve: bool = False) -> None:
        """Make the environment carry exactly what the table declares, and nothing it dropped."""
