import abc
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

if TYPE_CHECKING:
    from pathlib import Path

    from ....manifest.schema.spec import Spec
    from ....manifest.schema.toolchain import Toolchain
    from ..backend import Pixi
    from ..generated import Writer


class Ecosystem(Registry, abc.ABC):
    """One toolchain filled in after pixi has installed the conda environment.

    pixi owns conda and Python. Every other table a manifest declares (`[nodejs]`, `[rust]`,
    `[go]`) names a package manager that itself ships as a conda package, so it can only run
    once the environment exists, which is what makes these a second stage rather than more
    pixi dependency tables. An implementation binds to one merged table for one environment
    and answers three questions: what the compile has to generate before the manager can run,
    where the manager links its executables, and how to make the environment match the table.

    Concrete implementations enroll under this root, keyed by the manifest table they own
    through the `toolchain` class attribute.
    """

    toolchain: ClassVar[str] = ""

    # Whether this toolchain's generated files and install tree belong to the workspace rather
    # than to one environment. A manager that installs into the pixi prefix gets a fresh tree
    # per environment and reads only that environment's tables, while one writing into the
    # generated directory or the workspace root shares a single tree with every environment.
    # A shared toolchain therefore binds to the tables merged across the whole manifest, since
    # reading one environment's view would let installing that environment rewrite, or delete
    # outright, what another environment installed from.
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

        table: the ecosystem's table, already merged across every scope that applies.
        env: the environment being provisioned.
        project: the workspace's declared name, for a manager that needs one.
        workspace: the workspace root directory.
        out: the generated directory, where anything compiled for this toolchain lands.
        pixi: the backend owning the environment the managers run inside.
        """
        self.table = table
        self.env = env
        self.project = project
        self.workspace = workspace
        self.out = out
        self.pixi = pixi

    @property
    def deps(self) -> dict[str, Spec]:
        """Everything the table declares, runtime and dev requirements together."""
        return self.table.all_deps()

    def binary_dirs(self) -> tuple[Path, ...]:
        """Directories this toolchain links its executables into.

        None by default, which is the right answer for a manager installing straight into the
        pixi environment prefix, since activation already exports that prefix's `bin/`.
        """
        return ()

    def generate(self, files: Writer) -> None:
        """Write whatever this toolchain's installer reads, nothing by default.

        files: the generated-file writer, valid only while the sync lock is held.
        """

    @abc.abstractmethod
    def sync(self) -> None:
        """Make the environment carry exactly what the table declares, and nothing it dropped."""
