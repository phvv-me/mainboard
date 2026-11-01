from pathlib import Path

from patos import Model
from pydantic import Field

from ....constants import DATA
from ....core.project import Project


class Header(Model):
    """Workspace identity and solve surface.

    name: the one spelling of the workspace in events, trailers and views (`Key.workspace`), the
        directory name when the manifest declares none; never read the directory for it.
    scripts: workspace-relative shell scripts pixi sources after the dotenv loader on every
        entry of a full environment, for setup a static table cannot express (a library path
        across installed wheels). Lean environments skip them, since they also set a global
        PYTHONPATH and load every `.env`; what every environment needs runs in the activation
        steps instead (`runtime.activation`, e.g. the cuDNN sublibrary links).
    data: the paths that are data, not source; a host still ships what its sync include names.
        Left out of the compile digest, since no environment reads it.
    members: workspace-relative globs of the projects composed into this workspace (see
        `manifest.members`), each `!pattern` leaving out what it matches.
    dotfiles: the owner's chezmoi dotfiles (`user/repo` or any repository `chezmoi init` takes),
        applied on every host `host setup` onboards and refreshed on each `host sync`, so every
        machine has the same shell, commands and editor. The repository carries its own mb.toml
        and mb.lock for the command-line toolbox, which its apply installs with `install
        --locked`. Left out of the compile digest.
    lake: the directory the center's state lake lives in, its catalog `lake.sqlite` beside
        its data `lake/`; `~` expands. Empty keeps it in the workspace's generated directory,
        as every dispatch mirror does (see `lake_home`). Left out of the compile digest, since
        no environment reads it.
    tmp: the directory large transient folders go in (collection staging, fix mirrors:
        `upkeep.disk.transient.Transient`), a big disk's when the workspace's is small; `~`
        expands.
        Empty keeps them in the generated directory's `tmp`. Read on the center only, as `lake`
        is, and left out of the compile digest.
    """

    name: str
    version: str = "0.1.0"
    platforms: list[str] = []
    channels: list[str] = ["conda-forge"]
    dotenv: bool = True
    scripts: list[str] = []
    data: list[str] = Field(default=list(DATA), exclude=True)
    members: list[str] = []
    dotfiles: str = Field(default="", exclude=True)
    lake: str = Field(default="", exclude=True)
    tmp: str = Field(default="", exclude=True)

    @staticmethod
    def _placed(setting: str, root: Path, *, otherwise: Path) -> Path:
        """`setting` expanded on the center, the checkout holding the workspace's repository.

        A dispatch mirror ships without `.git` and keeps `otherwise`, under its generated
        directory, since the center's path means nothing on the host.
        """
        return Path(setting).expanduser() if setting and (root / ".git").exists() else otherwise

    def lake_home(self, root: Path) -> Path:
        """Where the lake of the workspace at `root` lives."""
        return self._placed(self.lake, root, otherwise=Project().out(root))

    def tmp_home(self, root: Path) -> Path:
        """Where the large transient folders of the workspace at `root` go."""
        return self._placed(self.tmp, root, otherwise=Project().out(root) / "tmp")
