import os
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from patos import FrozenModel
from pydantic import Field

from ..core.errors import MissionError
from ..core.execution.shell import become
from ..core.project import Project
from ..engines.compile.provisioner import Provisioner
from ..fleet.board import Board
from ..manifest.loading import load
from ..state.storage.holds import holding

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..engines.compile.backend import Pixi


class Session(FrozenModel):
    """What every verb of one CLI run reads: the workspace root and the project naming the tool.

    root: an explicit workspace root, discovered from the cwd when None.
    """

    root: Path | None = None
    project: Project = Field(default_factory=Project)

    def board(self, on: str) -> Board:
        return Board(self.workspace_root(), host=on)

    def environments(self) -> list[str]:
        """Every environment the workspace declares, `default` first."""
        return ["default", *load(self.project.manifest(self.workspace_root())).envs]

    def installed(self, env: str) -> Pixi:
        """The pixi holding `env` here, refused when `env` was never installed on this machine."""
        root = self.workspace_root()
        pixi = Provisioner(root, load(self.project.manifest(root))).pixi_for(env)
        if not pixi.ready(env):
            raise MissionError(
                f"environment {env!r} is not installed here; run `mb install {env}`"
            )
        return pixi

    def pixi_verb(self, verb: str, arguments: Sequence[str], *, env: str) -> NoReturn:
        """Hand this process to pixi's `verb` over the installed `env`, frozen to its lock."""
        pixi = self.installed(env)
        argv = [str(pixi.executable), verb, "--frozen", "-e", env, *pixi.scope(), *arguments]
        become(argv[0], argv, env={**os.environ, **pixi.overrides})

    def ssh_hosts(self) -> list[str]:
        manifest = holding(self.project.manifest(self.workspace_root()))
        return ["local", *(alias for alias, spec in manifest.hosts.items() if spec.kind == "ssh")]

    def workspace_root(self) -> Path:
        return self.root or self.project.find_root(Path.cwd())
