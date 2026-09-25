from patos import Model
from pydantic import Field

# What a workspace holds as data rather than source, in the sync filter language: never sealed or
# archived as a local trial's source, and pinned through `needs` or `resources` by what reads it.
DATA = ("/datasets/", "/references/", "**/evidence/")


class Header(Model):
    """Workspace identity and solve surface.

    scripts: workspace-relative shell scripts pixi sources after the dotenv loader on every
        entry, for setup a static table cannot express (a library path across installed wheels).
    data: the paths that are data, not source; a host still ships what its sync include names.
        Left out of the compile digest, since no environment reads it.
    members: workspace-relative globs of the projects composed into this workspace (see
        `manifest.members`), each `!pattern` leaving out what it matches.
    """

    name: str
    version: str = "0.1.0"
    platforms: list[str] = []
    channels: list[str] = ["conda-forge"]
    dotenv: bool = True
    scripts: list[str] = []
    data: list[str] = Field(default=list(DATA), exclude=True)
    members: list[str] = []
