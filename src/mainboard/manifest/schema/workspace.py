from patos import Model


class Header(Model):
    """Workspace identity and solve surface.

    scripts: workspace-relative shell scripts pixi sources after the dotenv loader on every
        entry, for setup a static table cannot express (a library path across installed wheels).
    members: workspace-relative globs of the projects composed into this workspace (see
        `manifest.members`), each `!pattern` leaving out what it matches.
    """

    name: str
    version: str = "0.1.0"
    platforms: list[str] = []
    channels: list[str] = ["conda-forge"]
    dotenv: bool = True
    scripts: list[str] = []
    members: list[str] = []
