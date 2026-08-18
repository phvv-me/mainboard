from patos import Model


class Header(Model):
    """Workspace identity and solve surface.

    scripts: shell scripts pixi sources on every entry into an environment, workspace-relative
        and running after the generated dotenv loader. It is the home for the setup a static
        table cannot express, a library path fanned out across installed wheels or a toolchain
        variable computed from what the environment actually holds.
    """

    name: str
    version: str = "0.1.0"
    platforms: list[str] = []
    channels: list[str] = ["conda-forge"]
    dotenv: bool = True
    scripts: list[str] = []
