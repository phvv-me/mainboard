from ...core.base import Declared


class Engine(Declared):
    """One declared serving engine: a command staged through a container, `serve`'s whole input.

    `serve` renders this into the exact launcher `run` already knows how to build for any
    command, through the containerize seam every other verb shares. No image is built here:
    the container's own `image` must already exist, declared like any other `[containers.*]`
    base, so a workspace that wants to serve a model brings the image and states the command
    that starts it.

    command: the shell command that starts serving, staged through this engine's environment.
    container: the `[containers.*]` table this engine runs inside, bare when empty.
    env: the environment the command runs under, `default` when empty.
    """

    command: str
    container: str = ""
    env: str = "default"
