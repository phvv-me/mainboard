import shlex
from collections.abc import Mapping

from plumbum import local

from mainboard.dispatch.transport.shells import HostShell

from .constants import PLAN, ROOT


class Recorded(HostShell):
    """A host shell that reaches no host.

    It keeps each command and each file it is handed, and exits every command 0 but those of
    `absent`.

    absent: the commands that exit 1.
    replies: what a command ending in each key prints.
    """

    def __init__(self, *absent: str, replies: Mapping[str, str] | None = None) -> None:
        super().__init__(local, PLAN, ROOT)
        self.absent = absent
        self.replies = replies or {}
        self.commands: list[str] = []
        self.written: dict[str, str] = {}

    def execute(self, line: str) -> tuple[int, str, str]:
        command = line.removeprefix(self.stage("", activate=False))
        self.commands.append(command)
        out = next((out for suffix, out in self.replies.items() if command.endswith(suffix)), "")
        return int(command in self.absent), out, ""

    def installed(self, python: str) -> tuple[bool, bool]:
        """Whether uv was told to install `python` and then to install the tool on it."""
        tool = f"--python {shlex.quote(python)}"
        return (
            f"uv python install --upgrade --no-bin {shlex.quote(python)}" in self.commands,
            any("uv tool install" in command and tool in command for command in self.commands),
        )

    def reset(self, *absent: str) -> None:
        """Forget the commands run so far, and answer `absent` as missing from now on."""
        self.absent = absent
        self.commands.clear()

    def write(self, path: str, *, text: str) -> None:
        self.written[path] = text
