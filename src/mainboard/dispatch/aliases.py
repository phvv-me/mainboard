# The ssh aliases a held rental is reached by, written into the user's own ssh config.
#
# A held machine has to answer to a name for the whole session, from this tool, from rsync and
# from a person typing `ssh <alias>`, and the one place all three look a name up is the ssh
# config. Each alias is one marked block, put first in the file so a broad `Host *` further down
# never overrides its address, and removed again by its markers when the machine is released.
# Nothing outside a block is ever touched.

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transport import Endpoint


class SshAliases:
    """The marked alias blocks in one ssh config file.

    path: the ssh config file, the user's own when None.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".ssh" / "config"

    def add(self, alias: str, endpoint: Endpoint) -> None:
        """Point `alias` at `endpoint`, replacing any block this tool wrote for it before.

        A rental's address is new and its host key will never be seen again, so the key is
        accepted on sight and kept out of `known_hosts`, as the dispatch itself connects.
        """
        lines = [
            f"Host {alias}",
            f"  HostName {endpoint.address}",
            *([f"  Port {endpoint.port}"] if endpoint.port else []),
            *([f"  User {endpoint.user}"] if endpoint.user else []),
            *(
                [f"  IdentityFile {endpoint.identity}", "  IdentitiesOnly yes"]
                if endpoint.identity
                else []
            ),
            "  StrictHostKeyChecking accept-new",
            f"  UserKnownHostsFile {os.devnull}",
            "  LogLevel ERROR",
        ]
        block = "\n".join([_opening(alias), *lines, _closing(alias)])
        self._write(f"{block}\n\n{self._without(alias)}")

    def remove(self, alias: str) -> None:
        """Take `alias`'s block out; a config that does not exist stays that way."""
        if self.path.is_file():
            self._write(self._without(alias))

    def _without(self, alias: str) -> str:
        """The config's text with `alias`'s block dropped and the blank lines around it trimmed."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        opening, closing = _opening(alias), _closing(alias)
        kept: list[str] = []
        inside = False
        for line in text.splitlines():
            if line == opening:
                inside = True
            elif line == closing:
                inside = False
            elif not inside:
                kept.append(line)
        body = "\n".join(kept).strip("\n")
        return f"{body}\n" if body else ""

    def _write(self, text: str) -> None:
        """Write the config with the owner-only mode ssh insists on."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.write_text(text, encoding="utf-8")
        self.path.chmod(0o600)


def _opening(alias: str) -> str:
    """The line that opens `alias`'s block."""
    return f"# >>> mainboard hold {alias} >>>"


def _closing(alias: str) -> str:
    """The line that closes `alias`'s block."""
    return f"# <<< mainboard hold {alias} <<<"
