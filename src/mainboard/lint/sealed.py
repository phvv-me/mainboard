import re
from functools import cache
from pathlib import Path

# A research node registers its claim by sealing the bytes of its sources: its `node.md` front
# matter carries the digest, and every file beside it is among the sources, so rewriting one (a
# formatter, a missing final newline) breaks the seal and turns the node's receipts into
# evidence of a different instrument. An all-zero digest is the placeholder before registration.
_SEAL = re.compile(r"^registration_sha256:\s*(?P<digest>[0-9a-f]{64})\s*$", re.MULTILINE)
_FENCE = "---"


class Sealed:
    """Which files a registration has sealed, so no lint step rewrites them.

    A file is sealed when it lies in or beneath a directory whose `node.md` carries a real
    registration digest. A source a node seals from outside its own directory is not found this
    way; the workspace's `[lint] exclude` names those.

    root: the workspace root, where the search up from a file stops.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._registered = cache(self._registers)

    def holds(self, path: Path) -> bool:
        """Whether `path` is sealed by a registered node above it."""
        return any(
            self._registered(directory)
            for directory in (path.parent, *path.parent.parents)
            if directory.is_relative_to(self.root)
        )

    @staticmethod
    def _registers(directory: Path) -> bool:
        """Whether `directory`'s node declares a seal other than the placeholder."""
        try:
            lines = (directory / "node.md").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError, NotADirectoryError, UnicodeDecodeError:
            return False
        if not lines or lines[0].rstrip() != _FENCE:
            return False
        closing = next((i for i, line in enumerate(lines[1:], 1) if line.rstrip() == _FENCE), 0)
        found = _SEAL.search("\n".join(lines[1:closing]))
        return found is not None and set(found["digest"]) != {"0"}
