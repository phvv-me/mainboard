import os
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

from ....constants import RECEIPTS_VAR
from ....dispatch.jobs.evidence import framed

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class Receipts:
    """The file a run writes its trial receipts to, framed back through the job's output.

    Named for this runner's process, since jobs sharing a node's temporary directory would
    otherwise hand each other's trials over. `dispatch.evidence` says why receipts are a file.
    """

    def __init__(self) -> None:
        self.path = Path(gettempdir()) / f"mainboard-receipts.{os.getpid()}.ndjson"
        self.staged = False

    def frame(self) -> str:
        """The framed block of what the command wrote, empty when it wrote nothing."""
        if not self.staged:
            return ""
        try:
            written = self.path.read_bytes()
        except FileNotFoundError:
            return ""
        self.path.unlink()
        return framed(written) if written else ""

    def stage(self, environment: MutableMapping[str, str]) -> None:
        """Point the command at an empty receipts file, emptied for a restarted container."""
        self.path.write_bytes(b"")
        environment.update(RECEIPTS_VAR.exported(str(self.path)))
        self.staged = True
