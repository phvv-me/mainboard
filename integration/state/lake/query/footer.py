import re

from patos import FrozenModel

_FOOTER = re.compile(r"(\d+) rows.*sha256=([0-9a-f]{64}).*snapshot_id=(\d+)")


class Footer(FrozenModel):
    """The rows, hash and snapshot a query's footer line states."""

    rows: int
    sha256: str
    snapshot_id: int

    @classmethod
    def of(cls, text: str) -> Footer:
        """The footer inside `text`, which must hold one."""
        match = _FOOTER.search(text)
        assert match is not None, text
        return cls(rows=int(match[1]), sha256=match[2], snapshot_id=int(match[3]))
