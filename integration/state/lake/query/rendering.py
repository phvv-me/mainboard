import re

from patos import FrozenModel

from mainboard.render import Stamp

from .footer import Footer

_DISPLAYED = re.compile(r"displayed=(\d+)")


class Rendering(FrozenModel):
    """What `Query.render` printed, and the stamp it answered."""

    out: str
    err: str
    stamp: Stamp

    @property
    def body(self) -> str:
        """Everything printed but the footer line."""
        return self.out if self.err else self.out[: self.out.rindex("\n# ") + 1]

    @property
    def displayed(self) -> int:
        """How many rows the footer says were displayed."""
        match = _DISPLAYED.search(self.footed)
        assert match is not None, self.footed
        return int(match[1])

    @property
    def footed(self) -> str:
        """The stream holding the footer: stderr beside a JSON array, else stdout."""
        return self.err or self.out

    @property
    def footer(self) -> Footer:
        return Footer.of(self.footed)
