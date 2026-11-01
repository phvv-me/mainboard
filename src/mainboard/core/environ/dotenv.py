import re
from pathlib import Path

from patos import FrozenModel

# A value wrapped whole in one matching pair of quotes, which is all the pair a value loses.
_QUOTED = re.compile(r"""^(["'])(.*)\1$""")


class Dotenv(FrozenModel):
    """The variables a workspace's `.env` defines.

    The file is data, never shell: `NAME=value` lines, blanks and `#` comments skipped, an
    `export ` prefix dropped, one matching pair of quotes off a value, nothing expanded or
    executed. A name defined twice takes its last line, as sourcing the file would. Whether the
    process environment or the file wins is the caller's call.

    values: each defined name with its value.
    """

    values: dict[str, str]

    @classmethod
    def at(cls, root: Path) -> Dotenv:
        """The `.env` in directory `root`, defining nothing when `root` holds none."""
        try:
            text = (root / ".env").read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls(values={})
        entries = (line.strip().removeprefix("export ").strip() for line in text.splitlines())
        pairs = (entry.partition("=") for entry in entries if not entry.startswith("#"))
        return cls(
            values={
                name: _QUOTED.sub(r"\2", value.strip())
                for key, sign, value in pairs
                if sign and (name := key.rstrip())
            }
        )
