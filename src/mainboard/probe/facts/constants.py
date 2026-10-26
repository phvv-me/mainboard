"""The constants of `facts` other modules share."""

from pathlib import Path

SYS_BLOCK = Path("/sys/block")

SKIP_PREFIXES = frozenset({"loop", "dm-", "sr", "ram", "zram", "fd"})
