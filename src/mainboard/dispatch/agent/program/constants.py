"""What both ends of a transfer agree on: chunk size, the kinds of walked paths, and the names of
a pinned tree's control files."""

# How much of a file is read or copied at a time, so no file ever has to fit in memory.
CHUNK = 1 << 20

# The tool's names, primary first, and the legacy one every release has named its files by.
# Spelled here because the agent imports nothing beyond the standard library; `core.project`
# derives the same names from pyproject.toml, and a test holds the two equal.
NAMES = ("mb", "mainboard")
LEGACY = "mainboard"

# A pinned tree's own control files, which no closure may ship and no live path may replace. A
# tree is read by whichever release a host runs, so each is written under the legacy name every
# release reads (`Project.marker` says why) and recognized under any.
STAMP = f".{LEGACY}-source"
CLOSURE = f".{LEGACY}-closure"
WRAPPERS = f".{LEGACY}-jobs"
RESERVED = tuple(f".{name}-{suffix}" for suffix in ("source", "closure", "jobs") for name in NAMES)

# Where a pin's record lands under the sources directory; `core.retention` reads it back.
PINS = ".retention"

# What one walked path is: a regular file, a directory, or a link carried as a link.
FILE, DIRECTORY, LINK = "f", "d", "l"
