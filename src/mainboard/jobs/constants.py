"""The constants of `jobs` other modules share."""

from ..core.project import Project

# Set for each process a fresh-process lane runs a cell in: the lane declares the total and ends
# the session itself, so a child reports its one cell and nothing else.
NESTED = Project().variable("FRESH_CELL")
