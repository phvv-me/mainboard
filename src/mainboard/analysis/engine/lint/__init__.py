"""Checking and fixing one project with the SQL rules.

A check extracts the fact families the selected rules' SQL names, then runs every rule's query at
once on parallel cursors (DuckDB plans them concurrently and parallelizes each). A fix runs every
fixable rule's edit query the same way, groups the edit rows by the finding they repair and fixes
the project to its fixpoint (fixing.py). A test runs every rule's examples through the same check
and fix.
"""

from .core import Lint, load_catalog, selected
from .verbs import Selection, check, compare, fix, test

__all__ = [
    "Lint",
    "Selection",
    "check",
    "compare",
    "fix",
    "load_catalog",
    "selected",
    "test",
]
