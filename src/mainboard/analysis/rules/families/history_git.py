"""The project's git history as tables, the one input of the history families.

`git log` stays the source (renames detected, merges left out, paths relative to the project's
root); SQL splits its text, so the tables are plain relations: `git_commits` (newest first),
`git_entries` (what each commit touched, a rename naming both of its paths) and `git_files` (the
source files git lists now that some commit touched). A project outside git, or without a commit,
has none of them.
"""

import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs git with fixed argv since=2026-10-15
from pathlib import Path
from typing import ClassVar

from patos import FrozenModel

from ...facts import Facts

_LOG = [
    "log",
    "-M",
    "--relative",
    "--name-status",
    "--no-merges",
    "--format=%x01%H%x09%an%x09%ct",
]
_LISTED = ["ls-files", "-z", "--cached", "--others", "--exclude-standard"]

_TABLES = [
    r"""
    CREATE TEMP TABLE git_fields AS
    SELECT n, string_split(line, chr(9)) AS f,
           sum(starts_with(line, chr(1))::INT) OVER (ORDER BY n) - 1 AS commit_order
    FROM (SELECT generate_subscripts(lines, 1) AS n, unnest(lines) AS line
          FROM (SELECT string_split($log, chr(10)) AS lines))
    """,
    """
    CREATE TABLE git_commits AS
    SELECT commit_order, f[2] AS author, f[3]::BIGINT AS seconds
    FROM git_fields WHERE starts_with(f[1], chr(1))
    """,
    """
    CREATE TABLE git_entries AS
    SELECT commit_order, f[1] AS status, f[len(f)] AS name,
           CASE WHEN len(f) = 3 AND starts_with(f[1], 'R') THEN f[2] END AS old_name
    FROM git_fields
    WHERE NOT starts_with(f[1], chr(1)) AND len(f) >= 2 AND f[2] <> ''
    """,
    "DROP TABLE git_fields",
]


class GitLog(FrozenModel):
    """`root`'s history, registered by `register`."""

    root: Path
    seconds: ClassVar[int] = 600

    def register(self, facts: Facts) -> None:
        """Create `git_commits`, `git_entries` and `git_files` in `facts`."""
        try:
            log = self._git("-c", "core.quotePath=false", *_LOG)
            listed = self._git("-c", "core.quotePath=false", *_LISTED).split("\0")
        except subprocess.CalledProcessError:
            log, listed = "", []
        facts.db.execute(_TABLES[0], {"log": log})
        for statement in _TABLES[1:]:
            facts.db.execute(statement)
        suffixes = tuple(s for ends in Facts.SUFFIXES.values() for s in ends)
        touched = facts.db.execute(
            "SELECT DISTINCT name FROM git_entries "
            "WHERE name IN (SELECT unnest($listed::VARCHAR[]))",
            {"listed": [name for name in listed if name.lower().endswith(suffixes)]},
        ).fetchall()
        facts.db.execute(
            "CREATE TABLE git_files AS SELECT unnest($names::VARCHAR[]) AS path",
            {"names": [name for (name,) in touched if (self.root / name).is_file()]},
        )

    def _git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *arguments],
            capture_output=True,
            check=True,
            timeout=self.seconds,
        ).stdout.decode(errors="replace")
