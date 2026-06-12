from __future__ import annotations

import csv

from plumbum import CommandNotFound, ProcessExecutionError, local


def whoami_groups() -> tuple[str, ...]:
    """Windows group names from `whoami /groups`, or () when unavailable.

    Runs `whoami /groups /fo csv /nh` and takes the group name (first CSV column)
    from each row, so a tool can report Windows group membership the way `grp` does
    on POSIX. Returns () when `whoami` is missing or exits non-zero, keeping callers
    platform-tolerant.
    """
    try:
        output = local["whoami"]("/groups", "/fo", "csv", "/nh")
    except (CommandNotFound, OSError, ProcessExecutionError):
        return ()
    return tuple(row[0] for row in csv.reader(output.splitlines()) if row)
