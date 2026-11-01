from typing import ClassVar

from mainboard.analysis.catalog import Part


class Paths(Part):
    """One finding per path in `paths`."""

    query: ClassVar[str] = (
        "SELECT unnest($paths::VARCHAR[]) AS path, 1 AS start_line, 0 AS start_column"
    )
    message: ClassVar[str | None] = "kept"
    paths: tuple[str, ...] = ()
