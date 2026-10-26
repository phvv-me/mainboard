"""Objective-C implementation inventory from kept project paths, without claiming an AST."""

from pathlib import Path

from ...catalog import Family
from ...engine.facts import Facts
from .project.core import ProjectTree


class AppleSource(Family):
    family = "AppleSource"
    relations = {
        "facts": """
            SELECT 'source:' || entry AS fact_id, entry AS path,
                   CASE WHEN ends_with(lower(entry), '.mm') THEN 'objective_cpp'
                        ELSE 'objective_c' END AS language,
                   1 AS start_line, 0 AS start_column
            FROM project_entries WHERE NOT is_directory
              AND (ends_with(lower(entry), '.m') OR ends_with(lower(entry), '.mm'))
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        ProjectTree(root=root).register(facts)
