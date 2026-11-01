"""Objective-C implementation inventory from kept project paths, without claiming an AST."""

from .project.core import TreeFamily


class AppleSource(TreeFamily):
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
