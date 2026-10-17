"""mcmr's RepositoryHistoryFact, derived from git's own log.

The log (`history_git.GitLog`) names every path a commit touched as it was spelled that day, and
a rename carries both spellings. A path answers to the name its rename chain ends in; a commit
keeps only the source files the project still holds, so every claim names a file a reader can open
(a deleted or split-apart file is dropped, a commit left with no file counts for none).

`changes` holds one row per commit with `paths` (what it touched here) and `other_file_count`
(what it touched elsewhere), `files` one row per source file with its authors, commits, recency,
length and import lines, and `values` the same two lists as rows, for the rules that join on them.
"""

from pathlib import Path

from ...catalog import Family, Function
from ...facts import Facts
from .history_git import GitLog


class HistoryLineCount(Function):
    """How many lines a text has: its line breaks, and one more for an unterminated last line."""

    content: str

    expression = r"""
        length(content) - length(replace(content, chr(10), ''))
        + (content <> '' AND NOT ends_with(content, chr(10)))::INT
    """


class HistoryImports(Function):
    """The trimmed lines of a text that open with the words languages state a dependency in."""

    content: str

    expression = r"""
        list_filter(
          list_transform(string_split(content, chr(10)), lambda line: trimmed(line)),
          lambda line: regexp_matches(line, '^(import |from |use |pub use |#include|export )')
                       OR contains(line, 'require('))
    """


class RepositoryHistory(Family):
    family = "RepositoryHistoryFact"
    verified = True

    relations = {
        "facts": r"""
            SELECT 'history' AS fact_id, '' AS path, 1 AS start_line, 0 AS start_column,
                   1 AS end_line, 0 AS end_column, NULL::VARCHAR AS language
            WHERE EXISTS (SELECT 1 FROM git_commits)
        """,
        "changes": r"""
            WITH RECURSIVE renames AS (
              SELECT old_name AS name, arg_max(name, commit_order) AS renamed
              FROM git_entries WHERE old_name IS NOT NULL GROUP BY old_name
            ), reach(name, renamed) AS (
              SELECT name, renamed FROM renames
              UNION
              SELECT reach.name, renames.renamed
              FROM reach JOIN renames ON renames.name = reach.renamed
            ), current AS (
              SELECT name, renamed AS current FROM reach
              WHERE renamed NOT IN (SELECT name FROM renames)
            ), touched AS (
              SELECT DISTINCT e.commit_order, coalesce(c.current, e.name) AS name
              FROM git_entries e LEFT JOIN current c ON c.name = e.name
            )
            SELECT 'history' AS fact_id, 'history' AS parent_id,
                   'history/changes:' || t.commit_order AS record_id,
                   any_value(c.author) AS author, any_value(c.seconds) AS seconds,
                   list(t.name ORDER BY t.name) FILTER (WHERE f.path IS NOT NULL) AS paths,
                   count(f.path) AS paths_length, count(*) - count(f.path) AS other_file_count
            FROM touched t JOIN git_commits c USING (commit_order)
            LEFT JOIN git_files f ON f.path = t.name
            GROUP BY t.commit_order
            HAVING count(f.path) > 0
        """,
        "files": r"""
            WITH tally AS (
              SELECT path, count(DISTINCT author) AS author_count, count(*) AS commits,
                     max(seconds) AS last
              FROM (SELECT unnest(paths) AS path, author, seconds FROM repository_history_changes)
              GROUP BY path
            )
            SELECT 'history' AS fact_id, 'history' AS parent_id,
                   'history/files:' || t.path AS record_id, t.path, t.author_count,
                   t.commits - t.author_count AS additional_commit_count,
                   ((SELECT max(seconds) FROM git_commits) - t.last) // 86400
                     AS days_since_last_change,
                   coalesce(history_line_count(s.content), 0) AS line_count,
                   import_is_test_path(t.path) AS is_test,
                   coalesce(history_imports(s.content), []) AS imports
            FROM tally t LEFT JOIN source_files s ON s.path = t.path
        """,
        "values": r"""
            SELECT fact_id, record_id AS parent_id, 'changes.paths' AS relation,
                   unnest(paths) AS string_value
            FROM repository_history_changes
            UNION ALL
            SELECT fact_id, record_id, 'files.imports', unnest(imports)
            FROM repository_history_files
        """,
    }

    @classmethod
    def provide(cls, facts: Facts, root: Path) -> None:
        """Load the git tables the relations read."""
        GitLog(root=root).register(facts)
