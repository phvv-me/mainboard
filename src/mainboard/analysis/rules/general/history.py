"""Rules over what a repository's commit history says about its files (mcmr's ALL-HIST family)."""

from ...catalog import Rule


class CoupledFilesThatNeverNameEachOther(Rule, id="ALL-HIST0003"):
    """Pairs that keep changing together while no import names the other, where others do. An
    import names a module by its file stem or by the class it takes from a package that re-exports
    it (`from ..typed import Struct` names `struct.py`); a package initializer changing with its
    own members is the package working as designed."""

    minimum_shared_commits: int = 5
    minimum_confidence: float = 0.5
    maximum_commit_files: int = 30

    query = r"""
        WITH paths AS (
          SELECT v.fact_id, v.parent_id AS change_id, v.string_value AS path
          FROM repository_history_values v JOIN repository_history_changes c
            ON c.fact_id = v.fact_id AND c.record_id = v.parent_id
          WHERE v.relation = 'changes.paths'
            AND c.paths_length + c.other_file_count <= $maximum_commit_files
        ), commits AS (
          SELECT fact_id, path, count(*) AS commits FROM paths GROUP BY ALL
        ), coupled AS (
          SELECT a.fact_id, a.path AS "left", b.path AS "right",
                 count(*) AS shared, any_value(l.commits) AS left_commits,
                 any_value(r.commits) AS right_commits
          FROM paths a
          JOIN paths b ON a.fact_id = b.fact_id AND a.change_id = b.change_id AND a.path < b.path
          JOIN commits l ON l.fact_id = a.fact_id AND l.path = a.path
          JOIN commits r ON r.fact_id = b.fact_id AND r.path = b.path
          WHERE NOT EXISTS (SELECT 1 FROM repository_history_files t WHERE t.is_test
                            AND t.fact_id = a.fact_id AND t.path IN (a.path, b.path))
            AND NOT (regexp_replace(a.path, '/[^/]*$', '') = regexp_replace(b.path, '/[^/]*$', '')
                     AND (ends_with(a.path, '/__init__.py') OR ends_with(b.path, '/__init__.py')))
          GROUP BY ALL
          HAVING shared >= $minimum_shared_commits
             AND shared >= $minimum_confidence * least(left_commits, right_commits)
        ), readers AS (
          SELECT *, CASE WHEN base IN ('__init__', 'mod', 'lib', 'index', 'main')
                         AND contains(peer, '/') THEN parent ELSE base END AS stem
          FROM (
            SELECT *, regexp_replace(regexp_replace(peer, '^.*/', '', 'g'), '\..*$', '', 'g') AS base,
                      regexp_replace(regexp_replace(peer, '/[^/]*$', '', 'g'), '^.*/', '', 'g') AS parent
            FROM (SELECT fact_id, "left", "right", "left" AS reader, "right" AS peer FROM coupled
                  UNION ALL
                  SELECT fact_id, "left", "right", "right", "left" FROM coupled))
        ), named AS (
          SELECT DISTINCT r.fact_id, r."left", r."right"
          FROM readers r
          JOIN repository_history_files f ON f.fact_id = r.fact_id AND f.path = r.reader
          JOIN repository_history_values i ON i.fact_id = f.fact_id AND i.parent_id = f.record_id
          WHERE i.relation = 'files.imports' AND contains(lower(i.string_value), lower(r.stem))
            AND regexp_matches(i.string_value, '(?i)(^|[^\p{L}\p{N}_])'
                || regexp_replace(r.stem, '([.+*?()|\[\]{}^$#&~-])', '\\\1', 'g')
                || '($|[^\p{L}\p{N}_])')
        )
        SELECT c.*, c."left" AS path, 1 AS start_line, 0 AS start_column,
               shared || CASE WHEN shared = 1 THEN ' focused commit' ELSE ' focused commits' END
                 AS shared_commits
        FROM coupled c ANTI JOIN named USING (fact_id, "left", "right")
        WHERE c.fact_id IN (SELECT fact_id FROM named)
    """
    message = (
        "`{left}` and `{right}` arrived together in {shared_commits} while neither names the "
        "other, out of the {left_commits} and {right_commits} each one saw"
    )


class LargeFileTheTeamKeepsReopening(Rule, id="ALL-HIST0001"):
    """Long files changed as often as a share of the busiest file, and recently."""

    maximum: int | None = None
    minimum_lines: int = 400
    minimum_commits: int = 2
    busy_share: float = 0.5
    stale_days: int = 180

    query = """
        WITH reopened AS (
          SELECT * FROM (
            SELECT *, author_count + additional_commit_count AS commits,
                   max(author_count + additional_commit_count) OVER (PARTITION BY fact_id)
                     AS busiest
            FROM repository_history_files)
          WHERE line_count >= $minimum_lines AND commits >= $minimum_commits
            AND commits >= busiest * $busy_share AND days_since_last_change <= $stale_days
        )
        SELECT path, 1 AS start_line, 0 AS start_column,
          '`' || path || '` runs ' || line_count
            || CASE WHEN line_count = 1 THEN ' line' ELSE ' lines' END
            || ' and took ' || commits || CASE WHEN commits = 1 THEN ' commit' ELSE ' commits' END
            || ' against the ' || busiest || ' the busiest file took, the last of them '
            || days_since_last_change
            || CASE WHEN days_since_last_change = 1 THEN ' day' ELSE ' days' END || ' ago'
            AS message
        FROM (SELECT *, count(*) OVER (PARTITION BY fact_id) AS reopened FROM reopened)
        WHERE reopened > $maximum
    """


class FileTooManyHandsHaveTouched(Rule, id="ALL-HIST0002"):
    """Files that many people changed across many commits."""

    minimum_authors: int = 4
    minimum_commits: int = 5

    query = """
        SELECT f.*, s.value FROM repository_history_facts f JOIN (
          SELECT fact_id, count(*) AS value FROM repository_history_files
          WHERE author_count >= $minimum_authors
            AND author_count + additional_commit_count >= $minimum_commits
          GROUP BY fact_id) s USING (fact_id)
    """
    message = "file too many hands have touched is {value} for `{fact_id}`"
