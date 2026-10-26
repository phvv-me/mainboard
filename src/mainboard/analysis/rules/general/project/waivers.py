"""Rules over suppression waivers (mcmr's ALL-WAIV0001)."""

from ....catalog import Function, Rule

_SPACE = " \t\n\r\x0b\x0c"


class ProjectDaysSince(Function):
    """The days from a `YYYY-MM-DD` date to today in UTC, negative for a future one, NULL when
    the text is no date."""

    written: str

    expression = r"""
        CASE WHEN regexp_matches(written, '^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
             THEN date_diff('day', TRY_CAST(written AS DATE),
                            CAST(now() AT TIME ZONE 'UTC' AS DATE))
        END"""


class WaiverDebt(Rule, id="ALL-WAIV0001"):
    """Files holding waivers that are undated, stale, expired, overly broad or lack metadata.

    A waiver is a Python comment holding `# noqa`, `# type: ignore`, `# pyrefly: ignore` or
    `# ty: ignore` (the first of these, in that order, that it contains). It justifies itself in
    `field=value` pairs after the marker: each value runs to the next field, and the `since=` and
    `expires=` dates (`YYYY-MM-DD`, counted against today in UTC) give its age and the days it
    has left."""

    required_metadata: tuple[str, ...] = ("reason",)
    maximum_age_days: int = 90

    query = (
        r"""
        WITH marked AS (
          SELECT t.file_id, t.path, t.text_id AS node, t.text, m.marker
          FROM core_text t
          CROSS JOIN LATERAL (
            SELECT marker FROM (VALUES ('# noqa', 1), ('# type: ignore', 2),
                                       ('# pyrefly: ignore', 3), ('# ty: ignore', 4))
                                AS markers(marker, priority)
            WHERE contains(t.text, marker) ORDER BY priority LIMIT 1) m
          WHERE t.kind = 'comment' AND t.language = 'python'
        ), comments AS (
          SELECT *, substr(text, strpos(text, marker) + length(marker)) AS tail,
                 contains(text, marker || chr(10)) OR ends_with(text, marker) AS is_overly_broad
          FROM marked
        ), tokens AS (
          SELECT c.file_id, c.node, g.position, c.tail_words[g.position] AS token
          FROM (SELECT *, list_filter(string_split_regex(tail, '\s+'), lambda w: w <> '')
                          AS tail_words
                FROM comments) c,
               generate_series(1, len(c.tail_words)) AS g(position)
        ), starts AS (
          SELECT *, regexp_extract(token, '^(\pL+)=', 1) AS name,
                 sum(regexp_matches(token, '^\pL+=')::INT)
                   OVER (PARTITION BY file_id, node ORDER BY position) AS field
          FROM tokens
        ), fields AS (
          SELECT file_id, node, arg_max(name, (name <> '')::INT) AS name, field,
                 trimmed(string_agg(CASE WHEN name <> '' THEN substr(token, length(name) + 2)
                                         ELSE token END, ' ' ORDER BY position)) AS value
          FROM starts WHERE field > 0
          GROUP BY file_id, node, field
        ), waivers AS (
          SELECT c.file_id, c.path, c.node, c.is_overly_broad,
                 project_days_since(s.value) AS age_days,
                 -project_days_since(e.value) AS expires_in_days
          FROM comments c
          LEFT JOIN (SELECT file_id, node, arg_max(value, field) AS value
                     FROM fields WHERE name = 'since' GROUP BY ALL) s
            ON s.file_id = c.file_id AND s.node = c.node
          LEFT JOIN (SELECT file_id, node, arg_max(value, field) AS value
                     FROM fields WHERE name = 'expires' GROUP BY ALL) e
            ON e.file_id = c.file_id AND e.node = c.node
        ), stated AS (
          SELECT file_id, node, count(DISTINCT name) AS metadata_count
          FROM (SELECT file_id, node, name, arg_max(value, field) AS value
                FROM fields GROUP BY file_id, node, name)
          WHERE list_contains($required_metadata::VARCHAR[], name)
            AND trim(value, '"""
        + _SPACE
        + r"""') <> ''
          GROUP BY file_id, node
        )
        SELECT w.path, 1 AS start_line, 0 AS start_column,
               'waiverfact:' || w.path AS fact_id, count(*) AS value
        FROM waivers w LEFT JOIN stated s ON s.file_id = w.file_id AND s.node = w.node
        WHERE w.age_days IS NULL OR w.age_days > $maximum_age_days
           OR (w.expires_in_days IS NOT NULL AND w.expires_in_days < 0)
           OR w.is_overly_broad OR coalesce(s.metadata_count, 0) < len(list_distinct($required_metadata::VARCHAR[]))
        GROUP BY w.path
    """
    )
    message = "waiver debt is {value} for `{fact_id}`"
