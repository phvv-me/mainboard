"""Rules over what comments say (mcmr's ALL-COMM content family)."""

from typing import Literal

from ...catalog import Relation, Rule

_SIZE = {"tokens": "token_count", "characters": "character_count", "lines": "line_count"}


class CommentsGroups(Relation):
    """The `comments_groups()` relation."""

    query = """
    SELECT g.*, coalesce(g.node_span_path, f.path) AS path,
           coalesce(g.node_span_start_line, f.start_line) AS start_line,
           coalesce(g.node_span_start_column, f.start_column) AS start_column
    FROM comment_groups g JOIN comment_facts f USING (fact_id)
"""


class CommentLength(Rule, id="ALL-COMM0001"):
    """The longest implementation comment of a file, when it takes too much of the maximum."""

    maximum: float = 20
    measure: Literal["tokens", "characters", "lines"] = "tokens"
    normalization_max: int = 200
    notice_markers: tuple[str, ...] = (
        "spdx-license-identifier",
        "copyright",
        "licensed under",
        "all rights reserved",
    )

    query = """
        SELECT path, start_line, start_column, largest, printf('%.4g', share) AS share_text,
               CASE WHEN largest = 1 THEN left($measure, length($measure) - 1)
                    ELSE $measure END AS noun
        FROM (
          SELECT path, start_line, start_column, {size} AS largest,
                 least({size} / $normalization_max * 100.0, 100.0) AS share
          FROM comments_groups()
          WHERE NOT is_documentation AND {size} IS NOT NULL
            AND len(list_filter($notice_markers::VARCHAR[],
                                lambda m: contains(lower(coalesce(node_text, '')), m))) = 0
          QUALIFY row_number() OVER (PARTITION BY fact_id ORDER BY {size} DESC, ordinal) = 1)
        WHERE share > $maximum
    """
    message = (
        "the largest comment group spans {largest} {noun}, which is {share_text} percent of the "
        "{normalization_max} {measure} used as the normalization maximum"
    )

    def sql(self) -> str:
        return self.query.replace("{size}", _SIZE[self.measure])
