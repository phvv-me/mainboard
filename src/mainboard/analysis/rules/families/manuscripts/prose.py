"""mcmr's ProseSegmentFact, derived from sitting_duck's Python trees.

A segment is one Python file; its sections are the docstrings of the file's functions and classes
in source order. Each section keeps the word count of every sentence and paragraph and the
lowercased opener of every sentence, the measurements the writing-rhythm rules compare.
"""

from ....catalog import Family
from ..imports.common import python_files

# Sentences end at `.`, `!` or `?` outside a code span; a run of backticks opens the span and the
# next run closes it (an unclosed span reaches the end of the text).
_SENTENCE_TOKENS = r"`+[^`]*(?:`+|$)|[.!?]|[^`.!?]+"

# A docstring reads like Python reads its literal: prefix and quotes dropped, escapes undone
# unless the prefix has an r.
_UNESCAPED = r"""
    CASE WHEN contains(lower(prefix), 'r') THEN between_quotes
         ELSE array_to_string(list_transform(
                regexp_extract_all(between_quotes, '\\(?:.|\n)|[^\\]+'),
                lambda t: CASE t WHEN '\\' THEN '\' WHEN '\n' THEN chr(10) WHEN '\t' THEN chr(9)
                            WHEN '\r' THEN chr(13) WHEN '\''' THEN chr(39) WHEN '\"' THEN chr(34)
                            WHEN '\' || chr(10) THEN '' ELSE t END), '') END
"""


def _words(text: str) -> str:
    return rf"len(regexp_extract_all({text}, '\S+'))"


class ProseSegment(Family):
    family = "ProseSegmentFact"
    relations = {
        "facts": f"""
            WITH {python_files("python_files")}
            SELECT row_number() OVER (ORDER BY f.path) - 1 AS fact_order,
                   'prosesegmentfact:' || f.path AS fact_id, f.path, 1 AS start_line,
                   0 AS start_column, m.end_line, 0 AS end_column, 'python' AS language
            FROM python_files f
            JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.node_id = 0
        """,
        "values": rf"""
            WITH {python_files("python_files")}, heads AS (
              SELECT d.fact_id, d.node_id AS definition, c.node_id AS statement, c.type
              FROM ast_nodes d
              JOIN python_files f ON f.fact_id = d.fact_id
              JOIN ast_nodes b ON b.fact_id = d.fact_id AND b.parent_id = d.node_id
                AND b.type = 'block'
              JOIN ast_nodes c ON c.fact_id = d.fact_id AND c.parent_id = b.node_id
                AND c.type <> 'comment'
              WHERE d.type IN ('function_definition', 'class_definition')
              QUALIFY row_number() OVER (PARTITION BY d.fact_id, d.node_id
                                         ORDER BY c.sibling_index) = 1
            ), inside AS (
              SELECT h.fact_id, h.definition, h.statement, x.node_id, x.type,
                     x.depth - s.depth - 1 AS between_hops,
                     x.node_id + x.descendant_count AS reach
              FROM heads h
              JOIN ast_nodes s ON s.fact_id = h.fact_id AND s.node_id = h.statement
              JOIN ast_nodes x ON x.fact_id = h.fact_id AND x.node_id > h.statement
                AND x.node_id <= h.statement + s.descendant_count
              WHERE h.type = 'expression_statement'
            ), expressions AS (
              SELECT c.fact_id, c.definition, c.node_id, c.type
              FROM inside c
              LEFT JOIN inside p ON p.fact_id = c.fact_id AND p.statement = c.statement
                AND p.type = 'parenthesized_expression' AND p.node_id < c.node_id
                AND c.node_id <= p.reach
              WHERE c.type IN ('string', 'concatenated_string')
              GROUP BY c.fact_id, c.definition, c.node_id, c.type, c.between_hops
              HAVING count(p.node_id) = c.between_hops
            ), literals AS (
              SELECT fact_id, definition, node_id AS string FROM expressions WHERE type = 'string'
              UNION ALL
              SELECT x.fact_id, x.definition, s.node_id
              FROM expressions x
              JOIN ast_nodes s ON s.fact_id = x.fact_id AND s.parent_id = x.node_id
                AND s.type = 'string'
              WHERE x.type = 'concatenated_string'
            ), contents AS (
              SELECT l.fact_id, l.definition, l.string, prefix,
                     CASE WHEN starts_with(rest, repeat(chr(34), 3))
                            OR starts_with(rest, repeat(chr(39), 3))
                          THEN substr(rest, 4, length(rest) - 6)
                          ELSE substr(rest, 2, length(rest) - 2) END AS between_quotes
              FROM literals l
              JOIN ast_nodes s ON s.fact_id = l.fact_id AND s.node_id = l.string,
              LATERAL (SELECT regexp_extract(s.name, '^[A-Za-z]*') AS prefix) p,
              LATERAL (SELECT substr(s.name, length(prefix) + 1) AS rest) r
            ), docstrings AS (
              SELECT fact_id, definition,
                     string_agg({_UNESCAPED}, '' ORDER BY string) AS content
              FROM contents
              GROUP BY fact_id, definition
              HAVING NOT bool_or(regexp_matches(prefix, '[fFbB]'))
            ), texts AS (
              SELECT fact_id, definition,
                     row_number() OVER (PARTITION BY fact_id ORDER BY definition) - 1 AS section,
                     trim(split_part(content, chr(10), 1)) AS summary,
                     trim(regexp_replace(substr(content, strpos(content, chr(10)) + 1),
                                         '(?m)^[^\S\n]+$', '', 'g'), chr(10)) AS body,
                     contains(content, chr(10)) AS multiline
              FROM docstrings
            ), docs AS (
              SELECT fact_id, section,
                     CASE WHEN multiline AND body <> '' THEN summary || chr(10) || chr(10) || body
                          ELSE summary END AS text
              FROM texts
            ), pieces AS (
              SELECT fact_id, section, unnest(parts) AS piece,
                     generate_subscripts(parts, 1) AS position
              FROM (SELECT *, regexp_extract_all(text, '{_SENTENCE_TOKENS}') AS parts FROM docs)
            ), sentences AS (
              SELECT fact_id, section, piece, position,
                     coalesce(sum((piece IN ('.', '!', '?'))::INTEGER) OVER (
                       PARTITION BY fact_id, section ORDER BY position
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS sentence
              FROM pieces
            ), spoken AS (
              SELECT fact_id, section, sentence, {_words("text")} AS words,
                     lower(regexp_extract(text, '\S+')) AS opener
              FROM (SELECT fact_id, section, sentence, string_agg(piece, '' ORDER BY position)
                             AS text
                    FROM sentences
                    WHERE piece NOT IN ('.', '!', '?')
                    GROUP BY fact_id, section, sentence)
            ), paragraphs AS (
              SELECT fact_id, section, {_words("paragraph")} AS words, position
              FROM (SELECT fact_id, section,
                           unnest(string_split(text, chr(10) || chr(10))) AS paragraph,
                           generate_subscripts(string_split(text, chr(10) || chr(10)), 1)
                             AS position
                    FROM docs)
            ), measured AS (
              SELECT fact_id, section, 'sections.sentence_word_counts.root' AS relation,
                     NULL AS string_value, words AS integer_value, sentence AS position
              FROM spoken WHERE words > 0
              UNION ALL
              SELECT fact_id, section, 'sections.sentence_openers', opener, NULL, sentence
              FROM spoken WHERE words > 0
              UNION ALL
              SELECT fact_id, section, 'sections.paragraph_word_counts.root', NULL, words, position
              FROM paragraphs WHERE words > 0
            )
            SELECT 'prosesegmentfact:' || files.path AS fact_id, relation,
                   'prosesegmentfact:' || files.path || '/sections:' || section AS parent_id,
                   row_number() OVER (PARTITION BY measured.fact_id, section, relation
                                      ORDER BY position) - 1 AS ordinal,
                   string_value, integer_value
            FROM measured JOIN ast_files files ON files.fact_id = measured.fact_id
        """,
    }
