"""mcmr's manuscript families, derived from the LaTeX files of the project.

A manuscript is a root (a file that declares a document class) with the files it includes spliced
in where they are included. `ManuscriptFact` reads that one reading-order stream (the helper
relation `element`, see manuscripts_stream.py) as the headings, numbered statements, floats,
labels, references, paragraphs and sentences of mcmr's skeleton; the notation and evidence
families read the same stream.

Typst manuscripts are not read: the research writes LaTeX, and the Typst files here are CVs.
"""

from ...catalog import Family, Function
from .manuscripts_reader import quoted, words
from .manuscripts_stream import ELEMENTS

ELEMENT = "manuscript_element"
SPACE_CHARS = r"\s\x{A0}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{85}"
# A sentence never ends at the dot of one of these words.
_ABBREVIATIONS = (
    "al",
    "cf",
    "def",
    "defs",
    "e.g",
    "eq",
    "eqs",
    "etc",
    "fig",
    "figs",
    "i.e",
    "no",
    "prop",
    "resp",
    "sec",
    "secs",
    "tab",
    "thm",
    "vs",
)
_FLOAT_ROLES = "('figure', 'table', 'algorithm')"
# The elements after which a paragraph ends: breaks, headings, captions and every environment
# but display mathematics.
_ENDS = """kind IN ('break', 'rowbreak', 'section', 'caption')
           OR (kind IN ('open', 'close') AND environment_role <> 'math')"""


class ManuscriptEndsSentence(Function):
    """Whether a whitespace-free token closes a sentence: it ends in `.`, `!` or `?` and is neither
    a single letter nor an abbreviation, both with their dot."""

    token: str

    expression = (
        "right(token, 1) IN ('.', '!', '?') AND NOT ((length(token) = 2 "
        r"AND regexp_matches(left(token, 1), '^\pL$')) "
        f"OR lower(left(token, length(token) - 1)) IN ({', '.join(map(quoted, _ABBREVIATIONS))}))"
    )


def _facts() -> str:
    return """
        SELECT row_number() OVER (ORDER BY path) - 1 AS fact_order,
               'manuscriptfact:' || path AS fact_id, path, 1 AS start_line, 0 AS start_column,
               1 AS end_line, 0 AS end_column, 'latex' AS language, path AS root
        FROM source_files
        WHERE path IN (SELECT path FROM ast_files WHERE language = 'latex')
          AND contains(content, chr(92) || 'documentclass')
    """


def _labels() -> str:
    """Each label with what it names: the innermost thing it sits in, else the first label of the
    section it falls to (the rest of a section's labels are equations)."""
    return f"""
        WITH sited AS (
          SELECT l.fact_id, l.reading_order, l.path, l.line, l.a AS name, l.section_number,
                 l.statement_number, l.float_number,
                 CASE WHEN l.in_math THEN 'equation' WHEN l.statement_number > 0 THEN 'statement'
                      WHEN l.float_number > 0 THEN 'float' WHEN l.in_proof THEN 'proof'
                      WHEN l.section_number = 0 THEN 'equation' ELSE 'section' END AS site
          FROM {ELEMENT} l WHERE l.kind = 'label'
        ), ranked AS (
          SELECT *, row_number() OVER (PARTITION BY fact_id, section_number, site = 'section'
                                       ORDER BY reading_order) AS rank
          FROM sited
        )
        SELECT l.fact_id, row_number() OVER (PARTITION BY l.fact_id ORDER BY l.reading_order) - 1
                 AS ordinal,
               l.name, l.reading_order, l.path, l.line, l.section_number, l.statement_number,
               l.float_number, l.site,
               CASE l.site
                 WHEN 'equation' THEN 'equation'
                 WHEN 'statement' THEN (SELECT rtrim(o.a, '*') FROM {ELEMENT} o
                                        WHERE o.fact_id = l.fact_id AND o.kind = 'open' AND o.declared
                                          AND o.statement_number = l.statement_number)
                 WHEN 'float' THEN (SELECT {_float_kind("o")} FROM {ELEMENT} o
                                    WHERE o.fact_id = l.fact_id AND o.kind = 'open'
                                      AND o.environment_role IN {_FLOAT_ROLES}
                                      AND o.float_number = l.float_number)
                 WHEN 'proof' THEN 'proof'
                 WHEN 'section' THEN CASE WHEN l.rank = 1 THEN
                      'section' || (SELECT s.n FROM {ELEMENT} s WHERE s.fact_id = l.fact_id
                                    AND s.kind = 'section' AND s.in_body
                                    AND s.section_number = l.section_number)
                      ELSE 'equation' END END AS kind
        FROM ranked l
    """


def _float_kind(alias: str) -> str:
    return f"{alias}.environment_role"


def _sections() -> str:
    return f"""
        SELECT s.fact_id, row_number() OVER (PARTITION BY s.fact_id ORDER BY s.reading_order) - 1
                 AS ordinal,
               s.reading_order, s.path, s.line, s.n AS level, s.a AS title,
               {words("s.a")} AS title_word_count,
               coalesce(arg_min(l.name, l.reading_order), '') AS label
        FROM {ELEMENT} s
        LEFT JOIN manuscript_labels l
          ON l.fact_id = s.fact_id AND l.section_number = s.section_number
         AND l.site = 'section' AND l.kind <> 'equation'
        WHERE s.kind = 'section' AND s.in_body
        GROUP BY s.fact_id, s.reading_order, s.path, s.line, s.n, s.a
    """


def _statements() -> str:
    return f"""
        WITH opened AS (
          SELECT o.fact_id, o.statement_number AS number, o.reading_order, o.path, o.line,
                 o.section_number, rtrim(o.a, '*') AS kind
          FROM {ELEMENT} o WHERE o.kind = 'open' AND o.declared
        ), kinds AS (
          SELECT fact_id, a AS name, arg_max(n, reading_order) AS owes_proof
          FROM {ELEMENT} WHERE kind = 'statement_kind' GROUP BY fact_id, a
        ), closed AS (
          SELECT fact_id, statement_number AS number, min(reading_order) AS close_order
          FROM {ELEMENT}
          WHERE kind = 'close' AND declared AND statement_number > 0
            AND environment_role NOT IN ('cells', 'math', 'proof', 'bibliography', 'figure',
                                         'table', 'algorithm')
          GROUP BY fact_id, statement_number
        ), cited AS (
          SELECT fact_id, statement_number AS number, count(*) AS keys
          FROM {ELEMENT} WHERE kind = 'citation' AND statement_number > 0 GROUP BY ALL
        ), named AS (
          SELECT fact_id, statement_number AS number, arg_max(name, reading_order) AS label
          FROM manuscript_labels WHERE site = 'statement' GROUP BY ALL
        ), proofs AS (
          SELECT fact_id, reading_order AS proof FROM {ELEMENT}
          WHERE kind = 'open' AND environment_role = 'proof'
        ), blocked AS (
          SELECT p.fact_id, p.proof, min(b.reading_order) AS block
          FROM proofs p LEFT JOIN {ELEMENT} b
            ON b.fact_id = p.fact_id AND b.reading_order > p.proof
           AND b.kind IN ('text', 'math', 'break', 'close')
          GROUP BY p.fact_id, p.proof
        ), headings AS (
          SELECT b.fact_id, b.proof, arg_min(r.a, r.reading_order) AS target
          FROM blocked b JOIN {ELEMENT} r
            ON r.fact_id = b.fact_id AND r.kind = 'reference' AND r.reading_order > b.proof
           AND (b.block IS NULL OR r.reading_order < b.block)
          GROUP BY b.fact_id, b.proof
        ), adjacent AS (
          SELECT c.fact_id, c.number,
                 max(e.reading_order) FILTER (WHERE e.kind = 'open' AND e.environment_role = 'proof')
                   AS proof_order,
                 arg_min(left({_strip("e.a")}, 48), e.reading_order) FILTER (WHERE e.kind = 'text')
                   AS discharge_head
          FROM closed c JOIN {ELEMENT} e
            ON e.fact_id = c.fact_id AND e.reading_order BETWEEN c.close_order + 1
                                                             AND c.close_order + 6
          GROUP BY c.fact_id, c.number
        )
        SELECT o.fact_id, o.number - 1 AS ordinal, o.reading_order, o.path, o.line,
               o.section_number, o.kind, coalesce(n.label, '') AS label,
               coalesce(k.owes_proof <> 0, false) AS owes_proof, coalesce(c.close_order, 0) AS close_order,
               coalesce((SELECT max(h.proof) FROM headings h
                         WHERE h.fact_id = o.fact_id AND h.target = n.label),
                        a.proof_order, 0) AS proof_order,
               coalesce(t.keys, 0) AS attribution_keys_length,
               coalesce(a.discharge_head, '') AS discharge_head, 0 AS word_count
        FROM opened o
        LEFT JOIN kinds k ON k.fact_id = o.fact_id AND k.name = o.kind
        LEFT JOIN closed c ON c.fact_id = o.fact_id AND c.number = o.number
        LEFT JOIN cited t ON t.fact_id = o.fact_id AND t.number = o.number
        LEFT JOIN named n ON n.fact_id = o.fact_id AND n.number = o.number
        LEFT JOIN adjacent a ON a.fact_id = o.fact_id AND a.number = o.number
    """


def _strip(expression: str) -> str:
    return rf"regexp_replace({expression}, '^\s+|\s+$', '', 'g')"


def _floats() -> str:
    return f"""
        WITH opened AS (
          SELECT o.fact_id, o.float_number AS number, o.reading_order, o.path, o.line,
                 {_float_kind("o")} AS kind
          FROM {ELEMENT} o WHERE o.kind = 'open' AND o.environment_role IN {_FLOAT_ROLES}
        ), named AS (
          SELECT fact_id, float_number AS number, arg_max(name, reading_order) AS label
          FROM manuscript_labels WHERE site = 'float' GROUP BY ALL
        ), captioned AS (
          SELECT fact_id, float_number AS number, arg_max(a, reading_order) AS caption
          FROM {ELEMENT} WHERE kind = 'caption' AND in_body AND float_number > 0 GROUP BY ALL
        )
        SELECT o.fact_id, row_number() OVER (PARTITION BY o.fact_id ORDER BY o.number) - 1
                 AS ordinal,
               o.reading_order, o.path, o.line, o.kind, coalesce(n.label, '') AS label,
               coalesce(c.caption, '') AS caption, coalesce({words("c.caption")}, 0) AS caption_word_count
        FROM opened o
        LEFT JOIN named n ON n.fact_id = o.fact_id AND n.number = o.number
        LEFT JOIN captioned c ON c.fact_id = o.fact_id AND c.number = o.number
    """


def _references() -> str:
    return f"""
        SELECT fact_id, row_number() OVER (PARTITION BY fact_id ORDER BY reading_order) - 1
                 AS ordinal,
               reading_order, path, line, section_number, a AS target, b AS command
        FROM {ELEMENT} WHERE kind = 'reference' AND in_body
    """


def _macros() -> str:
    """What a macro expands to in prose when its definition reads as plain words."""
    return f"""
        SELECT fact_id, a AS name, reading_order,
               CASE WHEN regexp_matches(body, '[\\\\$%]') OR NOT regexp_matches(body, '\\S')
                    THEN NULL ELSE ' ' || {_strip("body")} || ' ' END AS expansion
        FROM (SELECT fact_id, a, reading_order,
                     regexp_replace(regexp_replace(b,
                       '\\\\(?:emph|term|textbf|textit|textsc|underline)\\{{', '', 'g'),
                       '[{{}}\\n]', ' ', 'g') AS body
              FROM {ELEMENT} WHERE kind = 'macro')
    """


def _prose() -> str:
    """The running prose of the body, one row per paragraph with its text."""
    return f"""
        WITH body AS (
          SELECT * FROM {ELEMENT} WHERE in_body
        ), spoken AS (
          SELECT fact_id, reading_order, path, line, kind, environment_role, section_number,
                 float_number, in_cells, in_bibliography, a AS txt
          FROM body WHERE kind = 'text'
          UNION ALL
          SELECT u.fact_id, u.reading_order, u.path, u.line, u.kind, u.environment_role,
                 u.section_number, u.float_number, u.in_cells, u.in_bibliography, m.expansion
          FROM (SELECT * FROM body WHERE kind = 'macro_use') u
          ASOF LEFT JOIN manuscript_macro m
            ON u.fact_id = m.fact_id AND u.a = m.name AND u.reading_order >= m.reading_order
          UNION ALL
          SELECT fact_id, reading_order, path, line, kind, environment_role, section_number,
                 float_number, in_cells, in_bibliography, NULL
          FROM body WHERE kind NOT IN ('text', 'macro_use')
        ), cut AS (
          SELECT *, coalesce(sum(({_ENDS})::INTEGER) OVER (
                       PARTITION BY fact_id ORDER BY reading_order
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS paragraph
          FROM spoken
        )
        SELECT fact_id, paragraph, string_agg(txt, '' ORDER BY reading_order) AS body,
               min(reading_order) AS reading_order,
               arg_min(path, reading_order) AS path, arg_min(line, reading_order) AS line,
               arg_min(section_number, reading_order) AS section_number,
               arg_min(in_cells, reading_order) AS in_cells,
               arg_min(float_number > 0, reading_order) AS in_float,
               arg_min(in_bibliography, reading_order) AS in_bibliography
        FROM cut WHERE txt IS NOT NULL
        GROUP BY fact_id, paragraph
        HAVING regexp_matches(string_agg(txt, '' ORDER BY reading_order), '\\S')
    """


def _sentences() -> str:
    """Each paragraph cut where a word ends in `.`, `!` or `?` and is neither a single letter nor
    a known abbreviation; the cuts keep the paragraph's own spacing."""
    return rf"""
        WITH tokens AS (
          SELECT fact_id, paragraph, reading_order, path, line, unnest(parts) AS token,
                 generate_subscripts(parts, 1) AS position
          FROM (SELECT *, regexp_extract_all(body, '[^{SPACE_CHARS}]+|[{SPACE_CHARS}]+') AS parts
                FROM manuscript_prose)
        ), marked AS (
          SELECT *, manuscript_ends_sentence(token) AS ends FROM tokens
        ), cut AS (
          SELECT *, coalesce(sum(ends::INTEGER) OVER (
                       PARTITION BY fact_id, paragraph ORDER BY position
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS sentence
          FROM marked
        ), sentences AS (
          SELECT fact_id, paragraph, any_value(reading_order) AS reading_order,
                 any_value(path) AS path, any_value(line) AS line, sentence,
                 {_strip("string_agg(token, '' ORDER BY position)")} AS text
          FROM cut GROUP BY fact_id, paragraph, sentence
          HAVING regexp_matches(string_agg(token, '' ORDER BY position), '\S')
        )
        SELECT fact_id, reading_order, path, line,
               row_number() OVER (PARTITION BY fact_id, paragraph ORDER BY sentence) - 1 AS "index",
               {words("text")} AS word_count, text
        FROM sentences
    """


def _paragraphs() -> str:
    return f"""
        SELECT p.fact_id, p.reading_order, p.path, p.line, p.section_number,
               {words("p.body")} AS word_count, coalesce(s.sentence_count, 0) AS sentence_count,
               p.in_cells, p.in_float, p.in_bibliography
        FROM manuscript_prose p
        LEFT JOIN (SELECT fact_id, reading_order, count(*) AS sentence_count
                   FROM manuscript_sentences GROUP BY ALL) s
          ON s.fact_id = p.fact_id AND s.reading_order = p.reading_order
    """


class Manuscript(Family):
    family = "ManuscriptFact"
    verified = True
    evidence = frozenset({"latex"})
    relations = {
        "facts": _facts(),
        "element": ELEMENTS,
        "labels": _labels(),
        "macro": _macros(),
        "prose": _prose(),
        "sentences": _sentences(),
        "sections": _sections(),
        "statements": _statements(),
        "floats": _floats(),
        "references": _references(),
        "paragraphs": _paragraphs(),
    }
