"""The words, patterns and SQL texts the readers of a manuscript's sites share."""

from ...constants.manuscripts.core import SPACE_CHARS

INHERITED_PROOF_NAMES_SQL = """
        WITH inherited AS (
          SELECT p.fact_id, p.scope, list(DISTINCT m.name) AS names
          FROM manuscript_notation_proof_targets p
          JOIN manuscript_notation_marks m
            ON m.fact_id = p.fact_id AND m.statement_number = p.statement_number
           AND (m.declaration OR m.bound_hit)
           AND (m.local OR m.bound_hit OR m.locals_hit OR m.family_hit)
          GROUP BY p.fact_id, p.scope
        )
        SELECT m.*, m.locals_hit OR m.family_hit
                      OR coalesce(list_contains(i.names, m.name), false) AS local_here
        FROM manuscript_notation_marks m
        LEFT JOIN inherited i ON i.fact_id = m.fact_id AND i.scope = m.scope
    """
CAPTIONED_ROLE_RESOLUTION_SQL = """
        WITH RECURSIVE captioned AS (
          SELECT fact_id, reading_order, span, name, position
          FROM (SELECT h.fact_id, h.reading_order, h.span, h.purpose, h.text, u.name, u.position,
                       count(*) OVER (PARTITION BY h.fact_id, h.reading_order, h.piece) AS names
                FROM manuscript_notation_caption_heads h
                JOIN manuscript_notation_uses u
                  ON u.fact_id = h.fact_id AND u.reading_order = h.reading_order
                 AND u.piece = h.piece)
          WHERE (purpose = 'assignment' AND names = 1)
             OR (purpose = 'tuple' AND names = len(string_split(text, ',')))
        ), found AS (
          SELECT fact_id, name AS symbol, reading_order, position, path, line, section_number,
                 statement_number, display AS is_display,
                 CASE WHEN first_name = name THEN meaning ELSE '' END AS meaning,
                 appositive AND first_name = name AS appositive,
                 local OR bound_hit OR local_here AS is_local
          FROM manuscript_notation_occurrences
          WHERE declaration OR bound_hit
          UNION ALL
          SELECT n.fact_id, n.name, n.reading_order, 1000 * n.span + n.position, c.path, c.line,
                 c.section_number, 0, false, '', false, false
          FROM captioned n
          JOIN manuscript_notation_captions c
            ON c.fact_id = n.fact_id AND c.reading_order = n.reading_order AND c.span = n.span
        ), placed AS (
          SELECT *, row_number() OVER (PARTITION BY fact_id ORDER BY reading_order, position) - 1
                      AS ordinal
          FROM found
        ), ranked AS (
          SELECT *, row_number() OVER (PARTITION BY fact_id, symbol ORDER BY ordinal) AS rank
          FROM placed
        ), families AS (
          SELECT fact_id, symbol, list(meaning ORDER BY rank) AS roles,
                 list(appositive ORDER BY rank) AS appositive, count(*) AS size
          FROM ranked GROUP BY fact_id, symbol HAVING bool_or(appositive)
        ), walk(fact_id, symbol, done, roles) AS (
          SELECT fact_id, symbol, 0, roles FROM families
          UNION ALL
          SELECT w.fact_id, w.symbol, w.done + 1,
                 CASE WHEN f.appositive[w.done + 1] THEN list_concat(
                   list_slice(w.roles, 1, w.done),
                   [coalesce(list_filter(w.roles, lambda other, k:
                       k <> w.done + 1 AND other <> ''
                       AND list_sort(list_distinct(string_split(other, ' ')))
                           <> list_sort(list_distinct(string_split(w.roles[w.done + 1], ' ')))
                       AND (list_has_all(string_split(other, ' '),
                                         string_split(w.roles[w.done + 1], ' '))
                            OR list_has_all(string_split(w.roles[w.done + 1], ' '),
                                            string_split(other, ' '))))[1],
                             w.roles[w.done + 1])],
                   list_slice(w.roles, w.done + 2, len(w.roles)))
                 ELSE w.roles END
          FROM walk w JOIN families f ON f.fact_id = w.fact_id AND f.symbol = w.symbol
          WHERE w.done < f.size
        ), settled AS (
          SELECT w.fact_id, w.symbol, w.roles
          FROM walk w JOIN families f ON f.fact_id = w.fact_id AND f.symbol = w.symbol
          WHERE w.done = f.size
        )
        SELECT replace(r.fact_id, 'manuscriptfact:', 'manuscriptnotationfact:') AS fact_id,
               r.ordinal, r.symbol, r.reading_order, r.path, r.line, r.section_number,
               r.statement_number, r.is_display, coalesce(s.roles[r.rank], r.meaning) AS meaning,
               r.is_local
        FROM ranked r LEFT JOIN settled s ON s.fact_id = r.fact_id AND s.symbol = r.symbol
    """

# The words that introduce a symbol in running prose rather than merely use one.
CUES = [
    "be",
    "call",
    "called",
    "define",
    "defined",
    "defines",
    "denote",
    "denoted",
    "denotes",
    "let",
    "where",
    "with",
    "write",
    "writes",
    "written",
    "choose",
    "fix",
    "take",
    "put",
    "set",
]
# The words that open a noun phrase naming the symbol set right after it.
OPENERS = [
    "the",
    "a",
    "an",
    "its",
    "their",
    "each",
    "every",
    "at",
    "of",
    "for",
    "by",
    "in",
    "on",
    "to",
    "from",
    "into",
    "over",
    "under",
    "between",
    "per",
    "via",
    "with",
]
# The words that cannot belong to an appositive noun phrase.
BREAKS = [
    "and",
    "or",
    "but",
    "nor",
    "so",
    "then",
    "is",
    "are",
    "be",
    "been",
    "was",
    "were",
    "has",
    "have",
    "had",
    "gives",
    "give",
    "returns",
    "return",
    "takes",
    "take",
    "where",
    "when",
    "which",
    "while",
    "if",
    "not",
    "only",
    "also",
    "both",
    "all",
    "this",
    "that",
    "these",
    "those",
    "some",
    "any",
    "no",
    "as",
    "than",
    "denote",
    "denotes",
    "means",
]
ROLES = [
    "operand",
    "vector",
    "matrix",
    "scalar",
    "function",
    "operator",
    "index",
    "node",
    "set",
    "domain",
]
PREFIXES = ["denote ", "denotes ", "be ", "means ", "represents ", "stand for ", "stands for "]
LISTING = ["\\ldots", "\\cdots", "\\dots"]
DEFINING = [":=", "\\coloneqq", "\\equiv", "\\triangleq", "="]

INSIDE = r"\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
SYMBOL = r"(?:[A-Za-z]|\\[A-Za-z]+)(?:_(?:\{(?:[^{}]|\{[^{}]*\})*\}|\\[A-Za-z]+|.))?"
WORD = f"[^{SPACE_CHARS}]+"
