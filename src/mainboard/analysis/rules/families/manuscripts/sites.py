"""Where a manuscript introduces its symbols: the cues, roles and bound variables of a span.

A span of mathematics introduces a symbol when the words before it cue a definition (`let`,
`denote`), when it defines the symbol (`x := ...`), or when the words around it say what the
symbol is (a role phrase, or the noun phrase set beside it). A symbol bound by a sum, a
quantifier or a set-builder is local to its span, and one a theorem or proof declares is local to
that statement.
"""

from ....catalog import Function
from ..constants.manuscripts.core import SPACE_CHARS
from .reader import quoted

_INHERITED_PROOF_NAMES_SQL = """
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
_CAPTIONED_ROLE_RESOLUTION_SQL = """
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
_CUES = [
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
_OPENERS = [
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
_BREAKS = [
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
_ROLES = [
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
_PREFIXES = ["denote ", "denotes ", "be ", "means ", "represents ", "stand for ", "stands for "]
_LISTING = ["\\ldots", "\\cdots", "\\dots"]
_DEFINING = [":=", "\\coloneqq", "\\equiv", "\\triangleq", "="]

_INSIDE = r"\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
_SYMBOL = r"(?:[A-Za-z]|\\[A-Za-z]+)(?:_(?:\{(?:[^{}]|\{[^{}]*\})*\}|\\[A-Za-z]+|.))?"
_WORD = f"[^{SPACE_CHARS}]+"


def _list(words: list[str]) -> str:
    return "[" + ", ".join(map(quoted, words)) + "]"


class ManuscriptTrimmed(Function):
    """The text without the whitespace around it, Unicode's as Rust reads it."""

    text: str

    expression = f"regexp_replace(text, '^[{SPACE_CHARS}]+|[{SPACE_CHARS}]+$', '', 'g')"


class ManuscriptLastSentence(Function):
    """The last sentence of a text, cut as `manuscript_ends_sentence` cuts: what follows the last
    sentence end when that is more than whitespace, else the sentence before."""

    text: str

    expression = (
        f"list_transform([regexp_extract_all(text, '{_WORD}|[{SPACE_CHARS}]+')], "
        "lambda tokens: list_transform([list_filter(range(1, len(tokens) + 1), "
        "lambda i: manuscript_ends_sentence(tokens[i]))], lambda cuts: "
        "CASE WHEN len(cuts) = 0 THEN array_to_string(tokens, '') "
        "WHEN len(list_filter(list_slice(tokens, cuts[-1] + 1, len(tokens)), "
        f"lambda t: regexp_matches(t, '[^{SPACE_CHARS}]'))) > 0 "
        "THEN array_to_string(list_slice(tokens, cuts[-1] + 1, len(tokens)), '') "
        "ELSE array_to_string(list_slice(tokens, CASE WHEN len(cuts) > 1 THEN cuts[-2] + 1 "
        "ELSE 1 END, cuts[-1]), '') END)[1])[1]"
    )


class ManuscriptFirstSentence(Function):
    """The first sentence of a text, cut as `manuscript_ends_sentence` cuts; the whole text when
    it ends no sentence."""

    text: str

    expression = (
        f"list_transform([regexp_extract_all(text, '{_WORD}|[{SPACE_CHARS}]+')], "
        "lambda tokens: list_transform([list_filter(range(1, len(tokens) + 1), "
        "lambda i: manuscript_ends_sentence(tokens[i]))[1]], lambda cut: "
        "CASE WHEN cut IS NULL THEN text "
        "ELSE array_to_string(list_slice(tokens, 1, cut), '') END)[1])[1]"
    )


class ManuscriptIndexBase(Function):
    """What stands before a one-letter subscript (`x` of `x_i` and `x_{i}`), NULL when the symbol
    has none."""

    name: str

    expression = r"nullif(regexp_extract(name, '^(.*)_(?:\{[A-Za-z]\}|[A-Za-z])$', 1), '')"


class ManuscriptIndexLetter(Function):
    """The one-letter subscript of a symbol (`i` of `x_i` and `x_{i}`), NULL when it has none."""

    name: str

    expression = (
        r"coalesce(nullif(regexp_extract(name, '^(.*)_(?:\{([A-Za-z])\}|([A-Za-z]))$', 2), ''), "
        r"nullif(regexp_extract(name, '^(.*)_(?:\{([A-Za-z])\}|([A-Za-z]))$', 3), ''))"
    )


class ManuscriptMaskCommas(Function):
    """The text with the commas inside brackets, braces and parentheses turned into `§`, so the
    commas that remain separate its top-level items."""

    text: str

    expression = (
        "array_to_string(list_transform(range(1, length(text) + 1), lambda i: "
        "CASE WHEN substr(text, i, 1) = ',' AND "
        "length(regexp_replace(substr(text, 1, i), '[^(\\[{]', '', 'g')) "
        "> length(regexp_replace(substr(text, 1, i), '[^)\\]}]', '', 'g')) "
        "THEN '§' ELSE substr(text, i, 1) END), '')"
    )


def _definitions(source: str, *, carried: str = "") -> str:
    """The common table expressions `pieces`, `related`, `assignments` and `tuples`.

    They run over the spans of `source` (`fact_id`, `reading_order`, `expanded` and the `carried`
    columns): the top-level items of each span, the left side of the relation in each, and the
    parts of a tuple a call-like left side (`\\JSD(p, q) = ...`) introduces.
    """
    return f"""
        pieces AS (
          SELECT fact_id, reading_order{carried},
                 replace(unnest(string_split(manuscript_mask_commas(row), ',')), '§', ',') AS piece
          FROM (SELECT fact_id, reading_order{carried},
                       unnest(string_split(expanded, chr(92) || chr(92))) AS row
                FROM {source})
        ), related AS (
          SELECT *, list_min(list_filter(
                      list_transform({_list(_DEFINING)}, lambda t: strpos(piece, t)),
                      lambda x: x > 0)) AS stop
          FROM pieces
        ), assignments AS (
          SELECT fact_id, reading_order{carried}, 'assignment' AS purpose,
                 split_part(substr(piece, 1, stop - 1), '(', 1) AS text
          FROM related WHERE stop IS NOT NULL
        ), tuples AS (
          SELECT fact_id, reading_order{carried}, 'tuple' AS purpose,
                 substr(rest, 1, length(rest) - 1) AS text
          FROM (SELECT fact_id, reading_order{carried}, head, split_part(head, '(', 1) AS prefix,
                       substr(head, strpos(head, '(') + 1) AS rest
                FROM (SELECT fact_id, reading_order{carried},
                             manuscript_trimmed(rtrim(manuscript_trimmed(
                               substr(piece, 1, stop - 1)), '&')) AS head
                      FROM related WHERE stop IS NOT NULL))
          WHERE contains(head, '(') AND ends_with(rest, ')') AND manuscript_names_nothing(prefix)
    """


def heads() -> str:
    """The texts whose symbols mean something about a span.

    Those are the left side of each assignment, the variables a sum, quantifier or set-builder
    binds, and the parameters of a function the span defines.
    """
    sums = r"\\(?:sum|prod|bigcup|bigcap)\s*_"
    return f"""
        WITH {_definitions("manuscript_notation_spans")}
        ), binders AS (
          SELECT fact_id, reading_order, 'binder' AS purpose,
                 split_part(split_part(split_part(head, '=', 1), ':', 1), chr(92) || 'in', 1)
                   AS text
          FROM (SELECT fact_id, reading_order,
                       unnest(list_concat(
                         regexp_extract_all(expanded, {quoted(sums + _INSIDE)}, 1),
                         regexp_extract_all(expanded, {quoted(sums + "([A-Za-z]+)")}, 1))) AS head
                FROM manuscript_notation_spans)
          WHERE NOT contains(head, chr(92) || 'substack')
        ), builders AS (
          SELECT fact_id, reading_order, 'builder' AS purpose, head AS text
          FROM (SELECT fact_id, reading_order,
                       regexp_extract(expanded, {quoted(chr(92) * 2 + chr(92) + "{([^:]*):")}, 1)
                         AS head
                FROM manuscript_notation_spans)
          WHERE head <> '' AND NOT contains(head, chr(92) || '}}')
            AND NOT regexp_matches(head, '[(=+*/-]')
        ), quantified AS (
          SELECT fact_id, reading_order, 'quantified' AS purpose,
                 unnest(regexp_extract_all(expanded,
                   {quoted(r"\\(?:forall|exists)\s*(" + _SYMBOL + r"(?:\s*,\s*" + _SYMBOL + ")*)")},
                   1)) AS text
          FROM manuscript_notation_spans
        ), statement_spans AS (
          SELECT fact_id, reading_order, expanded, tail,
                 strpos(expanded, chr(92) || 'in') AS stop,
                 list_filter({_list(_LISTING)}, lambda d: contains(expanded, d))[1] AS dots
          FROM manuscript_notation_spans WHERE statement_number > 0
        ), vectors AS (
          SELECT fact_id, reading_order, 'vector' AS purpose, left_part AS text
          FROM (SELECT *, substr(expanded, 1, stop - 1) AS left_part,
                       regexp_replace(substr(expanded, stop + 3), '^[{SPACE_CHARS}]+', '') AS domain
                FROM statement_spans WHERE stop > 0)
          WHERE regexp_matches(left_part, {quoted(rf"^[\pL{SPACE_CHARS},{{}}\\]*$")})
            AND regexp_matches(domain, {quoted(r"^\\mathbb(?:\{[RC]\}| [RC])\^")})
        ), indices AS (
          SELECT fact_id, reading_order, 'index' AS purpose, left_part AS text
          FROM (SELECT *, substr(expanded, 1, stop - 1) AS left_part,
                       lower(manuscript_trimmed(rtrim(manuscript_trimmed(
                         manuscript_last_sentence(tail)), '·'))) AS before
                FROM statement_spans WHERE stop > 0)
          WHERE starts_with(before, 'for ')
            AND regexp_matches(left_part, {quoted(rf"^[A-Za-z{SPACE_CHARS},]*$")})
        ), listings AS (
          SELECT fact_id, reading_order, 'listed_left' AS purpose,
                 substr(expanded, 1, strpos(expanded, dots) - 1) AS text
          FROM statement_spans WHERE dots IS NOT NULL
          UNION ALL
          SELECT fact_id, reading_order, 'listed_right',
                 substr(expanded, strpos(expanded, dots) + length(dots))
          FROM statement_spans WHERE dots IS NOT NULL
        ), parameters AS (
          SELECT fact_id, reading_order, purpose, text
          FROM (SELECT fact_id, reading_order,
                       split_part(trim(split_part(expanded, '=', 1)), '(', 1) AS name,
                       substr(trim(split_part(expanded, '=', 1)),
                              strpos(trim(split_part(expanded, '=', 1)), '(') + 1) AS arguments
                FROM manuscript_notation_spans
                WHERE contains(expanded, '=') AND contains(split_part(expanded, '=', 1), '('))
          CROSS JOIN LATERAL (VALUES ('function', name), ('parameters', rtrim(arguments, ')')))
                            AS v(purpose, text)
          WHERE ends_with(arguments, ')')
            AND regexp_matches(rtrim(arguments, ')'), {quoted(r"^[\pL, \\]*$")})
        )
        SELECT *, 1000 + row_number() OVER (PARTITION BY fact_id, reading_order
                                            ORDER BY purpose, text) AS piece
        FROM (SELECT * FROM assignments UNION ALL SELECT * FROM binders
              UNION ALL SELECT * FROM builders UNION ALL SELECT * FROM quantified
              UNION ALL SELECT * FROM parameters UNION ALL SELECT * FROM vectors
              UNION ALL SELECT * FROM indices UNION ALL SELECT * FROM listings
              UNION ALL SELECT * FROM tuples)
    """


def caption_heads() -> str:
    """The texts of the definitions a caption's mathematics states, like `heads` for spans."""
    return f"""
        WITH {_definitions("manuscript_notation_captions", carried=", span")}
        )
        SELECT *, 1000 + row_number() OVER (PARTITION BY fact_id, reading_order
                                            ORDER BY span, purpose, text) AS piece
        FROM (SELECT * FROM assignments UNION ALL SELECT * FROM tuples)
    """


def readings() -> str:
    """What the prose around each span says about it.

    That is whether its words cue a definition, the role phrase they give the span's first symbol,
    and the symbols the span itself defines, binds or introduces.
    """
    cues = _list(_CUES)
    word_pattern = quoted(_WORD)
    return rf"""
        WITH keyed AS (
          SELECT s.*, lower(manuscript_last_sentence(s.tail)) AS lead,
                 manuscript_trimmed(rtrim(manuscript_trimmed(s.tail), '·')) AS connective,
                 lower(manuscript_trimmed(s.following)) AS after,
                 f.name AS first_name
          FROM manuscript_notation_spans s
          LEFT JOIN (SELECT fact_id, reading_order, name FROM manuscript_notation_uses
                     WHERE piece = 0 AND position = 1) f
            ON f.fact_id = s.fact_id AND f.reading_order = s.reading_order
        ), termed AS (
          SELECT *, manuscript_trimmed(rtrim(manuscript_trimmed(lead), '·')) AS before
          FROM keyed
        ), worded AS (
          SELECT k.*, EXISTS (
                   SELECT 1 FROM manuscript_notation_terms t
                   WHERE t.fact_id = replace(k.fact_id, 'manuscriptfact:', 'manuscriptnotationfact:')
                     AND t.term = k.before AND t.mark_order < k.reading_order) AS term_hit,
                 list_filter(list_transform(regexp_extract_all(lead, {word_pattern}),
                               lambda w: regexp_replace(w, '^[^\pL\pN]+|[^\pL\pN]+$', '', 'g')),
                             lambda w: w <> '') AS words,
                 regexp_extract(regexp_extract(lead, '[^,;]*$'), {word_pattern}) AS clause_head
          FROM termed k
        ), cued AS (
          SELECT *, (len(words) > 0 AND (list_contains({cues}, words[1])
                                         OR words[1] IN ('assume', 'suppose', 'given')))
                    OR list_contains({cues}, clause_head)
                    OR starts_with(regexp_replace(lead, '^[{SPACE_CHARS}]+', ''), 'for each ')
                    OR starts_with(regexp_replace(lead, '^[{SPACE_CHARS}]+', ''), 'for every ')
                    OR term_hit
                    OR len(list_filter(list_slice(words, greatest(1, len(words) - 3), len(words)),
                                       lambda w: list_contains({cues}, w))) > 0 AS is_cued
          FROM worded
        ), roled AS (
          SELECT *,
                 list_transform(list_filter({_list(_PREFIXES)}, lambda p: starts_with(after, p)),
                                lambda p: substr(after, length(p) + 1))[1] AS spoken,
                 list_filter({_list(["for ", "and ", "every ", "each "])},
                             lambda p: starts_with(before, p))[1] AS asked,
                 list_filter({_list([" have an ", " have a ", " has an ", " has a "])},
                             lambda p: contains(before, p))[1] AS possessing
          FROM cued
        ), meant AS (
          SELECT *, coalesce(
                   spoken,
                   CASE WHEN is_cued AND starts_with(after, 'for ') THEN substr(after, 5) END,
                   CASE WHEN term_hit THEN before END,
                   CASE WHEN asked IS NOT NULL
                          AND len(regexp_extract_all(substr(before, length(asked) + 1), {word_pattern})) <= 4
                          AND list_contains({_list(_ROLES)},
                                regexp_extract_all(substr(before, length(asked) + 1), {word_pattern})[-1])
                        THEN substr(before, length(asked) + 1) END,
                   CASE WHEN starts_with(after, 'in which') AND possessing IS NOT NULL
                        THEN regexp_extract(before, '.*' || possessing || '(.*)$', 1) END) AS role
          FROM roled
        ), roles AS (
          SELECT *, CASE WHEN regexp_matches(rtrim(manuscript_trimmed(role), '.,;:'), '\pL')
                         THEN rtrim(manuscript_trimmed(role), '.,;:') ELSE '' END AS bare_role
          FROM meant
        ), named AS (
          SELECT *, array_to_string(regexp_extract_all(
                      CASE WHEN starts_with(bare_role, 'the ') THEN substr(bare_role, 5)
                           WHEN starts_with(bare_role, 'a ') THEN substr(bare_role, 3)
                           WHEN starts_with(bare_role, 'an ') THEN substr(bare_role, 4)
                           ELSE bare_role END, {word_pattern}), ' ') AS said,
                    list_reverse(regexp_extract_all(regexp_extract(before, '[^,;:(]*$'),
                                                    {word_pattern})) AS reversed,
                    list_filter(range(1, length(expanded) + 1), lambda i:
                      (substr(expanded, i, 1) = '='
                       OR regexp_matches(substr(expanded, i, 4), '^\\in([^A-Za-z]|$)'))
                      AND length(substr(expanded, 1, i - 1))
                          - length(replace(substr(expanded, 1, i - 1), '{{', ''))
                          <= length(substr(expanded, 1, i - 1))
                          - length(replace(substr(expanded, 1, i - 1), '}}', ''))) AS relations
          FROM roles
        ), applied AS (
          SELECT *, k AS opener,
                 NOT display AND said = '' AND len(relations) > 0 AND first_name IS NOT NULL
                   AND regexp_replace(substr(expanded, 1, relations[1] - 1), '[{SPACE_CHARS}{{}}]', '', 'g')
                       = regexp_replace(first_name, '[{SPACE_CHARS}{{}}]', '', 'g') AS apposed
          FROM (SELECT *, list_filter(range(1, 5), lambda i: i <= len(reversed)
                                       AND list_contains({_list(_OPENERS)}, reversed[i]))[1] AS k
                FROM named)
        ), phrased AS (
          SELECT *, CASE WHEN apposed AND opener IS NOT NULL
                              AND len(list_filter(range(1, opener), lambda i:
                                       NOT regexp_matches(reversed[i], '^[A-Za-z-]+$')
                                       OR ends_with(reversed[i], 'ly')
                                       OR list_contains({_list(_BREAKS)}, reversed[i]))) = 0
                         THEN array_to_string(list_reverse(list_slice(reversed, 1, opener - 1)), ' ')
                         ELSE said END AS role_text
          FROM applied
        )
        SELECT fact_id, reading_order, path, line, section_number, statement_number, statement_kind,
               scope, block, display, in_proof, first_name, is_cued, before, connective, following,
               expanded, role_text AS meaning, apposed AND role_text <> '' AS appositive
        FROM phrased
    """


def marks() -> str:
    """Each occurrence of a symbol in a span, and whether it is declared, bound or local.

    A span declares a symbol by a cue, a definition or a role phrase. A symbol is local to its
    theorem or proof when a declaration in that scope made it local, when the span makes it local
    before its own symbols (vectors, indices of a `for ... in` clause, the symbols of a quantified
    or listed component), or when it is indexed by a local index and its family is declared.
    """
    own = "purpose NOT IN ('assignment', 'function', 'vector', 'index', 'listed_left', 'listed_right', 'tuple')"
    before = "(i.at_order < o.reading_order OR (i.at_order = o.reading_order AND i.at_pos <= o.position))"
    return rf"""
        WITH named AS (
          SELECT h.fact_id, h.reading_order, h.purpose, h.piece, h.text, u.name, u.position
          FROM manuscript_notation_heads h
          JOIN manuscript_notation_uses u
            ON u.fact_id = h.fact_id AND u.reading_order = h.reading_order AND u.piece = h.piece
        ), introduced AS (
          SELECT fact_id, reading_order, list(name) AS names
          FROM (SELECT fact_id, reading_order, any_value(name) AS name
                FROM named WHERE purpose = 'assignment'
                GROUP BY fact_id, reading_order, piece HAVING count(*) = 1
                UNION ALL
                SELECT fact_id, reading_order, name
                FROM named WHERE purpose = 'tuple'
                QUALIFY count(*) OVER (PARTITION BY fact_id, reading_order, piece)
                        = len(string_split(text, ',')))
          GROUP BY fact_id, reading_order
        ), functions AS (
          SELECT fact_id, reading_order FROM named WHERE purpose = 'function'
          GROUP BY fact_id, reading_order HAVING count(*) = 1
        ), bound AS (
          SELECT n.fact_id, n.reading_order, list(n.name) AS names
          FROM named n
          WHERE n.{own}
            AND (n.purpose <> 'parameters' OR EXISTS (
                   SELECT 1 FROM functions f
                   WHERE f.fact_id = n.fact_id AND f.reading_order = n.reading_order))
          GROUP BY n.fact_id, n.reading_order
        ), counted AS (
          SELECT fact_id, reading_order, count(*) AS own_names
          FROM manuscript_notation_uses WHERE piece = 0 GROUP BY ALL
        ), spans AS (
          SELECT r.*, coalesce(i.names, []::VARCHAR[]) AS introduced,
                 coalesce(b.names, []::VARCHAR[]) AS bound,
                 r.in_proof OR r.statement_kind IN ('theorem', 'lemma', 'proposition', 'corollary')
                   AS local,
                 r.is_cued OR (regexp_extract(r.connective, '[^ ]*$') = 'use'
                               AND starts_with(ltrim(r.following), 'for ')) AS cue_here,
                 r.connective IN ('and', 'or', ',', ', and') AS links,
                 coalesce(c.own_names, 0) AS own_names
          FROM manuscript_notation_readings r
          LEFT JOIN introduced i ON i.fact_id = r.fact_id AND i.reading_order = r.reading_order
          LEFT JOIN bound b ON b.fact_id = r.fact_id AND b.reading_order = r.reading_order
          LEFT JOIN counted c ON c.fact_id = r.fact_id AND c.reading_order = r.reading_order
        ), chained AS (
          SELECT *, bool_or(cue_here) OVER (PARTITION BY fact_id, chain ORDER BY reading_order)
                      AS cued
          FROM (SELECT *, sum((NOT links OR fresh)::INTEGER) OVER (
                              PARTITION BY fact_id ORDER BY reading_order) AS chain
                FROM (SELECT *, block <> coalesce(lag(block) OVER (
                                PARTITION BY fact_id ORDER BY reading_order), -1) AS fresh
                      FROM spans))
        ), listed AS (
          SELECT fact_id, reading_order, [left_name, right_name] AS names
          FROM (SELECT fact_id, reading_order,
                       arg_max(name, position) FILTER (purpose = 'listed_left'
                                                       AND contains(name, '_')) AS left_name,
                       arg_min(name, position) FILTER (purpose = 'listed_right'
                                                       AND contains(name, '_')) AS right_name
                FROM named WHERE purpose IN ('listed_left', 'listed_right')
                GROUP BY fact_id, reading_order)
          WHERE regexp_extract(left_name, '^(.*)_[^_]*$', 1)
                = regexp_extract(right_name, '^(.*)_[^_]*$', 1)
        ), made AS (
          SELECT c.fact_id, c.scope, c.reading_order, n.name
          FROM chained c
          JOIN named n ON n.fact_id = c.fact_id AND n.reading_order = c.reading_order
          WHERE (n.purpose = 'vector' AND starts_with(n.name, chr(92) || 'mathbf{{')
                 AND ends_with(n.name, '}}')) OR n.purpose = 'index'
          UNION ALL
          SELECT c.fact_id, c.scope, c.reading_order, unnest(l.names)
          FROM chained c JOIN listed l ON l.fact_id = c.fact_id AND l.reading_order = c.reading_order
          WHERE c.statement_number > 0
            AND (list_contains(regexp_extract_all(c.before, '[^{SPACE_CHARS}]+'), 'every')
                 OR list_contains(regexp_extract_all(c.before, '[^{SPACE_CHARS}]+'), 'each')
                 OR ends_with(c.before, 'primitive')
                 OR (c.cued AND starts_with(ltrim(c.following), 'for ')))
          UNION ALL
          SELECT c.fact_id, c.scope, c.reading_order, u.name
          FROM chained c
          JOIN manuscript_notation_uses u
            ON u.fact_id = c.fact_id AND u.reading_order = c.reading_order AND u.piece = 0
          WHERE c.statement_number > 0 AND c.own_names = 1
            AND (list_contains(regexp_extract_all(c.before, '[^{SPACE_CHARS}]+'), 'every')
                 OR list_contains(regexp_extract_all(c.before, '[^{SPACE_CHARS}]+'), 'each'))
        ), occurring AS (
          SELECT c.fact_id, c.reading_order, c.path, c.line, c.section_number, c.statement_number,
                 c.scope, c.display, c.local, c.meaning, c.appositive, c.first_name, u.name,
                 u.position, c.bound AS span_bound,
                 c.cued OR list_contains(c.introduced, u.name)
                   OR (c.meaning <> '' AND c.first_name = u.name) AS declaration,
                 list_contains(c.bound, u.name) AS bound_hit
          FROM chained c
          JOIN manuscript_notation_uses u
            ON u.fact_id = c.fact_id AND u.reading_order = c.reading_order AND u.piece = 0
        ), inserts AS (
          SELECT fact_id, scope, name, manuscript_index_base(name) AS base,
                 reading_order AS at_order, position AS at_pos
          FROM occurring WHERE local AND declaration
          UNION ALL
          SELECT fact_id, scope, name, manuscript_index_base(name), reading_order, 0 FROM made
        ), marked AS (
          SELECT o.*, manuscript_index_base(o.name) AS base,
                 manuscript_index_letter(o.name) AS letter,
                 EXISTS (SELECT 1 FROM inserts i
                         WHERE i.fact_id = o.fact_id AND i.scope = o.scope AND i.name = o.name
                           AND {before}) AS locals_hit
          FROM occurring o
        )
        SELECT o.*, o.base IS NOT NULL AND (
                 EXISTS (SELECT 1 FROM inserts i
                         WHERE i.fact_id = o.fact_id AND i.scope = o.scope
                           AND i.name = chr(92) || 'mathbf{{' || o.base || '}}' AND {before})
                 OR ((list_contains(o.span_bound, o.letter)
                      OR EXISTS (SELECT 1 FROM inserts i
                                 WHERE i.fact_id = o.fact_id AND i.scope = o.scope
                                   AND i.name = o.letter AND {before}))
                     AND (EXISTS (SELECT 1 FROM inserts i
                                  WHERE i.fact_id = o.fact_id AND i.scope = o.scope
                                    AND i.base = o.base AND {before})
                          OR EXISTS (SELECT 1 FROM marked t
                                     WHERE t.fact_id = o.fact_id AND t.base = o.base
                                       AND (t.declaration OR t.bound_hit)
                                       AND NOT (t.local OR t.bound_hit OR t.locals_hit)
                                       AND (t.reading_order < o.reading_order
                                            OR (t.reading_order = o.reading_order
                                                AND t.position < o.position)))))
               ) AS family_hit
        FROM marked o
    """


def occurrences() -> str:
    """The occurrences with what a proof inherits from the statement it proves.

    The symbols that statement declared are local in the proof.
    """
    return _INHERITED_PROOF_NAMES_SQL


def sites() -> str:
    """Each place a span declares or binds a symbol, with the role the prose gives it.

    An appositive role that repeats another role of the same symbol (one is words of the other)
    takes that role, for it is a reminder of one meaning. The sites merge in order, each seeing the
    roles the earlier ones settled.
    """
    return _CAPTIONED_ROLE_RESOLUTION_SQL
