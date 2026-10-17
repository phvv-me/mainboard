"""mcmr's ManuscriptEvidenceFact: the numbers a manuscript prints and the sources it cites.

The body is read in order as a run of sentences, equations, citations and references cut into
groups: a group ends where a sentence ends or a paragraph, heading or environment begins. Each
number carries what its group says around it (a ratio word before it or a percent sign after it,
a definition of the ratio, a table cited earlier in the paragraph).
"""

from ...catalog import Family, Function
from .manuscripts import SPACE_CHARS

_NUMBER = "[0-9](?:[0-9.,]*[0-9])?"
_RATIO_WORDS = "fraction|per cent|percent|proportion|ratio|share|times"
_RATIO_DEFINITIONS = ("ratio of", "divided by", "defined as", "numerator", "denominator")
_EMPIRICAL = ("measured", "observed", "reported", "empirical", "median", "percent", "rate")
# What follows a ratio word only when it names the quantity itself: `fraction` but not `refraction`.
_BOUNDARY = r"[^\pL\pN]"
_REGION = ("'break'", "'rowbreak'", "'section'", "'open'", "'close'")


class ManuscriptNumbers(Function):
    """The numbers a text states as written, a trailing dot or comma left out."""

    text: str

    expression = f"regexp_extract_all(text, '{_NUMBER}')"


class ManuscriptNamesRatio(Function):
    """Whether a text names a derived quantity, with a ratio word standing on its own."""

    text: str

    expression = f"regexp_matches(lower(text), '(^|{_BOUNDARY})({_RATIO_WORDS})({_BOUNDARY}|$)')"


class ManuscriptLastWords(Function):
    """The last `amount` whitespace-separated words of a text, joined by single spaces."""

    text: str
    amount: int

    expression = (
        f"list_transform([regexp_extract_all(text, '[^{SPACE_CHARS}]+')], lambda w: "
        f"array_to_string(list_slice(w, greatest(1, len(w) - amount + 1), len(w)), ' '))[1]"
    )


def _facts() -> str:
    return """
        SELECT fact_order, replace(fact_id, 'manuscriptfact:', 'manuscriptevidencefact:') AS fact_id,
               path, start_line, start_column, end_line, end_column, language, root
        FROM manuscript_facts
    """


def _events() -> str:
    """The body in reading order as what evidence reads: one row per sentence of running text or
    caption (a caption without its inline mathematics), per equation that states a quantity worth
    checking, per citation and cross reference, and per boundary that ends a group."""
    return rf"""
        WITH body AS (
          SELECT e.fact_id, e.reading_order, e.path, e.line, e.kind, e.a, e.b, e.section_number,
                 e.in_cells, e.in_proof,
                 coalesce(f.label, '') AS float_label,
                 CASE e.kind WHEN 'text' THEN e.a
                      WHEN 'caption' THEN array_to_string(
                        list_filter(str_split(e.a, '$'), lambda piece, i: i % 2 = 1), ' ') END
                   AS spoken
          FROM manuscript_element e
          LEFT JOIN manuscript_floats f
            ON f.fact_id = e.fact_id AND f.ordinal + 1 = e.float_number
          WHERE e.in_body
        ), tokens AS (
          SELECT fact_id, reading_order, unnest(parts) AS token,
                 generate_subscripts(parts, 1) AS position
          FROM (SELECT *, regexp_extract_all(spoken, '[^{SPACE_CHARS}]+|[{SPACE_CHARS}]+') AS parts
                FROM body WHERE spoken IS NOT NULL)
        ), cut AS (
          SELECT *, coalesce(sum(manuscript_ends_sentence(token)::INTEGER) OVER (
                       PARTITION BY fact_id, reading_order ORDER BY position
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS sub
          FROM tokens
        ), sentences AS (
          SELECT fact_id, reading_order, sub + 1 AS sub, string_agg(token, '' ORDER BY position) AS text
          FROM cut GROUP BY fact_id, reading_order, sub
          HAVING regexp_matches(string_agg(token, '' ORDER BY position), '[^{SPACE_CHARS}]')
        ), events AS (
          SELECT b.fact_id, b.reading_order, s.sub, 'sentence' AS kind, s.text,
                 regexp_matches(s.text, '[.!?][{SPACE_CHARS}]*$') AS terminated,
                 NULL AS raw, b.path, b.line, b.section_number, b.in_cells, b.in_proof,
                 b.float_label, NULL AS a, NULL AS b
          FROM sentences s JOIN body b USING (fact_id, reading_order)
          UNION ALL
          SELECT fact_id, reading_order, 0, 'math',
                 array_to_string(list_filter(manuscript_numbers(a),
                                             lambda x: contains(x, '.') OR length(x) > 2), ' '),
                 false, a, path, line, section_number, in_cells, in_proof, float_label, NULL, NULL
          FROM body WHERE kind = 'math'
          UNION ALL
          SELECT fact_id, reading_order, 0, kind, NULL, false, NULL, path, line, section_number,
                 in_cells, in_proof, float_label, a, b
          FROM body WHERE kind IN ('citation', 'reference')
          UNION ALL
          SELECT fact_id, reading_order, 0, 'boundary', NULL, false, NULL, path, line,
                 section_number, in_cells, in_proof, float_label, NULL, NULL
          FROM body WHERE kind IN ({", ".join(_REGION)})
        )
        SELECT *, kind = 'boundary' OR (kind = 'sentence' AND terminated) AS finish
        FROM events WHERE kind <> 'math' OR text <> ''
    """


def _sequenced() -> str:
    """Each event with its group, the paragraph-like block it sits in and the text of its group
    before, after and as a whole."""
    return f"""
        WITH ordered AS (
          SELECT *, row_number() OVER (PARTITION BY fact_id ORDER BY reading_order, sub) AS step,
                 CASE kind WHEN 'sentence' THEN text || CASE WHEN terminated THEN '' ELSE ' ' END
                      WHEN 'math' THEN text || ' ' ELSE '' END AS piece
          FROM manuscript_evidence_events
        ), grouped AS (
          SELECT *,
                 coalesce(sum(finish::INTEGER) OVER before, 0) AS sentence_order,
                 coalesce(sum((kind = 'boundary')::INTEGER) OVER before, 0) AS block
          FROM ordered
          WINDOW before AS (PARTITION BY fact_id ORDER BY step
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ), tailed AS (
          SELECT *,
                 coalesce(string_agg(piece, '') OVER (
                   PARTITION BY fact_id, sentence_order ORDER BY step
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), '') AS tail_before,
                 coalesce(string_agg(piece, '') OVER (
                   PARTITION BY fact_id, sentence_order ORDER BY step
                   ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING), '') AS tail_after,
                 string_agg(piece, '') OVER (
                   PARTITION BY fact_id, sentence_order ORDER BY step
                   ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS tail,
                 max(step) OVER (PARTITION BY fact_id, sentence_order) AS last_step
          FROM grouped
        ), tables AS (
          SELECT DISTINCT fact_id, name FROM manuscript_labels WHERE kind = 'table'
        ), signalled AS (
          SELECT t.*,
                 (t.kind = 'reference' AND tb.name IS NOT NULL)
                 OR (t.kind = 'math' AND manuscript_names_ratio(t.tail_before)
                     AND (contains(t.raw, '\\frac{{') OR contains(t.raw, '\\tfrac{{')))
                 OR (t.step = t.last_step AND manuscript_names_ratio(t.tail)
                     AND len(list_filter({list(_RATIO_DEFINITIONS)}::VARCHAR[],
                                         lambda phrase: contains(lower(t.tail), phrase))) > 0)
                   AS signal
          FROM tailed t LEFT JOIN tables tb ON tb.fact_id = t.fact_id AND tb.name = t.a
        )
        SELECT *, bool_or(signal) OVER (PARTITION BY fact_id, block ORDER BY step
                                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                    AS basis
        FROM signalled
    """


def _numbers() -> str:
    """Every number of every sentence and equation, with the company its group kept."""
    return rf"""
        WITH stated AS (
          SELECT *, manuscript_numbers(text) AS literals,
                 regexp_split_to_array(text, '{_NUMBER}') AS before_pieces
          FROM manuscript_evidence_sequenced WHERE kind IN ('sentence', 'math')
        ), listed AS (
          SELECT * EXCLUDE (literals, before_pieces), unnest(literals) AS literal,
                 generate_subscripts(literals, 1) AS position,
                 unnest(list_slice(before_pieces, 1, len(literals))) AS before_piece,
                 len(literals) AS stated_count
          FROM stated
        ), placed AS (
          SELECT *, (sum(length(before_piece) + length(literal)) OVER (
                      PARTITION BY fact_id, step ORDER BY position) - length(literal))::BIGINT
                      AS reached
          FROM listed
        ), counted AS (
          SELECT fact_id, sentence_order, sum(stated_count)::BIGINT AS sentence_number_count
          FROM (SELECT DISTINCT fact_id, sentence_order, step, stated_count FROM placed)
          GROUP BY ALL
        ), settled AS (
          SELECT fact_id, sentence_order, bool_or(basis) AS has_ratio_basis
          FROM manuscript_evidence_sequenced GROUP BY ALL
        ), measured AS (
          SELECT p.*, substr(p.text, 1, p.reached) AS before,
                 manuscript_last_words(p.tail_before || substr(p.text, 1, p.reached), 6) AS context,
                 regexp_replace(substr(p.piece, p.reached + length(p.literal) + 1) || p.tail_after,
                                '^[{SPACE_CHARS}]+', '') AS after
          FROM placed p
        )
        SELECT replace(m.fact_id, 'manuscriptfact:', 'manuscriptevidencefact:') AS fact_id,
               m.fact_id AS source_id, m.step, m.position, m.literal, m.reading_order, m.path,
               m.line, m.section_number, m.in_cells, m.float_label,
               (manuscript_names_ratio(m.context) AND NOT contains(m.context, 'confidence'))
                 OR starts_with(m.after, '%') OR starts_with(m.after, 'percent')
                 OR starts_with(m.after, 'per cent') AS names_ratio,
               c.sentence_number_count, m.sentence_order,
               regexp_matches(regexp_replace(regexp_replace(m.before, '-+$', ''),
                                             '[{SPACE_CHARS}]+$', ''), '(base|radix)$')
                 OR (m.kind = 'math' AND (m.in_proof OR (
                       len(list_filter({list(_EMPIRICAL)}::VARCHAR[],
                                       lambda word: contains(lower(m.tail_before), word))) = 0
                       AND regexp_matches(m.raw, '[=^<>]'))))
                 AS is_mathematical,
               s.has_ratio_basis
        FROM measured m
        JOIN counted c ON c.fact_id = m.fact_id AND c.sentence_order = m.sentence_order
        JOIN settled s ON s.fact_id = m.fact_id AND s.sentence_order = m.sentence_order
    """


def _citations() -> str:
    return """
        SELECT fact_id, fact_id || '/citations:' || ordinal AS record_id, ordinal, key, pin,
               reading_order, path, line, section_number, sentence_order
        FROM (SELECT replace(fact_id, 'manuscriptfact:', 'manuscriptevidencefact:') AS fact_id,
                     row_number() OVER (PARTITION BY fact_id ORDER BY step) - 1 AS ordinal,
                     a AS key, coalesce(b, '') AS pin, reading_order, path, line, section_number,
                     sentence_order
              FROM manuscript_evidence_sequenced WHERE kind = 'citation')
    """


def _numbers_final() -> str:
    return """
        SELECT fact_id, fact_id || '/numbers:' || ordinal AS record_id, ordinal, literal,
               reading_order, path, line, section_number, in_cells, float_label, names_ratio,
               sentence_number_count, sentence_order, is_mathematical, has_ratio_basis
        FROM (SELECT *, row_number() OVER (PARTITION BY fact_id ORDER BY step, position) - 1
                          AS ordinal
              FROM manuscript_evidence_stated)
    """


class ManuscriptEvidence(Family):
    family = "ManuscriptEvidenceFact"
    verified = True
    evidence = frozenset({"latex"})
    relations = {
        "facts": _facts(),
        "events": _events(),
        "sequenced": _sequenced(),
        "stated": _numbers(),
        "numbers": _numbers_final(),
        "citations": _citations(),
        "references": """
            SELECT replace(fact_id, 'manuscriptfact:', 'manuscriptevidencefact:') AS fact_id,
                   ordinal, target, command, reading_order, path, line, section_number
            FROM manuscript_references
        """,
    }
