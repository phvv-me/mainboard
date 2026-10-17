"""mcmr's ManuscriptNotationFact: the symbols, terms and notation index of a manuscript.

Every span of mathematics outside the notation index is read in order with the prose that comes
before and after it: its symbols are counted where they are used, and the spans that introduce a
symbol (a definition, a cue such as `let`, a role phrase) are its sites. The notation index (the
section titled `notation`) is read as rows: the symbols of a row's first cell and what the rest
of the row says.
"""

from ...catalog import Family, Function
from .manuscripts import SPACE_CHARS
from .manuscripts_reader import quoted
from .manuscripts_sites import caption_heads, heads, marks, occurrences, readings, sites

_TOKEN = r"\\[A-Za-z]+|\\|[^\\]+"
_LAST = 9223372036854775807
# The words a notation entry splits a symbol's senses with.
_SENSES = ("also", "elsewhere", "instead", "sense", "senses")


class ManuscriptExpanded(Function):
    """A span of mathematics with the macros the document declares replaced by what they stand
    for, three levels deep; `aliases` maps each macro (with its backslash) to its replacement."""

    text: str
    aliases: dict[str, str]

    expression = (
        "array_to_string(list_transform(regexp_extract_all("
        "array_to_string(list_transform(regexp_extract_all("
        "array_to_string(list_transform(regexp_extract_all(text, "
        f"{quoted(_TOKEN)}), lambda t: coalesce(map_extract_value(aliases, t), t)), ''), "
        f"{quoted(_TOKEN)}), lambda t: coalesce(map_extract_value(aliases, t), t)), ''), "
        f"{quoted(_TOKEN)}), lambda t: coalesce(map_extract_value(aliases, t), t)), '')"
    )


class ManuscriptWholeWordCount(Function):
    """How often a lowercase term stands as a word of its own in a lowercase text."""

    text: str
    term: str

    expression = (
        "len(list_filter(range(1, len(string_split(text, term))), lambda i: "
        "NOT regexp_matches(right(string_split(text, term)[i], 1), '[\\pL\\pN]') "
        "AND NOT regexp_matches(left(string_split(text, term)[i + 1], 1), '[\\pL\\pN]')))"
    )


def _facts() -> str:
    return """
        SELECT fact_order, replace(fact_id, 'manuscriptfact:', 'manuscriptnotationfact:') AS fact_id,
               path, start_line, start_column, end_line, end_column, language, root
        FROM manuscript_facts
    """


def _region() -> str:
    """Where the notation index is: from the first heading that says `notation` to the next
    heading at its own level or above."""
    return f"""
        SELECT fact_id, reading_order AS opened,
               coalesce((SELECT min(n.reading_order) FROM manuscript_sections n
                         WHERE n.fact_id = s.fact_id AND n.reading_order > s.reading_order
                           AND n.level <= s.level), {_LAST}) AS closed
        FROM manuscript_sections s WHERE contains(lower(title), 'notation')
        QUALIFY row_number() OVER (PARTITION BY fact_id ORDER BY reading_order) = 1
    """


def _aliases() -> str:
    return """
        SELECT fact_id, map_from_entries(list((name, replacement))) AS table
        FROM (SELECT fact_id, chr(92) || a AS name, arg_max(b, reading_order) AS replacement
              FROM manuscript_element WHERE kind = 'macro' GROUP BY ALL)
        GROUP BY fact_id
    """


def _live() -> str:
    """The body elements outside the notation index, each with the scope of the theorem or proof
    it sits in (a scope ends where anything outside a statement or proof begins, or a statement
    or proof environment opens or closes) and the flags that end a paragraph or a text run."""
    return """
        SELECT *, sum(clears::INTEGER) OVER (PARTITION BY fact_id ORDER BY reading_order) AS scope,
               kind IN ('break', 'rowbreak', 'section', 'close')
                 OR (kind = 'open' AND environment_role <> 'math') AS resets,
               kind NOT IN ('text', 'macro_use', 'emphasis') AS ends_run
        FROM (SELECT e.*, NOT (e.in_proof OR e.statement_number > 0)
                          OR (e.kind IN ('open', 'close') AND e.a IN ('proof', 'theorem', 'lemma',
                              'proposition', 'corollary', 'definition')) AS clears
              FROM manuscript_element e
              LEFT JOIN manuscript_notation_region r ON r.fact_id = e.fact_id
              WHERE e.in_body
                AND NOT coalesce(e.reading_order >= r.opened AND e.reading_order < r.closed, false))
    """


def _spans() -> str:
    """The spans of mathematics outside the notation index, each with the macros expanded, the
    text just before it in its paragraph, the text just after, and where it sits."""
    return """
        WITH blocked AS (
          SELECT *, coalesce(sum(resets::INTEGER) OVER (PARTITION BY fact_id ORDER BY reading_order
                       ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS block,
                    sum(ends_run::INTEGER) OVER (PARTITION BY fact_id ORDER BY reading_order
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run
          FROM manuscript_notation_live
        ), texted AS (
          SELECT *, last_value(CASE WHEN kind = 'text' THEN a END IGNORE NULLS) OVER (
                      PARTITION BY fact_id, block ORDER BY reading_order) AS tail,
                    coalesce(string_agg(CASE WHEN kind = 'text' THEN a END, '') OVER (
                      PARTITION BY fact_id, run ORDER BY reading_order
                      ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING), '') AS following
          FROM blocked
        )
        SELECT t.fact_id, t.reading_order, t.path, t.line, t.section_number, t.statement_number,
               coalesce(s.kind, '') AS statement_kind, t.in_proof, t.scope, t.n = 1 AS display,
               t.a AS math, manuscript_expanded(t.a, l.table) AS expanded,
               coalesce(t.tail, '') AS tail, manuscript_first_sentence(t.following) AS following,
               t.block
        FROM texted t
        LEFT JOIN manuscript_notation_aliases l ON l.fact_id = t.fact_id
        LEFT JOIN manuscript_statements s
          ON s.fact_id = t.fact_id AND s.ordinal + 1 = t.statement_number
        WHERE t.kind = 'math'
    """


def _captions() -> str:
    """The spans of mathematics a caption holds between dollar signs, each with its macros
    expanded and the line it starts on: mcmr reads them for the definitions they state, though
    not as uses."""
    return """
        SELECT c.fact_id, c.reading_order, c.path, c.section_number, r.span,
               c.line + length(array_to_string(list_slice(c.parts, 1, r.span - 1), ''))
                      - length(replace(array_to_string(list_slice(c.parts, 1, r.span - 1), ''),
                                       chr(10), '')) AS line,
               manuscript_expanded(c.parts[r.span], l.table) AS expanded
        FROM (SELECT fact_id, reading_order, path, line, section_number,
                     string_split(a, '$') AS parts
              FROM manuscript_notation_live WHERE kind = 'caption') c
        CROSS JOIN LATERAL (SELECT unnest(range(2, len(c.parts) + 1, 2)) AS span) r
        LEFT JOIN manuscript_notation_aliases l ON l.fact_id = c.fact_id
    """


def _proof_targets() -> str:
    """The statement each proof names right after it opens (`Proof of Theorem 2`), with the scope
    of the proof."""
    return """
        WITH proofs AS (
          SELECT fact_id, reading_order AS opened, scope FROM manuscript_notation_live
          WHERE kind = 'open' AND a = 'proof'
        ), blocks AS (
          SELECT p.fact_id, p.opened, p.scope, min(b.reading_order) AS block
          FROM proofs p
          LEFT JOIN manuscript_element b
            ON b.fact_id = p.fact_id AND b.reading_order > p.opened
           AND b.kind IN ('text', 'math', 'break', 'close')
          GROUP BY ALL
        ), named AS (
          SELECT b.fact_id, b.scope, arg_min(r.a, r.reading_order) AS label
          FROM blocks b
          JOIN manuscript_element r
            ON r.fact_id = b.fact_id AND r.kind = 'reference' AND r.reading_order > b.opened
           AND (b.block IS NULL OR r.reading_order < b.block)
          GROUP BY b.fact_id, b.opened, b.scope
        )
        SELECT n.fact_id, n.scope, arg_min(s.ordinal + 1, s.ordinal) AS statement_number
        FROM named n
        JOIN manuscript_statements s ON s.fact_id = n.fact_id AND s.label = n.label
        GROUP BY n.fact_id, n.scope
    """


def _index_items() -> str:
    """The elements of the notation index that make up its rows, each with the row it belongs to,
    whether the row has started, the cell it sits in, and what it adds to the row's meaning. A
    heading item, a row break or an environment end closes the row before it, and only content in
    a cell starts a row: the tabular's own closing element sits in a cell but starts none."""
    return """
        WITH items AS (
          SELECT e.fact_id, e.reading_order, e.path, e.line, e.kind, e.a, e.in_cells,
                 kind IN ('item_label', 'rowbreak', 'close') AS flushes
          FROM manuscript_element e
          JOIN manuscript_notation_region r
            ON r.fact_id = e.fact_id AND e.reading_order >= r.opened AND e.reading_order < r.closed
          WHERE e.in_body AND e.kind IN ('item_label', 'math', 'text', 'rowbreak', 'close')
        ), rowed AS (
          SELECT *, sum(flushes::INTEGER) OVER (PARTITION BY fact_id ORDER BY reading_order)
                      AS row_id
          FROM items
        ), started AS (
          SELECT *, NOT flushes OR kind = 'item_label' AS content,
                 bool_or(in_cells AND NOT flushes OR kind = 'item_label') OVER (
                   PARTITION BY fact_id, row_id ORDER BY reading_order
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS open
          FROM rowed
        ), kept AS (
          SELECT *, content AND open AS added,
                 CASE kind WHEN 'item_label' THEN 1
                      WHEN 'text' THEN length(a) - length(replace(a, '&', '')) ELSE 0 END AS cells
          FROM started
        )
        SELECT *, coalesce(sum(CASE WHEN added THEN cells ELSE 0 END) OVER (
                             PARTITION BY fact_id, row_id ORDER BY reading_order
                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS cell
        FROM kept
    """


def _item_texts() -> str:
    """The spans of mathematics the index names symbols in: its math, and the math between the
    dollar signs of each item label."""
    return """
        SELECT t.fact_id, t.reading_order, t.piece, manuscript_expanded(t.math, a.table) AS expanded
        FROM (SELECT fact_id, reading_order, 0 AS piece, a AS math
              FROM manuscript_notation_index_items WHERE kind = 'math'
              UNION ALL
              SELECT fact_id, reading_order, i AS piece, p AS math
              FROM (SELECT fact_id, reading_order, unnest(pieces) AS p,
                           generate_subscripts(pieces, 1) AS i
                    FROM (SELECT fact_id, reading_order,
                                 list_filter(str_split(a, '$'), lambda x, i: i % 2 = 0) AS pieces
                          FROM manuscript_notation_index_items WHERE kind = 'item_label'))) t
        LEFT JOIN manuscript_notation_aliases a ON a.fact_id = t.fact_id
    """


def _uses() -> str:
    """Every occurrence of a symbol in a span, in the index or in a head text, in order."""
    return """
        SELECT fact_id, reading_order, piece, unnest(names) AS name,
               generate_subscripts(names, 1) AS position
        FROM (SELECT *, manuscript_symbols(expanded) AS names
              FROM (SELECT fact_id, reading_order, 0 AS piece, expanded
                    FROM manuscript_notation_spans
                    UNION ALL
                    SELECT fact_id, reading_order, piece, expanded
                    FROM manuscript_notation_item_texts
                    UNION ALL
                    SELECT fact_id, reading_order, piece, text
                    FROM manuscript_notation_heads
                    UNION ALL
                    SELECT fact_id, reading_order, piece, text
                    FROM manuscript_notation_caption_heads))
    """


def _symbols() -> str:
    return """
        WITH flagged AS (
          SELECT *, NOT bound_hit AND NOT local_here AS free,
                 section_number <> coalesce(lag(section_number) OVER met, 0) AS entered,
                 NOT bound_hit AND NOT local_here
                   AND section_number <> coalesce(lag(section_number) OVER freed, 0)
                   AS entered_free
          FROM manuscript_notation_occurrences
          WINDOW met AS (PARTITION BY fact_id, name ORDER BY reading_order, position),
                 freed AS (PARTITION BY fact_id, name, NOT bound_hit AND NOT local_here
                           ORDER BY reading_order, position)
        )
        SELECT replace(fact_id, 'manuscriptfact:', 'manuscriptnotationfact:') AS fact_id, name,
               coalesce(arg_min(path, reading_order) FILTER (free), arg_min(path, reading_order))
                 AS path,
               coalesce(arg_min(line, reading_order) FILTER (free), arg_min(line, reading_order))
                 AS line,
               min(reading_order) AS first_order, count(*) AS use_count,
               count(*) FILTER (entered) AS section_count,
               count(*) FILTER (free) AS free_use_count,
               count(*) FILTER (entered_free) AS free_section_count,
               coalesce(min(reading_order) FILTER (free), 0) AS first_free_order
        FROM flagged GROUP BY fact_id, name
    """


def _terms() -> str:
    """Each phrase the author marked once, lowered; how often the body states it as a word, and
    where it first does."""
    return f"""
        WITH marked AS (
          SELECT e.fact_id, e.reading_order AS mark_order, e.path, e.line, e.section_number,
                 e.b AS command, term
          FROM (SELECT *, lower(rtrim(trim(a, ' ' || chr(9) || chr(10) || chr(13)), '.')) AS term
                FROM manuscript_element WHERE kind = 'emphasis' AND in_body) e
          LEFT JOIN manuscript_notation_region r ON r.fact_id = e.fact_id
          WHERE NOT coalesce(e.reading_order >= r.opened AND e.reading_order < r.closed, false)
            AND strlen(term) >= 4
            AND len(regexp_extract_all(term, '[^{SPACE_CHARS}]+')) <= 4
            AND NOT regexp_matches(term, '[$\\\\{{}}0-9]')
          QUALIFY row_number() OVER (PARTITION BY e.fact_id, term ORDER BY e.reading_order) = 1
        ), used AS (
          SELECT m.fact_id, m.term, t.reading_order,
                 manuscript_whole_word_count(lower(t.a), m.term) AS seen
          FROM marked m JOIN manuscript_element t
            ON t.fact_id = m.fact_id AND t.kind = 'text' AND t.in_body
               AND contains(lower(t.a), m.term)
        )
        SELECT replace(m.fact_id, 'manuscriptfact:', 'manuscriptnotationfact:') AS fact_id,
               m.term, m.command, m.mark_order, m.path, m.line, m.section_number,
               coalesce(sum(u.seen) FILTER (u.seen > 0), 0)::BIGINT AS use_count,
               coalesce(min(u.reading_order) FILTER (u.seen > 0), 0) AS first_use_order
        FROM marked m LEFT JOIN used u ON u.fact_id = m.fact_id AND u.term = m.term
        GROUP BY ALL
    """


def _entries() -> str:
    """One entry per symbol a row of the index names: the row's meaning, kept from its second
    cell, and the senses it separates."""
    return f"""
        WITH named AS (
          SELECT i.fact_id, i.row_id, i.reading_order, i.cell, i.kind, i.a,
                 coalesce(list(u.name ORDER BY u.piece, u.position) FILTER (u.name IS NOT NULL), []) AS names
          FROM manuscript_notation_index_items i
          LEFT JOIN manuscript_notation_uses u
            ON u.fact_id = i.fact_id AND u.reading_order = i.reading_order
          WHERE i.added
          GROUP BY ALL
        ), rows AS (
          SELECT fact_id, row_id,
                 flatten(list(names ORDER BY reading_order)) AS names,
                 trim(string_agg(CASE kind
                        WHEN 'math' THEN CASE WHEN cell > 0 THEN ' · ' ELSE '' END
                        WHEN 'text' THEN CASE WHEN cell > 0 THEN replace(a, '&', '')
                             WHEN contains(a, '&') THEN replace(substr(a, strpos(a, '&') + 1), '&', '')
                             ELSE '' END
                        ELSE '' END, '' ORDER BY reading_order),
                      ' ' || chr(9) || chr(10) || chr(13)) AS meaning
          FROM named GROUP BY fact_id, row_id
        ), places AS (
          SELECT r.fact_id, r.row_id,
                 coalesce(f.path, a.path, z.path) AS path, coalesce(f.line, a.line, z.line) AS line
          FROM rows r
          LEFT JOIN manuscript_notation_index_items f
            ON f.fact_id = r.fact_id AND f.row_id = r.row_id + 1 AND f.flushes
          LEFT JOIN manuscript_notation_region g ON g.fact_id = r.fact_id
          LEFT JOIN LATERAL (SELECT path, line FROM manuscript_element e
                             WHERE e.fact_id = r.fact_id AND e.in_body
                               AND e.reading_order >= g.closed
                             ORDER BY reading_order LIMIT 1) a ON true
          LEFT JOIN LATERAL (SELECT path, line FROM manuscript_element e
                             WHERE e.fact_id = r.fact_id ORDER BY reading_order DESC LIMIT 1) z
            ON true
        ), listed AS (
          SELECT replace(r.fact_id, 'manuscriptfact:', 'manuscriptnotationfact:') AS fact_id,
                 symbol, r.meaning,
                 len(list_filter({list(_SENSES)}::VARCHAR[],
                                 lambda w: contains(lower(r.meaning), w))) + 1 AS sense_count,
                 p.path, p.line
          FROM (SELECT *, unnest(list_filter(names, lambda x, i: i = 1 OR x <> names[i - 1]))
                            AS symbol FROM rows) r
          JOIN places p ON p.fact_id = r.fact_id AND p.row_id = r.row_id
        ), formatted AS (
          SELECT e.fact_id, s.name AS symbol, e.meaning, e.sense_count, e.path, e.line
          FROM listed e
          JOIN manuscript_notation_symbols s
            ON s.fact_id = e.fact_id AND starts_with(s.name, e.symbol)
           AND regexp_matches(substr(s.name, length(e.symbol) + 1), '^_\\{{\\\\mathbb\\{{.*\\}}\\}}$')
          WHERE contains(lower(e.meaning), 'a format subscript is used')
        ), covered AS (
          SELECT * FROM listed UNION ALL SELECT * FROM formatted
        ), componented AS (
          SELECT e.fact_id, s.name AS symbol, e.meaning, e.sense_count, e.path, e.line
          FROM covered e
          JOIN manuscript_notation_symbols s
            ON s.fact_id = e.fact_id
           AND regexp_extract(s.name, '^(.*)_[^_]*$', 1) = manuscript_index_base(e.symbol)
           AND regexp_matches(regexp_extract(s.name, '^.*_([^_]*)$', 1), '^[A-Za-z0-9]$')
          WHERE contains(lower(e.meaning), 'with scalar entries')
            AND EXISTS (SELECT 1 FROM covered v
                        WHERE v.fact_id = e.fact_id AND v.path = e.path AND v.line = e.line
                          AND v.symbol = chr(92) || 'mathbf{{' || manuscript_index_base(e.symbol)
                                         || '}}')
        )
        SELECT * FROM covered UNION ALL SELECT * FROM componented
    """


class ManuscriptNotation(Family):
    family = "ManuscriptNotationFact"
    verified = True
    evidence = frozenset({"latex"})
    relations = {
        "facts": _facts(),
        "region": _region(),
        "aliases": _aliases(),
        "live": _live(),
        "spans": _spans(),
        "proof_targets": _proof_targets(),
        "terms": _terms(),
        "index_items": _index_items(),
        "item_texts": _item_texts(),
        "heads": heads(),
        "captions": _captions(),
        "caption_heads": caption_heads(),
        "uses": _uses(),
        "readings": readings(),
        "marks": marks(),
        "occurrences": occurrences(),
        "symbols": _symbols(),
        "sites": sites(),
        "entries": _entries(),
    }
