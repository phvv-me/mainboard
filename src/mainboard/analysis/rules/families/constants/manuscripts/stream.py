"""The reading-order element stream of every LaTeX manuscript, as one SQL statement.

Stages, each a common table expression: the atoms of every LaTeX file; their kinds; the parts each
atom contributes (prose to append, elements to push), the first three in `atoms.py`; the elements
of each file, with its runs of prose flushed where mcmr's reader flushes them; the files each root
splices in; the flat stream in reading order; and the position of every element, from window
functions over that order.
"""

import re

from .atoms import ATOMS, WORD, atom_parts, first_group
from .reader import ARGUED_ONE, ARGUED_TWO, BRACKET, GROUP, INSIDE, SECTIONS
from .sql import quoted, role, strip

_STYLE = quoted(rf"^\\newtheoremstyle\*?{GROUP}(?:{GROUP}){{2}}\{{({INSIDE})\}}")

_AFTER_NAME = quoted(rf"^\\[A-Za-z]+\*?\s*(?:{GROUP}|\\[A-Za-z@]+)")

_BODY = quoted(rf"(?s)^\s*(?:{BRACKET}\s*)*\{{({INSIDE})\}}")

_OPTIONAL_GROUPS = quoted(rf"^{WORD}((?:{BRACKET}){{0,2}})\{{")

_HEADING = quoted(rf"^\\({'|'.join(SECTIONS)})\*?(?:{BRACKET})?\{{({INSIDE})\}}")

_PEEK = quoted(rf"^\{{({INSIDE})\}}")

_PEEKS = quoted(rf"^\{{{INSIDE}\}}")

_ARGUMENTS = f"""
    , styled AS (
      SELECT *, last_value(CASE WHEN kind = 'theoremstyle' THEN
                    {strip(first_group())} END IGNORE NULLS)
                  OVER (PARTITION BY path ORDER BY pos
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS style
      FROM typed
    ), italics AS (
      SELECT path, pos, {strip(f"regexp_extract(atom, {quoted(rf'^{WORD}\{{({INSIDE})\}}')}, 1)")} AS name,
             regexp_matches(regexp_extract(atom, {_STYLE}, 1), '\\\\(?:itshape|slshape)') AS italic
      FROM typed WHERE kind = 'newtheoremstyle'
    ), theorems AS (
      SELECT t.*, t.style IS NULL OR t.style = 'plain' OR coalesce(d.italic, false) AS owes_proof
      FROM styled t
      LEFT JOIN (SELECT t.path, t.pos, bool_or(i.italic) AS italic
                 FROM styled t JOIN italics i ON i.path = t.path AND i.name = t.style AND i.pos < t.pos
                 WHERE t.kind = 'theorem' GROUP BY t.path, t.pos) d
        ON d.path = t.path AND d.pos = t.pos
      WHERE t.kind = 'theorem'
    ), macros AS (
      SELECT *, CASE WHEN has_arguments THEN '\\text'
                     WHEN name = 'DeclareMathOperator' THEN '\\operatorname{{' || body || '}}'
                     ELSE body END AS replacement,
             regexp_extract({strip("body")}, {_HEADING}, 1) AS heading_command,
             regexp_extract({strip("body")}, {_HEADING}, 2) AS heading_title,
             regexp_matches(body, '^\\s*\\\\(?:text|textrm|textnormal|texttt|mbox)\\{{') AS textual
      FROM (SELECT *, regexp_matches(rest, '^\\s*\\[') AS has_arguments,
                   regexp_extract(rest, {_BODY}, 1) AS body
            FROM (SELECT *,
                         CASE WHEN regexp_matches(atom, {quoted(rf"^{WORD}\s*\{{")})
                              THEN ltrim({strip(first_group(rf"{WORD}\s*"))}, chr(92))
                              ELSE regexp_extract(atom, {quoted(rf"^{WORD}\s*\\([A-Za-z@]+)")}, 1)
                              END AS declared,
                         regexp_replace(atom, {_AFTER_NAME}, '') AS rest
                  FROM typed WHERE kind = 'macro'))
    ), raw_labels AS (
      SELECT path, pos, line, {strip("raw")} AS name, i
      FROM (SELECT path, pos, line, unnest(names) AS raw, generate_subscripts(names, 1) AS i
            FROM (SELECT *, regexp_extract_all(atom,
                      {quoted(rf"\\label\{{({INSIDE})\}}")}, 1) AS names
                  FROM typed WHERE kind IN ('raw_math', 'raw_verbatim')))
    ), targets AS (
      SELECT path, pos, line, name, {strip("raw")} AS target, i
      FROM (SELECT path, pos, line, name, unnest(pieces) AS raw,
                   generate_subscripts(pieces, 1) AS i
            FROM (SELECT *, string_split({first_group(rf"{WORD}(?:{BRACKET})?")}, ',') AS pieces
                  FROM typed WHERE kind = 'reference'))
    ), sources AS (
      SELECT path, pos, line, {strip("raw")} AS target, pin, i
      FROM (SELECT path, pos, line, pin, unnest(pieces) AS raw,
                   generate_subscripts(pieces, 1) AS i
            FROM (SELECT t.*, string_split({first_group(rf"{WORD}(?:{BRACKET}){{0,2}}")}, ',') AS pieces,
                         coalesce({strip("substr(brackets[2], 2, length(brackets[2]) - 2)")},
                                  {strip("substr(brackets[1], 2, length(brackets[1]) - 2)")}, '') AS pin
                  FROM typed t,
                       LATERAL (SELECT regexp_extract_all(
                                  regexp_extract(t.atom, {_OPTIONAL_GROUPS}, 1),
                                  {quoted(BRACKET)}) AS brackets) b
                  WHERE t.kind = 'citation'))
    ), emphasized AS (
      SELECT t.path, t.pos, t.line, t.name, t.kind,
             {strip(f"regexp_extract(substr(l.content, t.start_offset + t.size + 1, 4000), {_PEEK}, 1)")}
               AS marked
      FROM typed t JOIN latex l ON l.path = t.path
      WHERE t.kind = 'emphasis'
        AND regexp_matches(substr(l.content, t.start_offset + t.size + 1, 4000), {_PEEKS})
    )
"""

_RUNS = """
    , ordered AS (
      SELECT *, row_number() OVER (PARTITION BY path ORDER BY pos, sub) AS seq FROM all_parts
    ), counted AS (
      SELECT *, coalesce(sum(pushes::INTEGER) OVER (PARTITION BY path ORDER BY seq
                  ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS before
      FROM ordered
    ), lined AS (
      SELECT *, CASE WHEN NOT pushes AND is_word
                          AND NOT coalesce(bool_or(txt IS NOT NULL AND regexp_matches(txt, '\\S'))
                                OVER (PARTITION BY path, before ORDER BY seq
                                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), false)
                     THEN line END AS blank_line
      FROM counted
    ), carried AS (
      SELECT *, last_value(blank_line IGNORE NULLS) OVER (PARTITION BY path ORDER BY seq
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS text_line
      FROM lined
    ), pieces AS (
      SELECT path, 2 * (before + 1) AS key, elem AS kind, a, b, n, line
      FROM carried WHERE pushes
      UNION ALL
      SELECT path, 2 * before + 1, 'text', string_agg(txt, '' ORDER BY seq), NULL, NULL,
             coalesce(max_by(text_line, seq), 1)
      FROM carried WHERE NOT pushes
      GROUP BY path, before
      HAVING regexp_matches(string_agg(txt, '' ORDER BY seq), '\\S')
    ), file_elements AS (
      SELECT *, row_number() OVER (PARTITION BY path ORDER BY key) AS idx FROM pieces
    )
"""

_NAMED = r"regexp_replace(a, '^(\./)+', '')"

_SPLICE = f"""
    , wanted AS (
      SELECT path, idx, candidate, i AS priority
      FROM (SELECT path, idx, unnest(candidates) AS candidate,
                   generate_subscripts(candidates, 1) AS i
            FROM (SELECT path, idx, [joined, joined || '.tex', named, named || '.tex'] AS candidates
                  FROM (SELECT path, idx, named,
                               CASE WHEN directory = '' THEN named
                                    ELSE directory || '/' || named END AS joined
                        FROM (SELECT path, idx, {_NAMED} AS named,
                                     regexp_extract(path, '^(.*)/[^/]*$', 1) AS directory
                              FROM file_elements WHERE kind = 'include'))))
    ), includes AS (
      SELECT w.path, w.idx, arg_min(w.candidate, w.priority) AS target
      FROM wanted w JOIN latex l ON l.path = w.candidate
      GROUP BY w.path, w.idx
    ), roots AS (
      SELECT path AS root FROM latex WHERE contains(content, chr(92) || 'documentclass')
    ), tree(root, path, key, trail) AS (
      SELECT root, root, []::BIGINT[], [root] FROM roots
      UNION ALL
      SELECT t.root, i.target, t.key || [i.idx], t.trail || [i.target]
      FROM tree t JOIN includes i ON i.path = t.path
      WHERE NOT list_contains(t.trail, i.target)
    ), visited AS (
      SELECT * FROM tree
      QUALIFY row_number() OVER (PARTITION BY root, path ORDER BY key) = 1
    ), numbered AS (
      SELECT v.root, e.path, e.kind, e.a, e.b, e.n, e.line,
             row_number() OVER (PARTITION BY v.root ORDER BY v.key || [e.idx]) - 1 AS reading_order
      FROM visited v JOIN file_elements e ON e.path = v.path
      WHERE e.kind <> 'include'
    )
"""

_HEADINGS = """
    , defined AS (
      SELECT root, a AS name, b AS title, n // 2 AS level, n % 2 = 1 AS replaces, reading_order
      FROM numbered WHERE kind = 'heading_macro'
    ), timeline AS (
      SELECT root, name, reading_order,
             coalesce(last_value(CASE WHEN replaces THEN title END IGNORE NULLS) OVER seen,
                      first_value(title) OVER seen) AS title,
             coalesce(last_value(CASE WHEN replaces THEN level END IGNORE NULLS) OVER seen,
                      first_value(level) OVER seen) AS level
      FROM defined
      WINDOW seen AS (PARTITION BY root, name ORDER BY reading_order
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ), resolved AS (
      SELECT u.root, u.path, u.line, u.reading_order,
             CASE WHEN t.title IS NULL THEN u.kind ELSE 'section' END AS kind,
             CASE WHEN t.title IS NULL THEN u.a ELSE t.title END AS a, u.b,
             CASE WHEN t.title IS NULL THEN u.n ELSE t.level END AS n
      FROM (SELECT *, CASE WHEN kind = 'macro_use' THEN a END AS used FROM numbered) u
      ASOF LEFT JOIN timeline t
        ON u.root = t.root AND u.used = t.name AND u.reading_order >= t.reading_order
    )
"""

_FLOATS = "('figure', 'table', 'algorithm')"

_CLOSERS = "('cells', 'math', 'proof', 'bibliography', 'figure', 'table', 'algorithm')"


def _toggle(name: str, *, on: str, off: str) -> str:
    """A flag a window sets at its `on` elements and clears just after its `off` elements."""
    return (
        f"coalesce(last_value(CASE WHEN {on} THEN true WHEN prev_off_{name} THEN false END "
        f"IGNORE NULLS) OVER seen, false) AS in_{name}"
    )


_POSITIONS = f"""
    , declared AS (
      SELECT root, a AS name, arg_max(n, reading_order) AS owes_proof
      FROM resolved WHERE kind = 'statement_kind' GROUP BY root, a
    ), nest AS (
      SELECT *, sum(CASE kind WHEN 'open' THEN 1 WHEN 'close' THEN -1 ELSE 0 END) OVER seen
                  AS depth
      FROM resolved
      WINDOW seen AS (PARTITION BY root ORDER BY reading_order
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ), cell_kinds AS (
      SELECT DISTINCT root, name
      FROM (SELECT e.root, o.reading_order AS opened, o.a AS name
            FROM (SELECT * FROM nest WHERE kind IN ('text', 'rowbreak')) e
            ASOF JOIN (SELECT * FROM nest WHERE kind = 'open') o
              ON e.root = o.root AND e.depth = o.depth AND e.reading_order >= o.reading_order
            GROUP BY e.root, o.reading_order, o.a
            HAVING bool_or(e.kind = 'text' AND contains(e.a, '&')) AND bool_or(e.kind = 'rowbreak'))
    ), tagged AS (
      SELECT n.*, {role("n.a")} AS environment_role, d.name IS NOT NULL AS declared,
             c.name IS NOT NULL AS sets_cells
      FROM resolved n
      LEFT JOIN declared d ON d.root = n.root AND d.name = rtrim(n.a, '*')
                          AND n.kind IN ('open', 'close')
      LEFT JOIN cell_kinds c ON c.root = n.root AND c.name = n.a AND n.kind IN ('open', 'close')
    ), signals AS (
      SELECT *,
             kind = 'open' AND (environment_role = 'cells' OR sets_cells) AS on_cells,
             kind = 'close' AND (environment_role = 'cells' OR sets_cells) AS off_cells,
             kind = 'open' AND environment_role = 'math' AS on_math,
             kind = 'close' AND environment_role = 'math' AS off_math,
             kind = 'open' AND environment_role = 'proof' AS on_proof,
             kind = 'close' AND environment_role = 'proof' AS off_proof,
             kind = 'open' AND environment_role = 'bibliography' AS on_bibliography,
             kind = 'close' AND environment_role = 'bibliography' AS off_bibliography,
             kind = 'open' AND declared AS on_statement,
             kind = 'close' AND declared AND environment_role NOT IN {_CLOSERS}
               AS off_statement,
             kind = 'open' AND environment_role IN {_FLOATS} AS on_float,
             kind = 'close' AND environment_role IN {_FLOATS} AS off_float
      FROM tagged
    ), lagged AS (
      SELECT *, bool_or(kind = 'bodystart') OVER seen AS in_body,
             sum(on_statement::INTEGER) OVER seen AS statements_so_far,
             sum(on_float::INTEGER) OVER seen AS floats_so_far,
             lag(off_cells, 1, false) OVER walk AS prev_off_cells,
             lag(off_math, 1, false) OVER walk AS prev_off_math,
             lag(off_proof, 1, false) OVER walk AS prev_off_proof,
             lag(off_bibliography, 1, false) OVER walk AS prev_off_bibliography,
             lag(off_statement, 1, false) OVER walk AS prev_off_statement,
             lag(off_float, 1, false) OVER walk AS prev_off_float
      FROM signals
      WINDOW walk AS (PARTITION BY root ORDER BY reading_order),
             seen AS (PARTITION BY root ORDER BY reading_order
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ), state AS (
      SELECT 'manuscriptfact:' || root AS fact_id, root, reading_order, path, line, kind, a, b, n,
             in_body,
             sum((kind = 'section' AND in_body)::INTEGER) OVER seen AS section_number,
             coalesce(last_value(CASE WHEN on_statement THEN statements_so_far
                                      WHEN prev_off_statement THEN 0 END IGNORE NULLS) OVER seen,
                      0) AS statement_number,
             coalesce(last_value(CASE WHEN on_float THEN floats_so_far
                                      WHEN prev_off_float THEN 0 END IGNORE NULLS) OVER seen,
                      0) AS float_number,
             {_toggle("cells", on="on_cells", off="off_cells")},
             {_toggle("math", on="on_math", off="off_math")},
             {_toggle("proof", on="on_proof", off="off_proof")},
             {_toggle("bibliography", on="on_bibliography", off="off_bibliography")},
             environment_role, declared
      FROM lagged
      WINDOW seen AS (PARTITION BY root ORDER BY reading_order
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    )
"""

_PASS = f"""{ATOMS} {_ARGUMENTS}
    , parts AS (
      {atom_parts()}
    )
"""

# What a begin atom hands to `nested`: the bodies of its optional groups and of the mandatory ones
# an argued environment takes, those that spell a command.
_NESTED = f"""
    , nested_sources AS (
      SELECT path || '#' || pos || '.' || i AS key, path AS parent, pos AS parent_pos,
             line AS parent_line, i AS k, body AS content
      FROM (SELECT path, pos, line, unnest(bodies) AS body, generate_subscripts(bodies, 1) AS i
            FROM (SELECT path, pos, line, list_concat(brackets, groups) AS bodies
                  FROM (SELECT t.path, t.pos, t.line,
                               list_transform(regexp_extract_all(
                                                regexp_extract(rest, {quoted(f"^(?:{BRACKET})*")}),
                                                {quoted(BRACKET)}),
                                              lambda x: substr(x, 2, length(x) - 2)) AS brackets,
                               list_transform(CASE WHEN rtrim(t.environment, '*') IN
                                    ({", ".join(map(quoted, ARGUED_ONE + ARGUED_TWO))})
                                 THEN regexp_extract_all(regexp_replace(rest, {quoted(BRACKET)}, '', 'g'),
                                                         {quoted(GROUP)})
                                 ELSE []::VARCHAR[] END,
                                              lambda x: substr(x, 2, length(x) - 2)) AS groups
                        FROM typed t,
                             LATERAL (SELECT regexp_replace(t.atom,
                                        {quoted(r"^\\begin\*?\{[^{}]*\}")}, '') AS rest) r
                        WHERE t.kind IN ('begin', 'raw_math', 'raw_verbatim'))))
      WHERE contains(body, chr(92))
    )
"""

_NAMES = (
    "latex|cut|measured|atoms|typed|styled|italics|theorems|macros|raw_labels|targets|sources|"
    "emphasized|parts"
)


def _nested(sql: str) -> str:
    """The chain again under `n_` names, outside quoted text, for the bodies of `_NESTED`."""
    return re.sub(
        rf"('(?:[^']|'')*')|\b({_NAMES})\b",
        lambda match: match.group(1) or f"n_{match.group(2)}",
        sql,
    )


_MERGED = """
    , all_parts AS (
      SELECT * FROM parts
      UNION ALL
      SELECT s.parent, s.parent_pos, 10 + row_number() OVER (
               PARTITION BY s.parent, s.parent_pos ORDER BY s.k, p.pos, p.sub), p.pushes, p.elem,
             p.a, p.b, p.n, s.parent_line, NULL, false
      FROM n_parts p JOIN nested_sources s ON s.key = p.path
      WHERE p.pushes AND p.elem NOT IN ('break', 'rowbreak')
    )
"""

ELEMENTS = f"""
    WITH RECURSIVE latex AS (
      SELECT f.path, s.content
      FROM ast_files f JOIN source_files s ON s.path = f.path
      WHERE f.language = 'latex'
    ), {_PASS} {_NESTED}
    , n_latex AS (SELECT key AS path, content FROM nested_sources)
    , {_nested(_PASS)}
    {_MERGED} {_RUNS} {_SPLICE} {_HEADINGS} {_POSITIONS}
    SELECT * FROM state
"""
