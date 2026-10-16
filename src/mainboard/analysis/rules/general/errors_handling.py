"""Rules over how guarded regions answer the failures they catch (mcmr's ALL-ERRO handling)."""

import re

from ...catalog import Example, Relation, Rule
from ...facts import Setting

_WORD = "[A-Za-z_][A-Za-z0-9_]*"
_SPACE = r"[\s\p{Z}]"


class ErrorsGuardLines(Relation):
    """The `errors_guard_lines()` relation."""

    query = f"""
    SELECT *, length(line) - length(regexp_replace(line, '^{_SPACE}+', '')) AS indent,
           regexp_replace(line, '^{_SPACE}+|{_SPACE}+$', '', 'g') AS stripped
    FROM (
      SELECT g.* EXCLUDE (lines), unnest(g.lines) AS line,
             generate_subscripts(g.lines, 1) - 1 AS "offset"
      FROM (
        SELECT n.fact_order, f.language, n.ordinal AS guard, n.subtree_end AS guard_end,
               n.depth AS guard_depth, n.path, n.start_line, n.start_column, n.end_line,
               n.path || ':' || n.start_line
                 || CASE WHEN n.end_line > n.start_line THEN '-' || n.end_line ELSE '' END
                 AS location,
               str_split(decode(array_slice(encode(f.source), n.byte_start + 1,
                                            n.byte_start + n.byte_length)), chr(10)) AS lines
        FROM syntax_nodes n JOIN syntax_facts f USING (fact_order)
        WHERE n.kind = 'guard' AND f.kind = 'callable') g)
"""


def _opening(handlers: tuple[str, ...]) -> str:
    return "^[}\\t ]*(?:" + "|".join(map(re.escape, handlers)) + ")\\b"


class SwallowedError(Rule, id="ALL-ERRO0001", applicability="review"):
    """Failures a declaration catches and then answers with nothing."""

    handlers: tuple[str, ...] = ("except", "catch", "rescue")
    inert: tuple[str, ...] = ("pass", "continue", "...")
    discard: str = "_"
    failures_as_values: tuple[str, ...] = ("rust", "go")

    query = r"""
        WITH lines AS MATERIALIZED (SELECT * FROM errors_guard_lines()),
        clauses AS (
          SELECT fact_order, guard, path, start_line, start_column, location,
                 "offset" AS handler_offset, indent AS margin,
                 regexp_extract(line, '.*(?:\{|:)(.*)$', 1) AS inline
          FROM lines
          WHERE regexp_matches(line, $opening) AND (language <> 'python' OR indent = start_column)
        ), later AS (
          SELECT c.fact_order, c.guard, c.handler_offset, l."offset" AS line_offset, l.line,
                 NOT (l.indent > c.margin OR l.stripped IN ('{', '')) AS leaves
          FROM clauses c JOIN lines l ON l.fact_order = c.fact_order AND l.guard = c.guard
           AND l."offset" > c.handler_offset
        ), statements AS (
          SELECT fact_order, guard, handler_offset, inline AS statement FROM clauses
          UNION ALL
          SELECT fact_order, guard, handler_offset, line FROM (
            SELECT *, min(line_offset) FILTER (WHERE leaves)
                      OVER (PARTITION BY fact_order, guard, handler_offset) AS boundary
            FROM later)
          WHERE boundary IS NULL OR line_offset < boundary
        ), silent AS (
          SELECT fact_order, guard, handler_offset FROM (
            SELECT *, trim(str_split(str_split(statement, '#')[1], '//')[1], $trimmed) AS cleaned
            FROM statements)
          GROUP BY ALL
          HAVING NOT bool_or(cleaned <> '' AND NOT list_contains($inert::VARCHAR[], cleaned))
        ), queue_reads AS (
          SELECT r.statement_span_path AS path, r.statement_span_start_line AS start_line,
                 r.statement_span_start_column AS start_column
          FROM try_block_regions r JOIN (
            SELECT parent_id, count(*) AS handlers,
                   first(caught_import ORDER BY ordinal) AS import,
                   first(caught_is_tuple ORDER BY ordinal) AS is_tuple
            FROM try_block_regions_handlers GROUP BY parent_id) h ON h.parent_id = r.record_id
          WHERE h.handlers = 1 AND h.import = 'queue.Empty' AND NOT h.is_tuple
            AND NOT r.is_exception_group
            AND list_contains($queue_reads::VARCHAR[], r.protected_call_qualified_name)
        ), swallowed AS (
          SELECT 'guard at `' || any_value(location) || '` has ' || count(*)
                   || ' handlers that answer a failure with nothing' AS message,
                 path, start_line, start_column, fact_order, guard
          FROM clauses SEMI JOIN silent USING (fact_order, guard, handler_offset)
          ANTI JOIN queue_reads USING (path, start_line, start_column)
          GROUP BY fact_order, guard, path, start_line, start_column
        ), bindings AS (
          SELECT *, regexp_extract_all(str_split(text, '=')[1], $word) AS words FROM (
            SELECT n.fact_order, n.ordinal, n.subtree_end, n.path, n.start_line, n.start_column,
                   decode(array_slice(f.source, n.byte_start + 1,
                                      n.byte_start + n.byte_length)) AS text
            FROM syntax_nodes n JOIN (
              SELECT fact_order, encode(source) AS source FROM syntax_facts
              WHERE kind = 'callable' AND list_contains($failures_as_values::VARCHAR[], language)
            ) f USING (fact_order)
            WHERE n.kind = 'binding')
        )
        SELECT message, path, start_line, start_column, fact_order, guard FROM swallowed
        UNION ALL BY NAME
        SELECT DISTINCT '`' || trim(text, $space) || '` discards the result of a fallible call'
                 AS message, path, start_line, start_column
        FROM bindings b
        WHERE list_contains(words, $discard)
          AND list_bool_and(
            list_transform(words, lambda w: list_contains($declarators::VARCHAR[], w)))
          AND EXISTS (SELECT 1 FROM syntax_nodes c WHERE c.fact_order = b.fact_order
                      AND c.kind = 'call' AND c.ordinal >= b.ordinal AND c.ordinal < b.subtree_end)
        """

    # Python only: a `try:` whose one handler is `except X: pass` (or `...`) becomes
    # `with suppress(X):` over the same body, the handler lines gone. Left alone are guards with
    # an `else`, a `finally`, several handlers, or a handler that does anything else (`continue`
    # would no longer skip the rest of the loop body); logging or recovering instead is a decision.
    fix = r"""
        WITH guards AS (
          SELECT finding_key, path, start_line, start_column, fact_order, guard FROM finding
          WHERE guard IS NOT NULL AND regexp_matches(path, '\.pyi?$')
        ), parts AS (
          SELECT g.*, l."offset" AS off, l.stripped, l.indent,
                 max(l."offset") OVER (PARTITION BY g.finding_key) AS last_off
          FROM guards g JOIN errors_guard_lines() l
            ON l.fact_order = g.fact_order AND l.guard = g.guard
        ), clauses AS (
          SELECT finding_key, count(*) AS clause_count,
                 count(*) FILTER (WHERE regexp_matches(stripped, '^except\b')) AS handlers,
                 min(off) FILTER (WHERE regexp_matches(stripped, '^except\b')) AS handler_off
          FROM parts
          WHERE off > 0 AND indent = start_column
            AND regexp_matches(stripped, '^(except|else|finally|elif)\b')
          GROUP BY finding_key
        ), handled AS (
          SELECT p.finding_key, p.path, p.start_line, p.start_column, c.handler_off, p.last_off,
                 regexp_extract(h.stripped, '^except\s*([^:]*?)(?:\s+as\s+\w+)?\s*:\s*(.*)$', 1)
                   AS caught,
                 regexp_extract(h.stripped, '^except\s*([^:]*?)(?:\s+as\s+\w+)?\s*:\s*(.*)$', 2)
                   AS inline
          FROM parts p JOIN clauses c USING (finding_key)
          JOIN parts h ON h.finding_key = p.finding_key AND h.off = c.handler_off
          WHERE p.off = 0 AND p.stripped = 'try:' AND c.clause_count = 1 AND c.handlers = 1
            AND NOT starts_with(h.stripped, 'except*')
            AND NOT EXISTS (
              SELECT 1 FROM parts b WHERE b.finding_key = p.finding_key AND b.off > c.handler_off
                AND b.stripped NOT IN ('pass', '...'))
            AND (h.stripped LIKE '%:' OR regexp_matches(h.stripped, ':\s*(pass|\.\.\.)$'))
        ), spelled AS (
          SELECT *, CASE WHEN caught = '' THEN 'BaseException'
                         WHEN starts_with(caught, '(') AND ends_with(caught, ')')
                         THEN substr(caught, 2, length(caught) - 2)
                         ELSE caught END AS suppressed
          FROM handled
          WHERE (inline = '' AND last_off > handler_off)
             OR (inline IN ('pass', '...') AND last_off = handler_off)
        )
        SELECT finding_key, path, start_line, start_column, start_line AS end_line,
               start_column + 4 AS end_column, 'with suppress(' || suppressed || '):' AS text,
               'from contextlib import suppress' AS "import"
        FROM spelled
        UNION ALL BY NAME
        SELECT finding_key, path, start_line + handler_off AS start_line, 0 AS start_column,
               start_line + last_off + 1 AS end_line, 0 AS end_column, '' AS text
        FROM spelled
    """
    examples = (
        Example(
            bad=(
                "import shutil\n\n\ndef clean(path):\n    try:\n        shutil.rmtree(path)\n"
                "    except FileNotFoundError:\n        pass\n"
            ),
            good=(
                "import shutil\nfrom contextlib import suppress\n\n\ndef clean(path):\n"
                "    with suppress(FileNotFoundError):\n        shutil.rmtree(path)\n"
            ),
        ),
    )

    def parameters(self) -> dict[str, Setting]:
        opening = _opening(self.handlers)
        trimmed = " \t{};"
        queue_reads = [
            f"queue.{kind}.{method}"
            for kind in ("Queue", "SimpleQueue", "LifoQueue", "PriorityQueue")
            for method in ("get", "get_nowait")
        ]
        word = _WORD
        space = " \t\n\r\x0b\x0c"
        declarators = ["let", "const", "var", "val", "mut", self.discard]
        return super().parameters() | {
            "opening": opening,
            "trimmed": trimmed,
            "queue_reads": queue_reads,
            "word": word,
            "space": space,
            "declarators": declarators,
        }
