"""Rules over how exceptions are handled (mcmr's PY-EXCE handling family)."""

from ....catalog import Relation, Rule


class ExceptionsHandlingSuppressibleRegions(Relation):
    """Each `try` whose only protected statement is a one-line `return`, and whose only handler
    catches one named type and returns `None`: its statement span, the caught type, the protected
    return and the line it sits on."""

    query = """
        WITH returned AS (
          SELECT fact_id, parent_id AS region_id, min(text) AS protected_text,
                 min(span_start_line) AS protected_line
          FROM try_block_regions_protected_statements
          GROUP BY fact_id, parent_id
          HAVING count(*) = 1 AND min(kind) = 'return' AND starts_with(min(text), 'return ')
             AND NOT contains(min(text), chr(10))
        ), nothing AS (
          SELECT fact_id, parent_id AS handler_id FROM try_block_regions_handlers_body
          GROUP BY fact_id, parent_id
          HAVING count(*) = 1 AND min(kind) = 'return'
             AND min(text) IN ('return', 'return None')
        ), handled AS (
          SELECT fact_id, parent_id AS region_id, min(record_id) AS handler_id,
                 min(caught) AS caught
          FROM try_block_regions_handlers GROUP BY fact_id, parent_id
          HAVING count(*) = 1 AND min(caught) <> '' AND NOT min(caught_is_tuple)
             AND min(alias) = ''
        )
        SELECT r.fact_id, r.record_id, handled.caught, returned.protected_text,
               returned.protected_line, r.statement_span_path AS path,
               r.statement_span_start_line AS start_line,
               r.statement_span_start_column AS start_column,
               r.statement_span_end_line AS end_line, r.statement_span_end_column AS end_column
        FROM try_block_regions r
        JOIN returned ON returned.fact_id = r.fact_id AND returned.region_id = r.record_id
        JOIN handled ON handled.fact_id = r.fact_id AND handled.region_id = r.record_id
        JOIN nothing ON nothing.fact_id = r.fact_id AND nothing.handler_id = handled.handler_id
        WHERE NOT r.is_exception_group AND NOT r.has_else AND NOT r.has_finally
          AND r.statement_id IS NOT NULL
"""


class NullableExceptionReturnSuppression(Rule, id="PY-EXCE0004", applicability="safe"):
    """A guarded `return` whose only handler returns `None`, which `suppress` states directly."""

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM exceptions_handling_suppressible_regions() r
            JOIN try_block_facts f USING (fact_id)
            GROUP BY ALL
        """
    message = "nullable exception return is {value} for `{fact_id}`"
    # The `try` becomes `with suppress(Caught):` around the same return, then `return None`
    # (`contextlib.suppress` is imported unless the file does). A `try` with a comment in it is
    # left alone.
    fix = r"""
        WITH regions AS (
          SELECT f.fact_id, hash(r.record_id) AS finding_key, r.path, r.start_line, r.start_column,
                 r.end_line, r.end_column, r.caught, r.protected_text,
                 (SELECT regexp_extract(l.text, '^[ \t]*') FROM source_lines() l
                  WHERE l.path = r.path AND l.line = r.start_line) AS outer_indent,
                 (SELECT regexp_extract(l.text, '^[ \t]*') FROM source_lines() l
                  WHERE l.path = r.path AND l.line = r.protected_line) AS inner_indent,
                 NOT EXISTS (SELECT 1 FROM source_lines() l
                             WHERE l.path = r.path AND l.line BETWEEN r.start_line AND r.end_line
                               AND contains(l.text, '#')) AS has_no_comment,
                 EXISTS (SELECT 1 FROM import_binding_facts i
                         WHERE i.path = r.path AND i.name = 'suppress'
                           AND i.module = 'contextlib') AS has_suppress
          FROM finding f JOIN exceptions_handling_suppressible_regions() r USING (fact_id)
        )
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               'with suppress(' || caught || '):' || chr(10) || inner_indent || protected_text
                 || chr(10) || outer_indent || 'return None' AS text,
               if(has_suppress, NULL, 'from contextlib import suppress') AS "import"
        FROM regions WHERE has_no_comment
    """
