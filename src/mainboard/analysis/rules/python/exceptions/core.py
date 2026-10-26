"""Rules over exception handling and definitions (mcmr's PY-EXCE family)."""

from ....catalog import Example, Rule


class BroadTryLiteralSetup(Rule, id="PY-EXCE0001", applicability="safe"):
    """Literal local setup assignments needlessly protected by a broad `try`."""

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM try_block_regions r JOIN try_block_facts f USING (fact_id)
            WHERE r.leading_literal_assignment_count > 0 AND r.has_following_raising_operation
            GROUP BY ALL
        """
    message = "broad try literal setup is {value} for `{fact_id}`"
    # The literal assignments that open a `try` body move above the `try`, one level out. Each
    # must be one line, adjacent, and open the body right under the `try:` line (a comment among
    # them needs a person to say which statement it belongs to), the `try` has no `finally`, and
    # every line dedents by whitespace.
    fix = r"""
        WITH setups AS (
          SELECT f.path, r.record_id, r.statement_span_start_line AS try_line,
                 r.statement_span_start_column AS try_column, count(*) AS assignments,
                 min(a.span_start_line) AS first_line, max(a.span_end_line) AS last_line,
                 min(a.span_start_column) AS body_column,
                 max(a.span_start_column) AS maximum_body_column,
                 bool_and(a.span_start_line = a.span_end_line) AS one_line_each,
                 any_value(f.fact_id) AS fact_id, any_value(r.has_finally) AS has_finally,
                 any_value(r.is_exception_group) AS is_group
          FROM finding f
          JOIN try_block_regions r ON r.fact_id = f.fact_id
          JOIN try_block_regions_leading_assignments a ON a.parent_id = r.record_id
          WHERE r.leading_literal_assignment_count > 0 AND r.has_following_raising_operation
          GROUP BY f.path, r.record_id, r.statement_span_start_line, r.statement_span_start_column
        ), moves AS (
          SELECT s.*, s.body_column - s.try_column AS dedent,
                 (SELECT string_agg(substr(l.text, s.body_column - s.try_column + 1), chr(10)
                                    ORDER BY l.line)
                  FROM source_lines() l
                  WHERE l.path = s.path AND l.line BETWEEN s.first_line AND s.last_line
                    AND regexp_matches(l.text, '^[ \t]{' || (s.body_column - s.try_column) || '}'))
                   AS block,
                 (SELECT count(*) FROM source_lines() l
                  WHERE l.path = s.path AND l.line BETWEEN s.first_line AND s.last_line
                    AND regexp_matches(l.text, '^[ \t]{' || (s.body_column - s.try_column) || '}')
                 ) AS lines_found,
                 s.one_line_each AND s.body_column = s.maximum_body_column
                   AND s.last_line - s.first_line + 1 = s.assignments
                   AND s.first_line = s.try_line + 1
                   AND NOT s.has_finally AND NOT s.is_group AS adjacent
          FROM setups s
        ), checked AS (
          SELECT *, adjacent AND lines_found = assignments AND dedent > 0 AS movable FROM moves
        )
        SELECT hash(record_id) AS finding_key, path, try_line AS start_line, 0 AS start_column,
               try_line AS end_line, 0 AS end_column, block || chr(10) AS text
        FROM checked WHERE movable
        UNION ALL BY NAME
        SELECT hash(record_id) AS finding_key, path, first_line AS start_line, 0 AS start_column,
               last_line + 1 AS end_line, 0 AS end_column, '' AS text
        FROM checked WHERE movable
    """
    examples = (
        Example(
            bad=(
                """def read(path):
    try:
        mode = 'rb'
        return open(path, mode).read()
    except OSError:
        return b''
"""
            ),
            good=(
                """def read(path):
    mode = 'rb'
    try:
        return open(path, mode).read()
    except OSError:
        return b''
"""
            ),
        ),
    )


class BoundedExceptionRegion(Rule, id="PY-EXCE0002"):
    """`try` clauses guarding more than `maximum` statements, and `except`, `else` and `finally`
    clauses running more than `handling`. A wide `try` makes its handler answer for failures
    nobody can name, while a handler needs room to log, clean up and re-raise. Statements count
    through nested `if`, loops, `with` and `match`; a nested function, class or `try` is its own
    region."""

    maximum: int = 1
    handling: int = 3

    query = """
            WITH tries AS (
              SELECT file_id, node_id AS try, descendant_count AS size, start_line, start_column,
                     coalesce(owner_id, -1) AS home
              FROM core_node WHERE type = 'try_statement'
            ), parts AS (
              SELECT t.file_id, t.try, t.home, b.node_id AS block, b.descendant_count AS size,
                     CASE c.type WHEN 'block' THEN 'try' WHEN 'except_clause' THEN 'except'
                       WHEN 'except_group_clause' THEN 'except*' WHEN 'else_clause' THEN 'else'
                       ELSE 'finally' END AS clause,
                     if(c.type = 'block', t.start_line, c.start_line) AS start_line,
                     if(c.type = 'block', t.start_column, c.start_column) AS start_column
              FROM tries t
              JOIN core_node c ON c.file_id = t.file_id AND c.parent_id = t.try
                AND c.type IN ('block', 'except_clause', 'except_group_clause', 'else_clause',
                               'finally_clause')
              JOIN core_node b ON b.file_id = t.file_id AND b.type = 'block'
                AND (b.node_id = c.node_id OR b.parent_id = c.node_id)
            ), owned AS (
              SELECT s.file_id, s.node_id AS stmt, s.type, coalesce(s.owner_id, -1) AS home,
                     arg_max(t.try, t.try) AS owner
              FROM core_node s
              JOIN core_node p ON p.file_id = s.file_id AND p.node_id = s.parent_id
                AND p.type = 'block'
              JOIN tries t ON t.file_id = s.file_id AND s.node_id > t.try
                AND s.node_id <= t.try + t.size
              WHERE s.type <> 'comment'
              GROUP BY s.file_id, s.node_id, s.type, s.owner_id
            ), counts AS (
              SELECT b.file_id, b.clause, b.start_line, b.start_column,
                     if(b.clause = 'try', $maximum, $handling) AS ceiling,
                     count(s.stmt) AS value
              FROM parts b
              LEFT JOIN owned s ON s.file_id = b.file_id AND s.owner = b.try AND s.home = b.home
                AND s.stmt > b.block AND s.stmt <= b.block + b.size
                AND NOT list_contains(['function_definition', 'class_definition',
                                       'decorated_definition', 'try_statement'], s.type)
              GROUP BY ALL
            )
            SELECT f.path, c.start_line, c.start_column, c.clause, c.value, c.ceiling
            FROM counts c JOIN core_file f USING (file_id)
            WHERE f.language = 'python' AND c.value > c.ceiling
        """
    message = "the `{clause}` clause holds {value} statements, past the ceiling of {ceiling}"
    examples = (
        Example(
            bad=(
                """def load(path):
    try:
        text = path.read_text()
        return parse(text)
    except OSError:
        return None
"""
            ),
            good=(
                """def load(path):
    try:
        text = path.read_text()
    except OSError:
        return None
    return parse(text)
"""
            ),
        ),
        Example(
            bad=(
                """def load(path):
    try:
        return path.read_text()
    except OSError:
        log(path)
        clean(path)
        report(path)
        raise
"""
            ),
            good=(
                """def load(path):
    try:
        return path.read_text()
    except OSError:
        log(path)
        clean(path)
        raise
"""
            ),
        ),
    )


class SharedExceptionPlacement(Rule, id="PY-EXCE0003"):
    """Reused exception classes outside the narrowest justified `exceptions.py`."""

    minimum_importing_modules: int = 2
    preferred_module: str = "exceptions"

    query = """
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM exception_exceptions e JOIN exception_facts f USING (fact_id)
            LEFT JOIN (SELECT fact_id, parent_id, count(DISTINCT string_value) AS importers
                       FROM exception_values WHERE relation = 'exceptions.importing_modules'
                       GROUP BY fact_id, parent_id) i
              ON i.fact_id = e.fact_id AND i.parent_id = e.record_id
            WHERE coalesce(i.importers, 0) >= $minimum_importing_modules
              AND NOT ends_with(e.defining_module, '.' || $preferred_module)
            GROUP BY ALL
        """
    message = "shared exception placement is {value} for `{fact_id}`"
