"""mcmr's QueryFact, derived from sitting_duck's Python syntax trees.

Per Python file, the database operations written through SQLAlchemy or SQLModel that the
PY-SQLA rules read, each an operation on one call:

- `async_sessionmaker` and `session_commit`: a call whose written callee ends in that name (so
  `git.commit(...)` is one too), with the factory's `expire_on_commit` and unknown keywords;
- `exec_scalars` / `execute_scalars`: `<session>.exec|execute(<select chain>).scalars()`;
- `primary_key_first`: `<session>.exec(select(M).where(M.<key> == v)).first()`.

A session is a name the file declares as one (a parameter, a variable or an attribute annotated
`Session`/`AsyncSession`, a variable assigned or bound by `with` to `Session(...)`), where the
class resolves through the file's top-level imports to `sqlmodel`; a primary key is a `Field(...,
primary_key=True)` member of a `class M(SQLModel, table=True)`. The select chain is a run of method
calls ending in `sqlmodel.select(...)`; `execution_options` anywhere in it is noted. An operation
is inside a loop when the nearest enclosing loop or function body of its call is a loop body (a
`for`, `while` or its `else`).

The relations are the steps: the trimmed syntax tree of the files that hold a candidate call
(`nodes`), their attribute accesses and calls with the argument counts the shapes need, the
imports, the annotated names, the sessions, the primary keys, the scope bodies and the operations.

Differences from the kernel, kept on purpose: calls are found anywhere a syntax tree holds them
(`elif` tests, `match` subjects), where the kernel walks the statement positions it enumerates; a
file the grammar cannot parse is skipped, as the kernel skips what ruff cannot; a chained
assignment `a = b = Session(e)` names `a` and `b` but not a third target; the texts of the two
segments are not stated.
"""

from ...catalog import Family, Function

_KNOWN = (
    "['bind', 'class_', 'expire_on_commit', 'autoflush', 'autobegin', 'info', "
    "'join_transaction_mode']"
)
_NOT_EXPRESSION = "('(', ')', ',', 'comment', 'keyword_argument', 'dictionary_splat')"


class QueryDotted(Function):
    """The text when it is a plain dotted name (`session`, `self.db`), NULL otherwise."""

    text: str

    expression = r"CASE WHEN regexp_matches(text, '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$') THEN text END"


class QueryResolved(Function):
    """A written name with its head replaced by the origin the file's imports give it."""

    origin: str
    written: str

    expression = r"""
        CASE WHEN origin IS NULL THEN written WHEN NOT contains(written, '.') THEN origin
             ELSE origin || '.' || substr(written, instr(written, '.') + 1) END
    """


class QuerySession(Function):
    """Whether a resolved name is `sqlmodel`'s `Session` or `AsyncSession`."""

    resolved: str

    expression = r"""
        coalesce(starts_with(resolved, 'sqlmodel.')
                 AND regexp_extract(resolved, '[^.]*$') IN ('Session', 'AsyncSession'), false)
    """


class Query(Family):
    family = "QueryFact"
    verified = True
    relations = {
        "facts": r"""
            SELECT row_number() OVER (ORDER BY f.path) - 1 AS fact_order,
                   'queryfact:' || f.path AS fact_id, f.path, 1 AS start_line, 0 AS start_column,
                   m.end_line, m.end_column, 'python' AS language
            FROM ast_files f JOIN ast_nodes m ON m.fact_id = f.fact_id AND m.type = 'module'
            WHERE f.language = 'python' AND f.fact_id NOT IN (
              SELECT fact_id FROM ast_nodes
              WHERE type = 'ERROR' OR (children_count = 0 AND start_line = end_line
                                       AND start_column = end_column AND type <> 'module'))
        """,
        "nodes": r"""
            SELECT n.path, n.node_id, n.parent_id, n.sibling_index, n.depth, n.type, n.name,
                   n.start_line, n.start_column, n.end_line, n.end_column, n.descendant_count
            FROM ast_nodes n
            JOIN (SELECT DISTINCT a.path FROM ast_nodes a
                  SEMI JOIN query_facts f ON f.path = a.path
                  WHERE a.type = 'call'
                    AND a.name IN ('commit', 'async_sessionmaker', 'scalars', 'first')) r
              ON r.path = n.path
        """,
        "attributes": r"""
            SELECT p.path, p.node_id, arg_min(c.node_id, c.sibling_index) AS object_id,
                   arg_max(c.node_id, c.sibling_index) AS attr_id,
                   arg_max(c.name, c.sibling_index) AS attr,
                   arg_max(c.start_line, c.sibling_index) AS attr_start_line,
                   arg_max(c.start_column, c.sibling_index) AS attr_start_column,
                   arg_max(c.end_line, c.sibling_index) AS attr_end_line,
                   arg_max(c.end_column, c.sibling_index) AS attr_end_column
            FROM query_nodes p JOIN query_nodes c ON c.path = p.path AND c.parent_id = p.node_id
            WHERE p.type = 'attribute' AND c.type <> 'comment'
            GROUP BY p.path, p.node_id
        """,
        "imports": r"""
            WITH statements AS (
              SELECT path, node_id, type FROM query_nodes
              WHERE depth = 1 AND type IN ('import_statement', 'import_from_statement')
            ), plain AS (
              SELECT s.path, s.node_id * 100000 + c.sibling_index AS position,
                     CASE c.type WHEN 'aliased_import' THEN c.name
                                 ELSE split_part(c.name, '.', 1) END AS local,
                     CASE c.type WHEN 'aliased_import' THEN d.name ELSE c.name END AS origin
              FROM statements s
              JOIN query_nodes c ON c.path = s.path AND c.parent_id = s.node_id
                                AND c.type IN ('dotted_name', 'aliased_import')
              LEFT JOIN query_nodes d ON d.path = c.path AND d.parent_id = c.node_id
                                     AND d.type = 'dotted_name'
              WHERE s.type = 'import_statement'
            ), modules AS (
              SELECT s.path, s.node_id AS statement, i.sibling_index AS after,
                     CASE m.type WHEN 'dotted_name' THEN m.name ELSE r.name END AS module
              FROM statements s
              JOIN query_nodes m ON m.path = s.path AND m.parent_id = s.node_id
                                AND m.sibling_index = 1
              JOIN query_nodes i ON i.path = s.path AND i.parent_id = s.node_id
                                AND i.type = 'import'
              LEFT JOIN query_nodes r ON r.path = m.path AND r.parent_id = m.node_id
                                     AND m.type = 'relative_import' AND r.type = 'dotted_name'
              WHERE s.type = 'import_from_statement'
            ), named AS (
              SELECT m.path, m.statement * 100000 + c.sibling_index AS position, c.name AS local,
                     m.module || '.' || CASE c.type WHEN 'aliased_import' THEN d.name
                                                    ELSE c.name END AS origin
              FROM modules m
              JOIN query_nodes c ON c.path = m.path AND c.parent_id = m.statement
                                AND c.sibling_index > m.after
                                AND c.type IN ('dotted_name', 'aliased_import')
              LEFT JOIN query_nodes d ON d.path = c.path AND d.parent_id = c.node_id
                                     AND d.type = 'dotted_name'
              WHERE m.module IS NOT NULL
            )
            SELECT path, local, arg_max(origin, position) AS origin
            FROM (SELECT * FROM plain UNION ALL SELECT * FROM named) GROUP BY path, local
        """,
        "calls": rf"""
            WITH arguments AS (
              SELECT c.path, c.parent_id AS args_id,
                     count(*) FILTER (WHERE c.type NOT IN {_NOT_EXPRESSION}) AS positional,
                     count(*) FILTER (WHERE c.type IN ('keyword_argument', 'dictionary_splat'))
                       AS keywords,
                     arg_min(c.node_id, c.sibling_index)
                       FILTER (WHERE c.type NOT IN {_NOT_EXPRESSION}) AS argument_id
              FROM query_nodes c
              JOIN query_nodes a ON a.path = c.path AND a.node_id = c.parent_id
                                AND a.type = 'argument_list'
              GROUP BY c.path, c.parent_id
            ), shaped AS (
              SELECT x.path, x.node_id AS id, x.name, x.start_line, x.start_column, x.end_line,
                     x.end_column, x.descendant_count, f.node_id AS func_id, f.type AS func_type,
                     CASE f.type WHEN 'identifier' THEN f.name
                                 WHEN 'attribute' THEN query_dotted(f.name) END AS written,
                     CASE f.type WHEN 'identifier' THEN f.name WHEN 'attribute' THEN fa.attr END
                       AS annotated,
                     a.node_id AS args_id,
                     CASE a.type WHEN 'generator_expression' THEN 1
                                 ELSE coalesce(k.positional, 0) END AS positional,
                     CASE a.type WHEN 'generator_expression' THEN 0
                                 ELSE coalesce(k.keywords, 0) END AS keywords,
                     CASE a.type WHEN 'generator_expression' THEN a.node_id
                                 ELSE k.argument_id END AS argument_id
              FROM query_nodes x
              JOIN query_nodes f ON f.path = x.path AND f.parent_id = x.node_id
                                AND f.sibling_index = 0
              JOIN query_nodes a ON a.path = x.path AND a.parent_id = x.node_id
                                AND a.type IN ('argument_list', 'generator_expression')
              LEFT JOIN arguments k ON k.path = a.path AND k.args_id = a.node_id
              LEFT JOIN query_attributes fa ON fa.path = f.path AND fa.node_id = f.node_id
              WHERE x.type = 'call'
            )
            SELECT s.*,
                   coalesce(query_resolved(w.origin, s.written) = 'sqlmodel.select', false)
                     AS is_select,
                   query_session(query_resolved(n.origin, s.annotated)) AS is_session_factory
            FROM shaped s
            LEFT JOIN query_imports w ON w.path = s.path
                                     AND w.local = split_part(s.written, '.', 1)
            LEFT JOIN query_imports n ON n.path = s.path AND n.local = s.annotated
        """,
        "annotated": r"""
            SELECT p.path, p.node_id AS anchor_id,
                   CASE e.type
                     WHEN 'identifier' THEN e.name
                     WHEN 'attribute' THEN ea.attr
                     WHEN 'string' THEN sc.name
                     WHEN 'subscript' THEN CASE v.type WHEN 'identifier' THEN v.name
                                                       WHEN 'attribute' THEN va.attr END
                   END AS written
            FROM query_nodes p
            JOIN query_nodes t ON t.path = p.path AND t.parent_id = p.node_id AND t.type = 'type'
            JOIN query_nodes e ON e.path = t.path AND e.parent_id = t.node_id
                              AND e.sibling_index = 0
            LEFT JOIN query_attributes ea ON ea.path = e.path AND ea.node_id = e.node_id
            LEFT JOIN query_nodes sc ON sc.path = e.path AND sc.parent_id = e.node_id
                                    AND sc.type = 'string_content'
            LEFT JOIN query_nodes v ON v.path = e.path AND v.parent_id = e.node_id
                                   AND v.sibling_index = 0 AND e.type = 'subscript'
            LEFT JOIN query_attributes va ON va.path = v.path AND va.node_id = v.node_id
            WHERE p.type IN ('typed_parameter', 'typed_default_parameter', 'assignment')
        """,
        "sessions": r"""
            WITH typed AS (
              SELECT p.path,
                     CASE p.type WHEN 'assignment' THEN query_dotted(c.name)
                                 ELSE coalesce(nullif(p.name, ''), i.name) END AS name
              FROM query_nodes p
              JOIN query_annotated a ON a.path = p.path AND a.anchor_id = p.node_id
              LEFT JOIN query_imports m ON m.path = a.path
                                       AND m.local = split_part(a.written, '.', 1)
              JOIN query_nodes c ON c.path = p.path AND c.parent_id = p.node_id
                                AND c.sibling_index = 0
              LEFT JOIN query_nodes i ON i.path = c.path AND i.parent_id = c.node_id
                                     AND i.type = 'identifier'
              WHERE query_session(query_resolved(m.origin, a.written))
            ), factories AS (
              SELECT x.path, x.node_id, x.parent_id
              FROM query_nodes x
              JOIN query_nodes c ON c.path = x.path AND c.parent_id = x.node_id
                                AND c.sibling_index = 2 AND c.type = 'call'
              JOIN query_calls k ON k.path = c.path AND k.id = c.node_id AND k.is_session_factory
              WHERE x.type = 'assignment'
            ), assigned AS (
              SELECT path, node_id AS owner FROM factories
              UNION ALL
              SELECT f.path, f.parent_id FROM factories f
              JOIN query_nodes p ON p.path = f.path AND p.node_id = f.parent_id
                                AND p.type = 'assignment'
            ), bound AS (
              SELECT a.path, query_dotted(t.name) AS name
              FROM query_nodes a
              JOIN query_nodes w ON w.path = a.path AND w.node_id = a.parent_id
                                AND w.type = 'with_item'
              JOIN query_nodes c ON c.path = a.path AND c.parent_id = a.node_id
                                AND c.sibling_index = 0 AND c.type = 'call'
              JOIN query_calls k ON k.path = c.path AND k.id = c.node_id AND k.is_session_factory
              JOIN query_nodes g ON g.path = a.path AND g.parent_id = a.node_id
                                AND g.type = 'as_pattern_target'
              JOIN query_nodes t ON t.path = g.path AND t.parent_id = g.node_id
                                AND t.sibling_index = 0
              WHERE a.type = 'as_pattern'
            )
            SELECT DISTINCT path, name FROM (
              SELECT path, name FROM typed
              UNION ALL
              SELECT o.path, query_dotted(l.name) FROM assigned o
              JOIN query_nodes l ON l.path = o.path AND l.parent_id = o.owner
                                AND l.sibling_index = 0
              UNION ALL
              SELECT path, name FROM bound
            ) WHERE name IS NOT NULL
        """,
        "keys": r"""
            WITH classes AS (
              SELECT c.path, c.node_id, c.name AS model, al.node_id AS bases
              FROM query_nodes c
              JOIN query_nodes al ON al.path = c.path AND al.parent_id = c.node_id
                                 AND al.type = 'argument_list'
              WHERE c.type = 'class_definition'
            ), tables AS (
              SELECT c.* FROM classes c
              WHERE EXISTS (
                  SELECT 1 FROM query_nodes b
                  LEFT JOIN query_imports i ON i.path = b.path
                                           AND i.local = split_part(b.name, '.', 1)
                  WHERE b.path = c.path AND b.parent_id = c.bases
                    AND b.type IN ('identifier', 'attribute') AND query_dotted(b.name) IS NOT NULL
                    AND query_resolved(i.origin, b.name) = 'sqlmodel.SQLModel')
                AND EXISTS (
                  SELECT 1 FROM query_nodes k
                  JOIN query_nodes v ON v.path = k.path AND v.parent_id = k.node_id
                                    AND v.sibling_index = 2
                  WHERE k.path = c.path AND k.parent_id = c.bases AND k.type = 'keyword_argument'
                    AND k.name = 'table' AND v.type = 'true')
            ), kept AS (
              SELECT path, model, max(node_id) AS node_id FROM tables GROUP BY path, model
            )
            SELECT DISTINCT t.path, t.model, query_dotted(l.name) AS field
            FROM kept t
            JOIN query_nodes b ON b.path = t.path AND b.parent_id = t.node_id AND b.type = 'block'
            JOIN query_nodes s ON s.path = b.path AND s.parent_id = b.node_id
                              AND s.type = 'expression_statement'
            JOIN query_nodes a ON a.path = s.path AND a.parent_id = s.node_id
                              AND a.type = 'assignment'
            JOIN query_nodes y ON y.path = a.path AND y.parent_id = a.node_id AND y.type = 'type'
            JOIN query_nodes l ON l.path = a.path AND l.parent_id = a.node_id
                              AND l.sibling_index = 0
            JOIN query_nodes r ON r.path = a.path AND r.parent_id = a.node_id AND r.type = 'call'
                              AND r.sibling_index > y.sibling_index
            JOIN query_calls k ON k.path = r.path AND k.id = r.node_id
            LEFT JOIN query_imports i ON i.path = k.path
                                     AND i.local = split_part(k.written, '.', 1)
            WHERE query_resolved(i.origin, k.written) = 'sqlmodel.Field' AND EXISTS (
              SELECT 1 FROM query_nodes w
              JOIN query_nodes v ON v.path = w.path AND v.parent_id = w.node_id
                                AND v.sibling_index = 2
              WHERE w.path = k.path AND w.parent_id = k.args_id AND w.type = 'keyword_argument'
                AND w.name = 'primary_key' AND v.type = 'true')
              AND query_dotted(l.name) IS NOT NULL
        """,
        "bodies": r"""
            SELECT b.path, b.node_id, b.descendant_count,
                   p.type IN ('for_statement', 'while_statement') OR p.type = 'else_clause'
                     AS is_loop
            FROM query_nodes b
            JOIN query_nodes p ON p.path = b.path AND p.node_id = b.parent_id
            LEFT JOIN query_nodes g ON g.path = p.path AND g.node_id = p.parent_id
            WHERE b.type = 'block'
              AND (p.type IN ('function_definition', 'class_definition', 'for_statement',
                              'while_statement')
                   OR (p.type = 'else_clause' AND g.type IN ('for_statement', 'while_statement')))
        """,
        "operations": rf"""
            WITH roots AS (
              SELECT c.path, c.id, arg_min(d.type, d.depth) AS root_type
              FROM query_calls c
              JOIN query_nodes d ON d.path = c.path AND d.start_line = c.start_line
                                AND d.start_column = c.start_column
                                AND d.node_id BETWEEN c.id AND c.id + c.descendant_count
                                AND d.type NOT IN ('call', 'attribute')
              WHERE c.name IN ('commit', 'async_sessionmaker')
                AND c.func_type IN ('identifier', 'attribute')
              GROUP BY c.path, c.id
            ), keywords AS (
              SELECT c.path, c.id,
                     bool_or(k.type = 'dictionary_splat' OR NOT list_contains({_KNOWN}, k.name))
                       AS has_unknown_keywords,
                     arg_min(v.type, k.sibling_index) FILTER (WHERE k.name = 'expire_on_commit')
                       AS expire_value
              FROM query_calls c
              JOIN query_nodes k ON k.path = c.path AND k.parent_id = c.args_id
                                AND k.type IN ('keyword_argument', 'dictionary_splat')
              LEFT JOIN query_nodes v ON v.path = k.path AND v.parent_id = k.node_id
                                     AND v.sibling_index = 2 AND k.type = 'keyword_argument'
              WHERE c.name IN ('commit', 'async_sessionmaker')
              GROUP BY c.path, c.id
            ), named AS (
              SELECT c.path, c.id, c.start_line, c.start_column, c.end_line, c.end_column,
                     CASE c.name WHEN 'commit' THEN 'session_commit' ELSE c.name END AS kind,
                     coalesce(k.expire_value, '') <> 'false' AS expire_on_commit,
                     coalesce(k.has_unknown_keywords, false) AS has_unknown_keywords
              FROM query_calls c
              JOIN roots r ON r.path = c.path AND r.id = c.id AND r.root_type = 'identifier'
              LEFT JOIN keywords k ON k.path = c.path AND k.id = c.id
            ), execs AS (
              SELECT e.path, e.id, e.name, e.end_line, e.end_column, e.argument_id,
                     fa.attr_start_line, fa.attr_start_column, fa.attr_end_line,
                     fa.attr_end_column
              FROM query_calls e
              JOIN query_attributes fa ON fa.path = e.path AND fa.node_id = e.func_id
              JOIN query_nodes r ON r.path = fa.path AND r.node_id = fa.object_id
                                AND r.type IN ('identifier', 'attribute')
              WHERE e.name IN ('exec', 'execute') AND e.func_type = 'attribute'
                AND e.keywords = 0 AND e.positional = 1
                AND EXISTS (SELECT 1 FROM query_sessions s
                            WHERE s.path = r.path AND s.name = query_dotted(r.name))
            ), scalars AS (
              SELECT s.path, s.id, s.start_line, s.start_column, s.end_line, s.end_column,
                     x.name AS method, x.argument_id,
                     x.attr_start_line, x.attr_start_column, x.attr_end_line, x.attr_end_column,
                     x.end_line AS execute_end_line, x.end_column AS execute_end_column
              FROM query_calls s
              JOIN query_attributes sa ON sa.path = s.path AND sa.node_id = s.func_id
              JOIN execs x ON x.path = s.path AND x.id = sa.object_id
              WHERE s.name = 'scalars' AND s.func_type = 'attribute' AND s.positional = 0
                AND s.keywords = 0
            ), spine AS (
              SELECT h.path, h.argument_id, d.node_id, d.type,
                     row_number() OVER (PARTITION BY h.path, h.argument_id ORDER BY d.depth) AS rn
              FROM (SELECT DISTINCT path, argument_id FROM scalars) h
              JOIN query_nodes q ON q.path = h.path AND q.node_id = h.argument_id
              JOIN query_nodes d ON d.path = q.path AND d.start_line = q.start_line
                                AND d.start_column = q.start_column
                                AND d.node_id BETWEEN q.node_id AND q.node_id + q.descendant_count
            ), marked AS (
              SELECT s.*, c.is_select, c.positional, a.attr
              FROM spine s
              LEFT JOIN query_calls c ON c.path = s.path AND c.id = s.node_id
              LEFT JOIN query_attributes a ON a.path = s.path AND a.node_id = s.node_id
            ), selections AS (
              SELECT m.path, m.argument_id,
                     max(m.positional) FILTER (WHERE m.rn = f.at) AS selected_expression_count,
                     coalesce(bool_or(m.attr = 'execution_options')
                                FILTER (WHERE m.type = 'attribute'), false)
                       AS has_execution_options
              FROM (SELECT path, argument_id, min(rn) AS at FROM marked
                    WHERE type = 'call' AND is_select GROUP BY path, argument_id) f
              JOIN marked m ON m.path = f.path AND m.argument_id = f.argument_id AND m.rn <= f.at
              GROUP BY m.path, m.argument_id, f.at
              HAVING count(*) FILTER (WHERE m.type = CASE WHEN m.rn % 2 = 1 THEN 'call'
                                                          ELSE 'attribute' END) = f.at
            ), firsts AS (
              SELECT f.path, f.id, f.start_line, f.start_column, f.end_line, f.end_column,
                     CASE WHEN q.name = 'execution_options' AND q.func_type = 'attribute'
                          THEN qa.object_id ELSE x.argument_id END AS query_id,
                     coalesce(q.name = 'execution_options' AND q.func_type = 'attribute', false)
                       AS has_execution_options
              FROM query_calls f
              JOIN query_attributes fa ON fa.path = f.path AND fa.node_id = f.func_id
              JOIN execs x ON x.path = f.path AND x.id = fa.object_id AND x.name = 'exec'
              LEFT JOIN query_calls q ON q.path = x.path AND q.id = x.argument_id
              LEFT JOIN query_attributes qa ON qa.path = q.path AND qa.node_id = q.func_id
              WHERE f.name = 'first' AND f.func_type = 'attribute' AND f.positional = 0
                AND f.keywords = 0
            ), keyed AS (
              SELECT f.*
              FROM firsts f
              JOIN query_calls w ON w.path = f.path AND w.id = f.query_id AND w.name = 'where'
                                AND w.func_type = 'attribute' AND w.keywords = 0
                                AND w.positional = 1
              JOIN query_attributes wa ON wa.path = w.path AND wa.node_id = w.func_id
              JOIN query_calls s ON s.path = w.path AND s.id = wa.object_id AND s.is_select
                                AND s.keywords = 0 AND s.positional = 1
              JOIN query_nodes m ON m.path = s.path AND m.node_id = s.argument_id
              LEFT JOIN query_attributes ma ON ma.path = m.path AND ma.node_id = m.node_id
              JOIN (SELECT p.path, p.node_id AS predicate,
                           list(c.type ORDER BY c.sibling_index) AS shape,
                           arg_min(c.node_id, c.sibling_index) AS left_id
                    FROM query_nodes p
                    JOIN query_nodes c ON c.path = p.path AND c.parent_id = p.node_id
                                      AND c.type <> 'comment'
                    WHERE p.type = 'comparison_operator' GROUP BY p.path, p.node_id) cmp
                ON cmp.path = w.path AND cmp.predicate = w.argument_id
              JOIN query_attributes la ON la.path = cmp.path AND la.node_id = cmp.left_id
              JOIN query_nodes o ON o.path = la.path AND o.node_id = la.object_id
              JOIN query_keys k ON k.path = la.path AND k.field = la.attr
              WHERE len(cmp.shape) = 3 AND cmp.shape[2] = '=='
                AND k.model = regexp_extract(query_dotted(o.name), '[^.]*$')
                AND k.model = CASE m.type WHEN 'identifier' THEN m.name
                                          WHEN 'attribute' THEN ma.attr END
            ), found AS (
              SELECT path, id, start_line, start_column, end_line, end_column, kind,
                     'sqlalchemy' AS framework, expire_on_commit, has_unknown_keywords,
                     0 AS selected_expression_count, false AS has_primary_key_equality,
                     false AS has_execution_options,
                     NULL::BIGINT AS execute_start_line, NULL::BIGINT AS execute_start_column,
                     NULL::BIGINT AS execute_end_line, NULL::BIGINT AS execute_end_column,
                     NULL::BIGINT AS scalars_start_line, NULL::BIGINT AS scalars_start_column
              FROM named
              UNION ALL
              SELECT s.path, s.id, s.start_line, s.start_column, s.end_line, s.end_column,
                     s.method || '_scalars', 'sqlmodel', true, false, n.selected_expression_count,
                     false, n.has_execution_options, s.attr_start_line, s.attr_start_column,
                     s.attr_end_line, s.attr_end_column, s.execute_end_line, s.execute_end_column
              FROM scalars s
              JOIN selections n ON n.path = s.path AND n.argument_id = s.argument_id
              UNION ALL
              SELECT path, id, start_line, start_column, end_line, end_column,
                     'primary_key_first', 'sqlmodel', true, false, 1, true, has_execution_options,
                     NULL, NULL, NULL, NULL, NULL, NULL
              FROM keyed
            ), looped AS (
              SELECT f.path, f.id, coalesce(arg_max(b.is_loop, b.node_id), false) AS is_inside_loop
              FROM found f
              LEFT JOIN query_bodies b ON b.path = f.path AND b.node_id < f.id
                                      AND f.id <= b.node_id + b.descendant_count
              GROUP BY f.path, f.id
            )
            SELECT q.fact_order, q.fact_id, q.fact_id AS parent_id,
                   q.fact_id || '/operations:' || (n - 1) AS record_id, n - 1 AS ordinal,
                   f.kind, f.framework, l.is_inside_loop, f.expire_on_commit,
                   f.has_unknown_keywords, f.selected_expression_count,
                   f.has_primary_key_equality, f.has_execution_options,
                   f.path || ':' || f.start_line || ':' || f.start_column || ':call' AS node_id,
                   'call' AS node_kind, f.path AS node_span_path,
                   f.start_line AS node_span_start_line, f.start_column AS node_span_start_column,
                   f.end_line AS node_span_end_line, f.end_column AS node_span_end_column,
                   syntax_slice(s.content, f.start_line, f.start_column, f.end_line, f.end_column)
                     AS node_text,
                   CASE WHEN f.scalars_start_line IS NOT NULL
                        THEN f.path || ':' || f.end_line || ':' || f.end_column || ':scalars'
                        END AS scalars_segment_id,
                   CASE WHEN f.scalars_start_line IS NOT NULL THEN 'call-segment' END
                     AS scalars_segment_kind,
                   CASE WHEN f.scalars_start_line IS NOT NULL THEN f.path END
                     AS scalars_segment_span_path,
                   f.scalars_start_line AS scalars_segment_span_start_line,
                   f.scalars_start_column AS scalars_segment_span_start_column,
                   CASE WHEN f.scalars_start_line IS NOT NULL THEN f.end_line END
                     AS scalars_segment_span_end_line,
                   CASE WHEN f.scalars_start_line IS NOT NULL THEN f.end_column END
                     AS scalars_segment_span_end_column,
                   CASE WHEN f.execute_start_line IS NOT NULL
                        THEN f.path || ':' || f.execute_start_line || ':' || f.execute_start_column
                             || ':method' END AS execute_segment_id,
                   CASE WHEN f.execute_start_line IS NOT NULL THEN 'method' END
                     AS execute_segment_kind,
                   CASE WHEN f.execute_start_line IS NOT NULL THEN f.path END
                     AS execute_segment_span_path,
                   f.execute_start_line AS execute_segment_span_start_line,
                   f.execute_start_column AS execute_segment_span_start_column,
                   f.execute_end_line AS execute_segment_span_end_line,
                   f.execute_end_column AS execute_segment_span_end_column
            FROM (SELECT *, row_number() OVER (PARTITION BY path ORDER BY id) AS n FROM found) f
            JOIN looped l ON l.path = f.path AND l.id = f.id
            JOIN query_facts q ON q.path = f.path
            JOIN source_files s ON s.path = f.path
        """,
    }
