"""Rules over exception handling and definitions (mcmr's PY-EXCE family)."""

from ....catalog import Example, Rule

_BOOM = "class Boom(Exception):\n    pass\n"
_PACKAGE = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
    "src/pkg/a/__init__.py": "",
    "src/pkg/b/__init__.py": "",
}
_SHARED = {
    **_PACKAGE,
    "src/pkg/a/errs.py": _BOOM,
    "src/pkg/b/u2.py": "from ..a.errs import Boom\n\n\ndef g():\n    raise Boom\n",
}


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


class BoundedExceptionRegion(Rule, id="PY-EXCE0002", applicability="review"):
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
              SELECT b.file_id, b.try, b.clause, b.start_line, b.start_column,
                     if(b.clause = 'try', $maximum, $handling) AS ceiling,
                     count(s.stmt) AS value
              FROM parts b
              LEFT JOIN owned s ON s.file_id = b.file_id AND s.owner = b.try AND s.home = b.home
                AND s.stmt > b.block AND s.stmt <= b.block + b.size
                AND NOT list_contains(['function_definition', 'class_definition',
                                       'decorated_definition', 'try_statement'], s.type)
              GROUP BY ALL
            )
            SELECT f.path, c.start_line, c.start_column, c.clause, c.value, c.ceiling, c.file_id,
                   CASE WHEN c.clause = 'try' THEN c.try END AS node
            FROM counts c JOIN core_file f USING (file_id)
            WHERE f.language = 'python' AND c.value > c.ceiling
        """
    message = "the `{clause}` clause holds {value} statements, past the ceiling of {ceiling}"
    # A `try` guarding too much keeps its first `maximum` statements and hands the rest to what
    # runs when none of them raised: after the whole statement when every handler exits (`return`,
    # `raise`, `continue`, `break`), else in an `else:` clause. The handlers, and the comments
    # that go with each part, move untouched. Exceptions the moved statements raise are no longer
    # caught, the change in behavior this fix is reviewed for. Left alone: a `try` with `else` or
    # `finally`, one whose kept statements are themselves wide (they nest a block), a body opening
    # on the line of the `try`, one indented with tabs (the rewrite indents with spaces), and the
    # handling ceiling (a design call).
    fix = r"""
        WITH located AS (
          SELECT f.finding_key, f.path, f.node, f.file_id, t.start_line, t.start_column
          FROM finding f JOIN core_node t ON t.file_id = f.file_id AND t.node_id = f.node
          WHERE NOT EXISTS (SELECT 1 FROM source_lines() x
                            WHERE x.path = f.path AND x.line BETWEEN t.start_line AND t.end_line
                              AND regexp_matches(x.text, '^ *\t'))
        ), body AS (
          SELECT l.finding_key, s.node_id, s.start_line, s.start_column,
                 row_number() OVER (PARTITION BY l.finding_key ORDER BY s.node_id) AS rank,
                 EXISTS (SELECT 1 FROM core_node k
                         WHERE k.file_id = l.file_id AND k.type = 'block' AND k.node_id > s.node_id
                           AND k.node_id <= s.node_id + s.descendant_count) AS nested
          FROM located l
          JOIN core_node b ON b.file_id = l.file_id AND b.parent_id = l.node AND b.type = 'block'
          JOIN core_node s ON s.file_id = b.file_id AND s.parent_id = b.node_id
            AND s.type <> 'comment'
        ), unit AS (
          SELECT b.finding_key, repeat(' ', b.start_column - l.start_column) AS indent
          FROM body b JOIN located l USING (finding_key)
          WHERE b.rank = 1 AND b.start_line > l.start_line
        ), clauses AS (
          SELECT l.finding_key, list(c.node_id ORDER BY c.node_id) AS handlers,
                 bool_and(c.type = 'except_clause') AS plain,
                 bool_and(list_contains(['return_statement', 'raise_statement',
                                         'continue_statement', 'break_statement'], (
                   SELECT max_by(s.type, s.node_id) FROM core_node k JOIN core_node s
                     ON s.file_id = k.file_id AND s.parent_id = k.node_id AND s.type <> 'comment'
                   WHERE k.file_id = c.file_id AND k.parent_id = c.node_id AND k.type = 'block')))
                   AS exits
          FROM located l
          JOIN core_node c ON c.file_id = l.file_id AND c.parent_id = l.node
            AND c.type NOT IN ('block', 'try', ':', 'comment')
          GROUP BY l.finding_key
        )
        SELECT l.finding_key, 'rewrite-node' AS operation, l.path, l.node,
               MAP {'KEEP': list(b.node_id ORDER BY b.rank) FILTER (b.rank <= $maximum),
                    'TAIL': list(b.node_id ORDER BY b.rank) FILTER (b.rank > $maximum),
                    'HANDLERS': any_value(c.handlers)} AS captures,
               'try:' || chr(10) || any_value(u.indent) || '$$$KEEP' || chr(10)
                 || '$$$HANDLERS' || chr(10)
                 || if(any_value(c.exits), '', 'else:' || chr(10) || any_value(u.indent))
                 || '$$$TAIL' AS template
        FROM located l
        JOIN body b USING (finding_key)
        JOIN unit u USING (finding_key)
        JOIN clauses c USING (finding_key)
        WHERE c.plain
        GROUP BY l.finding_key, l.path, l.node
        HAVING count(*) FILTER (b.rank > $maximum) > 0
           AND NOT bool_or(b.nested AND b.rank <= $maximum)
    """
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
        text = path.read_text()
        data = parse(text)
        return data
    except FileNotFoundError:
        return None
    except OSError:
        raise
"""
            ),
            good=(
                """def load(path):
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        raise
    data = parse(text)
    return data
"""
            ),
        ),
        Example(
            bad=(
                """def load(path):
    try:
        text = path.read_text()
        # parse what was read
        parse(text)
    except OSError:
        print(path)
"""
            ),
            good=(
                """def load(path):
    try:
        text = path.read_text()
    except OSError:
        print(path)
    else:
        # parse what was read
        parse(text)
"""
            ),
        ),
        Example(
            bad=(
                """def load(path):
    try:
        text = path.read_text()
        parse(text)
    finally:
        print(path)
"""
            ),
            good=(
                """def load(path):
    try:
        text = path.read_text()
        parse(text)
    finally:
        print(path)
"""
            ),
        ),
        Example(
            bad=(
                """def load(path):
    try:
        if path:
            text = path.read_text()
        use(path)
    except OSError:
        return None
"""
            ),
            good=(
                """def load(path):
    try:
        if path:
            text = path.read_text()
        use(path)
    except OSError:
        return None
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
        report(path)
        raise
"""
            ),
        ),
        Example(
            bad=(
                """def load(path):
    try:
        text = path.read_text()
        return parse(text)
    except OSError:
        log(path)
        clean(path)
        raise
"""
            ),
            good=(
                """def load(path):
    try:
        text = path.read_text()
    except OSError:
        log(path)
        clean(path)
        raise
    return parse(text)
"""
            ),
        ),
    )


# An exception that sits right in an `exceptions.py` is imported only from within the folder holding
# it; the importers of its top package (a test suite apart) stand outside that folder when they
# are not that package or one below it.
_STRAYED = r"""
    regexp_matches(f.path, '(^|/)' || $preferred_module || '\.pyi?$')
    AND array_to_string(list_slice(string_split(e.defining_module, '.'), 1, -2), '.') <> ''
    AND len(list_filter(
          e.importers,
          lambda m: split_part(m, '.', 1) = split_part(e.defining_module, '.', 1)
            AND NOT starts_with(m || '.', array_to_string(
                list_slice(string_split(e.defining_module, '.'), 1, -2), '.') || '.'))) > 0
"""


class SharedExceptionPlacement(Rule, id="PY-EXCE0003", applicability="review"):
    """Exception classes imported by other modules, outside the `exceptions.py` of the lowest
    package holding the class and every one of its importers."""

    minimum_importing_modules: int = 2
    preferred_module: str = "exceptions"

    query = f"""
            SELECT f.path, f.start_line, f.start_column, f.fact_id, count(*) AS value
            FROM exception_exceptions e JOIN exception_facts f USING (fact_id)
            LEFT JOIN (SELECT fact_id, parent_id, count(DISTINCT string_value) AS importers
                       FROM exception_values WHERE relation = 'exceptions.importing_modules'
                       GROUP BY fact_id, parent_id) i
              ON i.fact_id = e.fact_id AND i.parent_id = e.record_id
            WHERE CASE WHEN ends_with(e.defining_module, '.' || $preferred_module)
                       THEN {_STRAYED}
                       ELSE coalesce(i.importers, 0) >= $minimum_importing_modules END
            GROUP BY ALL
        """
    message = "shared exception placement is {value} for `{fact_id}`"
    # One move per finding (`move-definitions`), all the module's shared exceptions at once, into
    # the `exceptions.py` of the lowest package holding the module and every importer of its top
    # package (a test suite apart), with each importing statement respelled. An exception already
    # in an `exceptions.py` moves up when an importer stands outside the folder of that file.
    # Left alone: what the operation refuses (a class reading a name of its module that stays, a
    # destination defining the name otherwise, an importer binding the module as `import a as b`,
    # a cycle).
    fix = rf"""
        WITH moved AS (
          SELECT f.finding_key, f.path, e.name, e.defining_module, e.importers
          FROM finding f JOIN exception_exceptions e USING (fact_id)
          WHERE CASE WHEN ends_with(e.defining_module, '.' || $preferred_module)
                     THEN {_STRAYED}
                     ELSE len(e.importers) >= $minimum_importing_modules END
        ), touching AS (
          SELECT m.finding_key, if(c.is_package, string_split(c.module, '.'),
                                   list_slice(string_split(c.module, '.'), 1, -2)) AS package
          FROM moved m JOIN core_file c
            ON c.module = m.defining_module
               OR (list_contains(m.importers, c.module)
                   AND split_part(c.module, '.', 1) = split_part(m.defining_module, '.', 1))
        ), common AS (
          SELECT finding_key, list_reduce(list(package), lambda a, b: list_slice(a, 1,
                   coalesce(list_position(list_transform(range(1, least(len(a), len(b)) + 1),
                                                         lambda k: a[k] = b[k]), false) - 1,
                            least(len(a), len(b))))) AS package
          FROM touching GROUP BY finding_key
        ), placed AS (
          SELECT m.finding_key, m.path, list(m.name ORDER BY m.name) AS names,
                 any_value(string_split(regexp_replace(m.path, '/[^/]*$', ''), '/')) AS folders,
                 any_value(len(string_split(m.defining_module, '.')) - 1
                           + is_initializer_path(m.path)::INTEGER) AS depth,
                 any_value(len(c.package)) AS kept
          FROM moved m JOIN common c USING (finding_key)
          GROUP BY m.finding_key, m.path
        ), destined AS (
          SELECT finding_key, path, names,
                 array_to_string(list_slice(folders, 1, len(folders) - (depth - kept)), '/')
                   || '/' || $preferred_module || '.py' AS destination
          FROM placed WHERE kept > 0
        )
        SELECT d.finding_key, 'move-definitions' AS operation, r.*
        FROM destined d, move_row(d.path, d.names, d.destination) r
    """
    examples = (
        Example(
            bad="from .a.errs import Boom\n\n\ndef f():\n    raise Boom\n",
            good="from .exceptions import Boom\n\n\ndef f():\n    raise Boom\n",
            path="src/pkg/u1.py",
            context=_SHARED,
        ),
        Example(
            bad="from ..a.errs import Boom\n\n\ndef f():\n    raise Boom\n",
            good="from ..exceptions import Boom\n\n\ndef f():\n    raise Boom\n",
            path="src/pkg/b/u1.py",
            context=_SHARED,
        ),
        Example(
            bad="from .a import Boom\n\n\ndef f():\n    raise Boom\n",
            good="from .exceptions import Boom\n\n\ndef f():\n    raise Boom\n",
            path="src/pkg/u1.py",
            context={
                **_PACKAGE,
                "src/pkg/a/__init__.py": _BOOM,
                "src/pkg/b/u2.py": "from ..a import Boom\n\n\ndef g():\n    raise Boom\n",
            },
        ),
        Example(
            bad="from .a.errs import Boom\n\n\ndef f():\n    raise Boom\n",
            good="from .a.errs import Boom\n\n\ndef f():\n    raise Boom\n",
            path="src/pkg/u1.py",
            context={
                **_SHARED,
                "src/pkg/a/errs.py": "LIMIT = 3\n\n\nclass Boom(Exception):\n    limit = LIMIT\n",
            },
        ),
        Example(
            bad="from ..a.exceptions import Boom\n\n\ndef g():\n    raise Boom\n",
            good="from ..exceptions import Boom\n\n\ndef g():\n    raise Boom\n",
            path="src/pkg/b/u2.py",
            context={
                **_PACKAGE,
                "src/pkg/a/exceptions.py": _BOOM,
                "src/pkg/a/u1.py": "from .exceptions import Boom\n\n\ndef f():\n    raise Boom\n",
            },
        ),
    )
