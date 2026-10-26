"""SQL the call-side families (calls, functions, parameters, attribute accesses) share. The scalar
macros here (`calls_member_visibility`, `calls_annotation_name`) are installed before families
are computed. `PythonBase` names the private tables one family builds first (its Python files
with their module names, a narrow copy of their syntax nodes, their lines, every function and
class with its dotted name, and the names each file imports) and later relations read; `texts`
cuts source text for the spans a family asks for."""

from ....catalog import Function


class CallsMemberVisibility(Function):
    """`public`, `protected` (one leading underscore) or `private` (two) for an attribute name;
    double-underscore protocol names are public."""

    name: str
    expression = """CASE WHEN starts_with(name, '__') AND ends_with(name, '__') THEN 'public'
                         WHEN starts_with(name, '__') THEN 'private'
                         WHEN starts_with(name, '_') THEN 'protected' ELSE 'public' END"""


class CallsAnnotationName(Function):
    """The name an annotation's source text stands for: a string's content, a subscript's
    subject, a dotted name's last segment."""

    text: str
    expression = "regexp_extract(split_part(trim(text, chr(34) || chr(39)), '[', 1), '[^.]+$')"


def listed(values: list[str] | tuple[str, ...]) -> str:
    """A SQL list literal of strings."""
    return "[" + ", ".join("'" + value.replace("'", "''") + "'" for value in values) + "]"


def _line_slice(text: str, *, plain: str, start: str, end: str) -> str:
    """SQL for the bytes `start` to `end` of the line `text`.

    `plain` says the line is ASCII, where characters are bytes and `substr` is cheap.
    """
    return (
        f"CASE WHEN {plain} THEN substr({text}, {start} + 1, {end} - {start}) "
        f"ELSE decode(array_slice(encode({text}), {start} + 1, {end})) END"
    )


class PythonBase:
    """The private tables a family builds first, named with the family's prefix, and the SQL
    that creates them (`relations()` lists them in dependency order). Their columns avoid
    `fact_id`, which the engine would narrow by language."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.files, self.nodes, self.lines, self.defs, self.imports = (
            f'"{family}/{name}"' for name in ("files", "nodes", "lines", "defs", "imports")
        )

    def _defs(self) -> str:
        """Every function and class definition, where it sits and its dotted name.

        The name reads `module.Class.method`, and `kind` is `class`, `method` (a function directly
        in a class) or `function`.
        """
        return f"""
            WITH RECURSIVE named AS (
              SELECT file, node, type, name, scope, start_line, start_column, end_line,
                     end_column, descendant_count, klass, func
              FROM {self.nodes} WHERE type IN ('function_definition', 'class_definition')
            ), qualified(file, node, qualified_name, owner_kind) AS (
              SELECT d.file, d.node, f.module || '.' || d.name, 'module'
              FROM named d JOIN {self.files} f USING (file) WHERE d.scope = 0
              UNION ALL
              SELECT d.file, d.node, q.qualified_name || '.' || d.name,
                     CASE WHEN p.type = 'class_definition' THEN 'class' ELSE 'function' END
              FROM named d JOIN qualified q ON q.file = d.file AND q.node = d.scope
              JOIN named p ON p.file = q.file AND p.node = q.node
            )
            SELECT d.*, q.qualified_name, q.owner_kind,
                   CASE WHEN d.type = 'class_definition' THEN 'class'
                        WHEN q.owner_kind = 'class' THEN 'method' ELSE 'function' END AS kind
            FROM named d JOIN qualified q USING (file, node)
        """

    def _imports(self) -> str:
        """Every name an `import` or `from ... import` statement binds.

        Each row holds the `binding` a file reads the name by and the dotted `target` it stands
        for, a relative module resolved against the importing module as mcmr does.
        """
        return f"""
            WITH statements AS (
              SELECT n.file, f.module, f.is_package, n.node, n.type, n.name AS text
              FROM {self.nodes} n JOIN {self.files} f USING (file)
              WHERE n.type IN ('import_statement', 'import_from_statement')
            ), items AS (
              SELECT s.file, s.module, s.is_package, s.node AS statement,
                     s.type AS statement_type, s.text AS module_text, c.node AS item,
                     c.type AS item_type, c.name AS item_name, d.name AS imported
              FROM statements s
              JOIN {self.nodes} c ON c.file = s.file AND c.parent = s.node
                AND c.type IN ('dotted_name', 'aliased_import')
                AND (s.type = 'import_statement' OR c.sibling_index >= 3)
              LEFT JOIN {self.nodes} d ON d.file = c.file AND d.parent = c.node
                AND d.sibling_index = 0 AND c.type = 'aliased_import'
            ), resolved AS (
              SELECT *, length(module_text) - length(ltrim(module_text, '.')) AS level,
                     ltrim(module_text, '.') AS rest,
                     CASE WHEN is_package THEN string_split(module, '.')
                          ELSE list_slice(string_split(module, '.'), 1, -2) END AS parts
              FROM items WHERE statement_type = 'import_from_statement'
            ), joined AS (
              SELECT *, CASE
                WHEN level = 0 THEN rest
                WHEN level > len(parts) THEN repeat('.', level) || rest
                WHEN len(parts) + 1 - level = 0 THEN rest
                WHEN rest = '' THEN array_to_string(
                  list_slice(parts, 1, len(parts) + 1 - level), '.')
                ELSE array_to_string(
                  list_slice(parts, 1, len(parts) + 1 - level), '.') || '.' || rest
                END AS source_module
              FROM resolved
            )
            SELECT file, statement, item, item_name AS binding,
                   CASE WHEN item_type = 'aliased_import' THEN imported ELSE item_name END
                     AS target,
                   false AS from_import
            FROM items WHERE statement_type = 'import_statement'
            UNION ALL
            SELECT file, statement, item, item_name AS binding,
                   source_module || '.' || CASE WHEN item_type = 'aliased_import' THEN imported
                                                ELSE item_name END AS target,
                   true AS from_import
            FROM joined
        """

    def _lines(self) -> str:
        """Every line of every Python file, numbered from 1, without its line break.

        Each row also says whether the line is plain ASCII, since columns are UTF-8 bytes.
        """
        return f"""
            SELECT f.file, l.n::INT AS n, l.text, strlen(l.text) = length(l.text) AS plain
            FROM {self.files} f
            JOIN (SELECT path, generate_subscripts(string_split(content, chr(10)), 1) AS n,
                         unnest(string_split(content, chr(10))) AS text
                  FROM source_files) l USING (path)
        """

    def _nodes(self) -> str:
        """The syntax nodes of the Python files, narrowed to the columns the families read.

        They are keyed by file id and node id. `scope` is the innermost function, class, lambda or
        module a node sits in (a definition node sits in the scope that holds it); sitting_duck
        also counts a `for` or `while` statement as a scope, which is looked through here.
        """
        return f"""
            WITH RECURSIVE raw AS (
              SELECT f.file, n.node_id::INT AS node, n.parent_id::INT AS parent, n.type,
                     CASE WHEN n.type IN ('string', 'string_content', 'concatenated_string',
                                          'comment', 'interpolation') THEN NULL ELSE n.name END
                       AS name,
                     n.sibling_index::INT AS sibling_index, n.start_line::INT AS start_line,
                     n.start_column::INT AS start_column, n.end_line::INT AS end_line,
                     n.end_column::INT AS end_column, n.descendant_count::INT AS descendant_count,
                     n.children_count::INT AS children_count, n.scope.current::INT AS scope,
                     n.scope.function::INT AS func, n.scope.class::INT AS klass
              FROM ast_nodes n JOIN {self.files} f ON f.ast_fact = n.fact_id
            ), loops AS (
              SELECT file, node, scope FROM raw
              WHERE type IN ('for_statement', 'while_statement')
            ), chase(file, node, up) AS (
              SELECT file, node, scope FROM loops
              UNION ALL
              SELECT c.file, c.node, l.scope
              FROM chase c JOIN loops l ON l.file = c.file AND l.node = c.up
            ), outside AS (
              SELECT c.file, c.node, c.up AS scope FROM chase c
              ANTI JOIN loops l ON l.file = c.file AND l.node = c.up
            )
            SELECT r.file, r.node, r.parent, r.type, r.name, r.sibling_index, r.start_line,
                   r.start_column, r.end_line, r.end_column, r.descendant_count,
                   r.children_count, coalesce(o.scope, r.scope) AS scope, r.func,
                   r.klass
            FROM raw r LEFT JOIN outside o ON o.file = r.file AND o.node = r.scope
        """

    def module_files(self) -> str:
        """One row per Python file: its integer id, path and dotted module name.

        mcmr names a module by walking up `__init__.py` directories to an import root, and joins a
        namespace portion to a package of the same name under another root.
        """
        return r"""
            WITH py AS (
              SELECT row_number() OVER (ORDER BY path)::INT AS file, fact_id, path
              FROM ast_files WHERE language = 'python'
            ), regular AS (
              SELECT DISTINCT regexp_replace(path, '(^|/)[^/]*$', '') AS directory
              FROM py WHERE regexp_matches(path, '(^|/)__init__\.py$')
            ), outer_packages AS (
              SELECT p.directory FROM regular p
              WHERE p.directory <> '' AND NOT EXISTS (
                SELECT 1 FROM regular q WHERE q.directory <> '' AND q.directory <> p.directory
                  AND starts_with(p.directory, q.directory || '/'))
            ), package_names AS (
              SELECT DISTINCT regexp_extract(directory, '[^/]*$') AS name FROM outer_packages
            ), ancestors AS (
              SELECT DISTINCT array_to_string(list_slice(parts, 1, i), '/') AS directory
              FROM (SELECT list_slice(string_split(path, '/'), 1, -2) AS parts FROM py),
                   range(1, 40) AS r(i)
              WHERE i <= len(parts)
            ), namespaces AS (
              SELECT a.directory FROM ancestors a
              WHERE regexp_extract(a.directory, '[^/]*$') IN (SELECT name FROM package_names)
                AND NOT EXISTS (
                  SELECT 1 FROM regular r
                  WHERE (r.directory = a.directory OR starts_with(r.directory, a.directory || '/'))
                    AND regexp_extract(r.directory, '[^/]*$')
                      = regexp_extract(a.directory, '[^/]*$'))
            ), directories AS (
              SELECT directory FROM regular WHERE directory <> ''
              UNION SELECT directory FROM namespaces
            ), roots AS (
              SELECT DISTINCT regexp_replace(p.directory, '(^|/)[^/]*$', '') AS root
              FROM directories p
              WHERE NOT EXISTS (SELECT 1 FROM directories q WHERE q.directory <> p.directory
                                AND starts_with(p.directory, q.directory || '/'))
                AND contains(p.directory, '/')
            )
            SELECT py.file, py.fact_id AS ast_fact, py.path,
                   replace(regexp_replace(regexp_replace(
                     CASE WHEN r.root IS NULL THEN py.path ELSE substr(py.path, length(r.root) + 2)
                     END, '\.pyi?$', ''), '/__init__$', ''), '/', '.') AS module,
                   regexp_matches(py.path, '(^|/)__init__\.py$') AS is_package
            FROM py LEFT JOIN LATERAL (
              SELECT max_by(root, length(root)) AS root FROM roots
              WHERE starts_with(py.path, root || '/')) r ON true
        """

    def relations(self) -> dict[str, str]:
        return {
            "files": self.module_files(),
            "nodes": self._nodes(),
            "lines": self._lines(),
            "defs": self._defs(),
            "imports": self._imports(),
        }

    def texts(self, spans: str, lines: str | None = None) -> str:
        """The source text of every distinct span `spans` selects.

        Spans carry `file`, `start_line`, `start_column`, `end_line` and `end_column`, in UTF-8
        byte columns and lines from 1. A span on one line cuts its line, a longer one joins the
        pieces of the lines it covers (`lines` defaults to the Python files' own).
        """
        lines = lines or self.lines
        single = _line_slice("l.text", plain="l.plain", start="s.start_column", end="s.end_column")
        head = _line_slice("l.text", plain="l.plain", start="e.start_column", end="strlen(l.text)")
        tail = _line_slice("l.text", plain="l.plain", start="0", end="e.end_column")
        return f"""
            WITH spans AS (SELECT DISTINCT file, start_line, start_column, end_line, end_column
                           FROM ({spans}))
            SELECT s.file, s.start_line, s.start_column, s.end_line, s.end_column, {single} AS text
            FROM spans s JOIN {lines} l ON l.file = s.file AND l.n = s.start_line
            WHERE s.start_line = s.end_line
            UNION ALL
            SELECT file, start_line, start_column, end_line, end_column,
                   string_agg(piece, chr(10) ORDER BY n) AS text
            FROM (
              SELECT e.file, e.start_line, e.start_column, e.end_line, e.end_column, e.n,
                     CASE WHEN e.n = e.start_line THEN {head}
                          WHEN e.n = e.end_line THEN {tail}
                          ELSE l.text END AS piece
              FROM (SELECT *, unnest(range(start_line, end_line + 1))::INT AS n
                    FROM spans WHERE start_line < end_line) e
              JOIN {lines} l ON l.file = e.file AND l.n = e.n)
            GROUP BY file, start_line, start_column, end_line, end_column
        """
