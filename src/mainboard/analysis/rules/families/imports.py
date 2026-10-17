"""mcmr's ImportBindingFact, derived from sitting_duck's syntax trees: one fact per name an import
statement binds, with its statement, binding and module spans as nodes.

The name a binding reads is counted the way mcmr counts it (every `identifier` in a name position
of the file, shadowing ignored), because deleting an import over a shadowed name is a rule's
decision, not the fact's. A string inside an annotation, a subscript or a typing constructor
(`cast`, `TypeVar` ...) counts the names it spells when it reads as an expression, as a forward
reference does; `Literal[...]` and the extras of `Annotated[...]` are text.
"""

from ...catalog import Family
from .imports_common import ALL_STRINGS, LINES, MODULE_NAMES, PYTHON, name_reads

_SYNTAX = (
    PYTHON
    + r"""
  , statements AS MATERIALIZED (
    SELECT fact_id, path, node_id AS statement, type, name AS statement_name,
           start_line, start_column, end_line, end_column
    FROM ast_nodes
    WHERE type IN ('import_statement', 'import_from_statement', 'future_import_statement')
      AND fact_id IN (SELECT fact_id FROM python_files)
  ),
  parts AS MATERIALIZED (
    SELECT s.fact_id, s.statement, c.node_id AS part, c.type AS part_type, c.name AS part_name,
           c.sibling_index AS position, c.start_line, c.start_column, c.end_line, c.end_column
    FROM statements s
    JOIN ast_nodes c ON c.fact_id = s.fact_id AND c.parent_id = s.statement
    WHERE c.type IN ('dotted_name', 'aliased_import', 'wildcard_import', 'relative_import',
                     'import', '__future__')
  ),
  keywords AS (
    SELECT fact_id, statement, min(position) AS keyword FROM parts
    WHERE part_type = 'import' GROUP BY ALL
  ),
  from_modules AS (
    SELECT p.fact_id, p.statement,
           coalesce(d.node_id, CASE WHEN p.part_type <> 'relative_import' THEN p.part END)
             AS node,
           coalesce(d.start_line, p.start_line) AS start_line,
           coalesce(d.start_column, p.start_column) AS start_column,
           coalesce(d.end_line, p.end_line) AS end_line,
           coalesce(d.end_column, p.end_column) AS end_column
    FROM parts p
    JOIN keywords k ON k.fact_id = p.fact_id AND k.statement = p.statement
    LEFT JOIN ast_nodes d ON d.fact_id = p.fact_id AND d.parent_id = p.part
      AND p.part_type = 'relative_import' AND d.type = 'dotted_name'
    WHERE p.part_type IN ('dotted_name', 'relative_import', '__future__')
      AND p.position < k.keyword
  ),
  bound AS (
    SELECT s.fact_id, s.path, s.statement, s.type AS statement_type, s.statement_name,
           s.start_line, s.start_column, s.end_line, s.end_column,
           p.part AS binding, p.position,
           p.start_line AS binding_start_line, p.start_column AS binding_start_column,
           p.end_line AS binding_end_line, p.end_column AS binding_end_column,
           CASE p.part_type WHEN 'aliased_import' THEN a.name
                            WHEN 'wildcard_import' THEN '*' ELSE p.part_name END AS imported_name,
           CASE WHEN p.part_type = 'aliased_import' THEN p.part_name
                WHEN p.part_type = 'wildcard_import' THEN '*'
                WHEN s.type = 'import_statement' THEN split_part(p.part_name, '.', 1)
                ELSE p.part_name END AS name,
           p.part_type = 'aliased_import' AS aliased,
           CASE WHEN s.type = 'import_statement' THEN coalesce(a.node_id, p.part)
                ELSE m.node END AS module_node,
           CASE WHEN s.type = 'import_statement' THEN coalesce(a.start_line, p.start_line)
                ELSE m.start_line END AS module_start_line,
           CASE WHEN s.type = 'import_statement' THEN coalesce(a.start_column, p.start_column)
                ELSE m.start_column END AS module_start_column,
           CASE WHEN s.type = 'import_statement' THEN coalesce(a.end_line, p.end_line)
                ELSE m.end_line END AS module_end_line,
           CASE WHEN s.type = 'import_statement' THEN coalesce(a.end_column, p.end_column)
                ELSE m.end_column END AS module_end_column,
           count(*) OVER (PARTITION BY s.fact_id, s.statement) AS bindings
    FROM statements s
    JOIN keywords k ON k.fact_id = s.fact_id AND k.statement = s.statement
    JOIN parts p ON p.fact_id = s.fact_id AND p.statement = s.statement
      AND p.part_type IN ('dotted_name', 'aliased_import', 'wildcard_import')
      AND p.position > k.keyword
    LEFT JOIN ast_nodes a ON a.fact_id = p.fact_id AND a.parent_id = p.part
      AND p.part_type = 'aliased_import' AND a.type = 'dotted_name'
    LEFT JOIN from_modules m ON m.fact_id = s.fact_id AND m.statement = s.statement
  )
"""
)

_ANALYSIS = r"""
  bodies AS (
    SELECT fact_id, parent_id, min(node_id) AS node_id,
           min_by(descendant_count, node_id) AS descendant_count
    FROM ast_nodes WHERE type = 'block' GROUP BY fact_id, parent_id
  ),
  guarded AS (
    SELECT DISTINCT s.fact_id, s.statement
    FROM statements s
    JOIN ast_nodes t ON t.fact_id = s.fact_id AND t.type = 'try_statement'
      AND s.statement BETWEEN t.node_id + 1 AND t.node_id + t.descendant_count
    JOIN bodies body
      ON body.fact_id = t.fact_id AND body.parent_id = t.node_id
     AND s.statement BETWEEN body.node_id AND body.node_id + body.descendant_count
    JOIN ast_nodes e ON e.fact_id = t.fact_id AND e.parent_id = t.node_id
      AND e.type = 'except_clause'
    WHERE NOT EXISTS (SELECT 1 FROM ast_nodes x WHERE x.fact_id = e.fact_id
                        AND x.parent_id = e.node_id
                        AND x.type NOT IN ('except', 'block', 'comment'))
       OR EXISTS (SELECT 1 FROM ast_nodes x WHERE x.fact_id = e.fact_id AND x.type = 'identifier'
                    AND x.name IN ('ImportError', 'ModuleNotFoundError')
                    AND x.node_id BETWEEN e.node_id + 1 AND e.node_id + e.descendant_count
                    AND x.node_id < (SELECT max(b.node_id) FROM ast_nodes b
                                     WHERE b.fact_id = e.fact_id AND b.parent_id = e.node_id
                                       AND b.type = 'block'))
  ),
  type_only AS (
    SELECT DISTINCT s.fact_id, s.statement
    FROM statements s
    JOIN ast_nodes i ON i.fact_id = s.fact_id AND i.type = 'if_statement'
      AND s.statement BETWEEN i.node_id + 1 AND i.node_id + i.descendant_count
    JOIN ast_nodes c ON c.fact_id = i.fact_id AND c.parent_id = i.node_id
      AND c.sibling_index = 1 AND c.type = 'identifier' AND c.name = 'TYPE_CHECKING'
    JOIN bodies b
      ON b.fact_id = i.fact_id AND b.parent_id = i.node_id
     AND s.statement BETWEEN b.node_id AND b.node_id + b.descendant_count
  ),
  wanted AS MATERIALIZED (SELECT DISTINCT fact_id, name FROM bound),
  name_reads AS (
    SELECT i.fact_id, i.name, count(*) AS reads
    FROM ast_nodes i
    JOIN wanted w ON w.fact_id = i.fact_id AND w.name = i.name
    @JOINS@
    WHERE @FILTER@
    GROUP BY ALL
  ),
  containers AS MATERIALIZED (
    SELECT s.fact_id, s.node_id AS container, s.node_id + s.descendant_count AS last,
           regexp_extract(v.name, '[A-Za-z_]\w*$') AS base,
           v.node_id + v.descendant_count AS after_base
    FROM ast_nodes s
    JOIN ast_nodes v ON v.fact_id = s.fact_id AND v.parent_id = s.node_id AND v.sibling_index = 0
    WHERE s.type = 'subscript'
    UNION ALL
    SELECT g.fact_id, p.node_id, p.node_id + p.descendant_count,
           regexp_extract(v.name, '[A-Za-z_]\w*$'), v.node_id + v.descendant_count
    FROM ast_nodes g
    JOIN ast_nodes v ON v.fact_id = g.fact_id AND v.parent_id = g.node_id AND v.sibling_index = 0
    JOIN ast_nodes p ON p.fact_id = g.fact_id AND p.parent_id = g.node_id
      AND p.type = 'type_parameter'
    WHERE g.type = 'generic_type'
  ),
  typed_regions AS (
    SELECT fact_id, node_id AS first, node_id + descendant_count AS last FROM ast_nodes
    WHERE type = 'type'
    UNION ALL
    SELECT fact_id, after_base + 1, last FROM containers
    UNION ALL
    SELECT a.fact_id, a.node_id, a.node_id + a.descendant_count
    FROM ast_nodes c
    JOIN ast_nodes a ON a.fact_id = c.fact_id AND a.parent_id = c.node_id
      AND a.type = 'argument_list'
    WHERE c.type = 'call' AND c.name IN ('cast', 'NamedTuple', 'NewType', 'ParamSpec',
            'TypeAliasType', 'TypeVar', 'TypeVarTuple', 'TypedDict')
  ),
  text_regions AS (
    SELECT fact_id, after_base + 1 AS first, last FROM containers WHERE base = 'Literal'
    UNION ALL
    SELECT c.fact_id, min(e.node_id + e.descendant_count) + 1, any_value(c.last)
    FROM containers c
    JOIN ast_nodes e ON e.fact_id = c.fact_id AND e.parent_id = c.container
      AND e.node_id > c.after_base AND e.type NOT IN ('[', ']', ',', '(', ')')
    WHERE c.base = 'Annotated'
    GROUP BY c.fact_id, c.container
  ),
  forward_references AS (
    SELECT DISTINCT s.fact_id, s.node_id,
           regexp_replace(trim(regexp_replace(s.name, '^[rRuU]', ''), '"'''), '\s+', ' ', 'g')
             AS text
    FROM ast_nodes s
    JOIN typed_regions r ON r.fact_id = s.fact_id AND s.node_id BETWEEN r.first AND r.last
    WHERE s.type = 'string' AND regexp_matches(s.name, '^[rRuU]?["'']')
      AND s.fact_id IN (SELECT fact_id FROM python_files)
      AND NOT EXISTS (SELECT 1 FROM text_regions h WHERE h.fact_id = s.fact_id
                        AND s.node_id BETWEEN h.first AND h.last)
  ),
  string_reads AS (
    SELECT q.fact_id, q.word AS name, count(*) AS reads
    FROM (
      SELECT fact_id, unnest(regexp_extract_all(
               regexp_replace(
                 regexp_replace(regexp_replace(text, 'Literal\s*\[[^\]]*\]', 'Literal', 'g'),
                                '["'']', ' ', 'g'),
                 '\.\s*[A-Za-z_]\w*', '', 'g'), '[A-Za-z_]\w*')) AS word
      FROM forward_references
      WHERE NOT regexp_matches(
              regexp_replace(text, '\b(and|or|if|else|in|is|not|for|lambda|await|as)\b', ',', 'g'),
              '[A-Za-z0-9_\])]\s+[A-Za-z_]')
        AND NOT regexp_matches(text, '^[/*%<>=,)\]}.]|[-+*/%<>=,:.&|^~(\[{]$')
        AND (regexp_matches(text, '[\[({]') OR NOT regexp_matches(text, ':|(^|[^=!<>])=($|[^=])'))
    ) q
    JOIN wanted w ON w.fact_id = q.fact_id AND w.name = q.word
    GROUP BY ALL
  ),
  reads AS (
    SELECT fact_id, name, sum(reads) AS reads
    FROM (SELECT * FROM name_reads UNION ALL SELECT * FROM string_reads) GROUP BY ALL
  ),
  exported AS (SELECT DISTINCT fact_id, value FROM all_strings),
  facts AS (
    SELECT b.*, m.module AS importer_module,
           CASE b.statement_type WHEN 'import_statement' THEN b.imported_name
                                 WHEN 'future_import_statement' THEN '__future__'
                                 ELSE ltrim(b.statement_name, '.') END AS module,
           CASE WHEN b.statement_type = 'import_from_statement'
                THEN length(b.statement_name) - length(ltrim(b.statement_name, '.')) ELSE 0
           END AS relative_level,
           coalesce(r.reads, 0) AS reference_count,
           g.statement IS NOT NULL AS has_documented_side_effect,
           t.statement IS NOT NULL AS is_type_only,
           x.fact_id IS NOT NULL OR (b.aliased AND b.name = b.imported_name) AS is_reexported
    FROM bound b
    JOIN module_names m ON m.fact_id = b.fact_id
    LEFT JOIN reads r ON r.fact_id = b.fact_id AND r.name = b.name
    LEFT JOIN guarded g ON g.fact_id = b.fact_id AND g.statement = b.statement
    LEFT JOIN type_only t ON t.fact_id = b.fact_id AND t.statement = b.statement
    LEFT JOIN exported x ON x.fact_id = b.fact_id AND x.value = b.name
  )
"""

_JOINS, _FILTER = name_reads()
_ANALYSIS = _ANALYSIS.replace("@JOINS@", _JOINS).replace("@FILTER@", _FILTER)


class ImportBinding(Family):
    family = "ImportBindingFact"
    verified = True
    relations = {
        "facts": f"""
            WITH {MODULE_NAMES}, {ALL_STRINGS}, {_SYNTAX}, {_ANALYSIS}, {LINES},
            shaped AS (
              SELECT f.*, l.lines, split_part(f.module, '.', 1) AS module_root,
                     split_part(f.importer_module, '.', 1) AS project
              FROM facts f JOIN lines l ON l.path = f.path
            )
            SELECT (row_number() OVER (ORDER BY path, start_line, start_column, position)
                    - 1)::UBIGINT AS fact_order,
                   'import:' || path || ':' || name AS fact_id, path,
                   start_line::UBIGINT AS start_line, start_column::UBIGINT AS start_column,
                   end_line::UBIGINT AS end_line, end_column::UBIGINT AS end_column,
                   'python' AS language, name, module, imported_name, importer_module,
                   path || ':' || statement || ':import' AS declaration_id,
                   import_source_slice(lines, start_line, start_column, end_line, end_column)
                     AS declaration_text,
                   path || ':' || binding || ':sequence-item' AS binding_id,
                   coalesce(path || ':' || module_node || ':module', '') AS module_node_id,
                   reference_count::UBIGINT AS reference_count,
                   relative_level::UBIGINT AS relative_level,
                   reference_count > 0 AS has_qualifying_use,
                   relative_level = 0 AND module_root <> project AS is_external,
                   is_reexported, is_type_only, has_documented_side_effect,
                   relative_level > 0 AS is_relative,
                   relative_level > 0 OR module_root = project AS is_project_owned,
                   bindings = 1 AS is_sole_binding,
                   list_bool_or(list_transform(string_split(module, '.'),
                     lambda part: starts_with(part, '_') AND NOT starts_with(part, '__')))
                     AS has_private_module_component,
                   starts_with(imported_name, '_') AND NOT starts_with(imported_name, '__')
                     AS is_private_member,
                   starts_with(imported_name, '_') AND NOT starts_with(imported_name, '__')
                     AND NOT regexp_matches(substr(imported_name, 2), '\\p{{Ll}}')
                     AS is_private_uppercase_constant,
                   imported_name = '*' AS is_wildcard
            FROM shaped
        """,
        "nodes": f"""
            WITH {LINES}, placed AS (
              SELECT fact_id, path, 'declaration' AS role, declaration_id AS node_id,
                     'import' AS kind FROM import_binding_facts
              UNION ALL
              SELECT fact_id, path, 'binding', binding_id, 'sequence-item'
              FROM import_binding_facts
              UNION ALL
              SELECT fact_id, path, 'module', module_node_id, 'module'
              FROM import_binding_facts WHERE module_node_id <> ''
            )
            SELECT p.fact_id, p.role, 0::UBIGINT AS ordinal, p.node_id, p.path,
                   n.start_line::UBIGINT AS start_line, n.start_column::UBIGINT AS start_column,
                   n.end_line::UBIGINT AS end_line, n.end_column::UBIGINT AS end_column, p.kind,
                   import_source_slice(l.lines, n.start_line, n.start_column, n.end_line,
                                       n.end_column) AS text
            FROM placed p
            JOIN ast_files a ON a.path = p.path
            JOIN ast_nodes n ON n.fact_id = a.fact_id
              AND n.node_id = regexp_extract(p.node_id, ':([0-9]+):[a-z-]+$', 1)::BIGINT
            JOIN lines l ON l.path = p.path
        """,
    }
