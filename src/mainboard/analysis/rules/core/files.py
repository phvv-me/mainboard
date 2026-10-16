"""`core_file` (one row per file) and `core_node` (one row per syntax node, the tree every other
relation and rule walks)."""

from ..families.imports_common import MODULE_NAMES, python_files

FILE = f"""
    WITH {MODULE_NAMES},
    {python_files("parsed")}
    SELECT (row_number() OVER (ORDER BY f.path))::INTEGER AS file_id, f.fact_id, f.path,
           f.language, m.module, regexp_matches(f.path, '(^|/)__init__\\.pyi?$') AS is_package,
           ends_with(f.path, '.pyi') AS is_stub,
           f.language <> 'python' OR f.fact_id IN (SELECT fact_id FROM parsed) AS parses
    FROM ast_files f LEFT JOIN module_names m USING (fact_id)
"""

NODE = """
    SELECT f.file_id, n.node_id, n.parent_id, n.sibling_index, n.type, n.semantic_type, n.flags,
           n.name, n.start_line, n.start_column, n.end_line, n.end_column, n.children_count,
           n.descendant_count, greatest(n.scope."function", n.scope."class") AS owner_id,
           n.scope."function" AS function_id, n.scope."class" AS class_id
    FROM ast_nodes n JOIN core_file f USING (fact_id)
"""
