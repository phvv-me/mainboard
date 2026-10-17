"""The names a module binds at its top level, which a fix's new name must not collide with."""

from ...catalog import Relation


class ControlArtifactsModuleNames(Relation):
    """Every name a Python module binds at its top level (functions, classes, imports and
    assigned names), by file: what a name a fix adds must not collide with."""

    query = r"""
    SELECT f.path, s.name FROM symbol_symbols s JOIN symbol_facts f USING (fact_order, fact_id)
    WHERE s.scope = 'module'
    UNION ALL SELECT path, name FROM function_functions WHERE scope = 'module'
    UNION ALL SELECT path, name FROM class_classes WHERE scope = 'module'
    UNION ALL SELECT path, CASE WHEN contains(text, ' as ')
                                THEN regexp_extract(text, '(\w+)\s*$', 1)
                                ELSE split_part(text, '.', 1) END
    FROM import_binding_nodes WHERE role = 'binding'
"""
