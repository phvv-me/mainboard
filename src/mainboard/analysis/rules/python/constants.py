"""Rules over module constants (mcmr's PY-CONS family)."""

from ...catalog import Example, Relation, Rule


class PlacedSymbols(Relation):
    """The `placed_symbols()` relation."""

    query = """
    SELECT s.* REPLACE (
      CASE WHEN s.reference_id IS NULL THEN f.path
           ELSE s.reference_declaration_span_path END AS path,
      CASE WHEN s.reference_id IS NULL THEN f.start_line
           ELSE s.reference_declaration_span_start_line END AS start_line,
      CASE WHEN s.reference_id IS NULL THEN f.start_column
           ELSE s.reference_declaration_span_start_column END AS start_column)
    FROM symbol_symbols s JOIN symbol_facts f USING (fact_order, fact_id)
"""


class PublicModuleConstant(Rule, id="PY-CONS0001", applicability="review"):
    """Project module constants exposed without a leading underscore."""

    query = """
            SELECT * FROM placed_symbols()
            WHERE scope = 'module' AND is_constant_assignment
              AND upper(name) = name AND lower(name) <> name AND NOT starts_with(name, '_')
        """
    message = "`{name}` is a public module constant rather than a private file detail"
    # The constant becomes `_NAME` at its declaration and every reference its module holds, when
    # the kernel saw all of them and nothing outside the module can reach the name: no import
    # of that name, no attribute access with it, no star import of the module, no `__all__`
    # entry, no `_NAME` already in the module. Names are matched project-wide (an upper bound on
    # importers), and a constant that other modules use is PY-CONS0002's to move, not renamed.
    # Review: strings and dynamic lookups (`getattr`, templates) cannot be seen.
    fix = r"""
        WITH renamable AS (
          SELECT * FROM finding f
          WHERE f.reference_id IS NOT NULL AND f.reference_are_references_complete
            AND NOT EXISTS (SELECT 1 FROM import_binding_facts i
                            WHERE i.imported_name = f.name OR i.name = f.name
                               OR (i.is_wildcard AND ends_with(
                                     regexp_replace(
                                       regexp_replace(f.path, '(/__init__)?\.py$', ''),
                                       '/', '.', 'g'),
                                     ltrim(i.module, '.'))))
            AND NOT EXISTS (SELECT 1 FROM core_ref a
                            WHERE a.parent_type = 'attribute' AND a.place = 1
                              AND a.name = f.name)
            AND NOT EXISTS (SELECT 1 FROM module_all_declarations d
                            WHERE d.span_path = f.path
                              AND (contains(d.text, '"' || f.name || '"')
                                   OR contains(d.text, '''' || f.name || '''')))
            AND NOT EXISTS (SELECT 1 FROM symbol_symbols t
                            WHERE t.reference_declaration_span_path = f.path
                              AND t.scope = 'module'
                              AND (t.name = '_' || f.name
                                   OR (t.name = f.name AND t.record_id <> f.record_id)))
        )
        SELECT finding_key, reference_declaration_span_path AS path,
               reference_declaration_span_start_line AS start_line,
               reference_declaration_span_start_column AS start_column,
               reference_declaration_span_end_line AS end_line,
               reference_declaration_span_end_column AS end_column, '_' || name AS text
        FROM renamable
        UNION ALL BY NAME
        SELECT r.finding_key, x.span_path AS path, x.span_start_line AS start_line,
               x.span_start_column AS start_column, x.span_end_line AS end_line,
               x.span_end_column AS end_column, '_' || r.name AS text
        FROM renamable r JOIN symbol_symbols_reference_references x ON x.parent_id = r.record_id
    """
    examples = (
        Example(
            bad="LIMIT = 3\n\n\ndef clamp(value):\n    return min(value, LIMIT)\n",
            good="_LIMIT = 3\n\n\ndef clamp(value):\n    return min(value, _LIMIT)\n",
        ),
    )


class CrossModuleProjectConstantImport(Rule, id="PY-CONS0002"):
    """Project-owned constants imported outside their defining module. A capitalized name the
    project defines as a function or a class (`PK`, `FK`) is a callable in the style of a column
    DSL, not a constant."""

    query = """
            SELECT i.path, i.start_line, i.start_column,
                   'import:' || i.path || ':' || i.name AS fact_id
            FROM core_import i
            WHERE i.is_project_owned
              AND regexp_matches(ltrim(if(i.imported_name <> '', i.imported_name, i.name), '_'),
                                 '^[A-Z][A-Z0-9_]*$')
              AND NOT EXISTS (SELECT 1 FROM core_def d
                              WHERE d.language = 'python' AND d.owner_id IS NULL
                                AND d.name = if(i.imported_name <> '', i.imported_name, i.name))
        """
    message = "`{fact_id}` satisfies cross module project constant import"


class DependencySafeConstantOrder(Rule, id="PY-CONS0003"):
    """Constants declared later than their earliest dependency-safe top-level position."""

    query = """
            SELECT m.path, m.start_line, m.start_column, m.fact_id, count(*) AS value
            FROM module_constant_placements c JOIN module_facts m USING (fact_id)
            WHERE c.intervening_statement_count > 0 AND NOT m.is_test
            GROUP BY ALL
        """
    message = "dependency safe constant order is {value} for `{fact_id}`"
