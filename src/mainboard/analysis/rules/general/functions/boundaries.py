"""Rules over what a callable asks of its callers and hides from them (mcmr ALL-FUNC)."""

from ....catalog import Example, Rule


class TransparentUnaryWrapper(Rule, id="ALL-FUNC0005", applicability="review"):
    """A public unary function that only forwards its one argument."""

    query = """
        SELECT * FROM function_functions f
        WHERE scope IN ('module', 'method') AND visibility = 'public' AND NOT is_async
          AND returns_single_call AND forwards_only_parameter_unchanged
          AND NOT is_overload AND NOT is_recursive
          AND entity_id IN (SELECT function_id FROM function_parameters
                            GROUP BY function_id HAVING count(*) = 1)
          AND NOT EXISTS (SELECT 1 FROM function_decorators d
                          WHERE d.function_id = f.entity_id AND d.decorator <> 'staticmethod')
    """
    message = "`{fact_id}` satisfies transparent unary wrapper"
    examples = (
        Example(
            bad="from math import sqrt\n\n\ndef root(value):\n    return sqrt(value)\n",
            good="from math import sqrt\n\n\nroot = sqrt\n",
        ),
    )
    # The definition becomes the alias `name = callee`, its docstring kept as comment lines. Only
    # a module function forwarding to a plain dotted name already bound above it (an import or an
    # earlier module-level definition, so the alias can run at import time) whose one parameter
    # has no default.
    fix = r"""
        SELECT f.finding_key, f.definition_path AS path, f.definition_start_line AS start_line,
               f.definition_start_column AS start_column, f.definition_end_line AS end_line,
               f.definition_end_column AS end_column,
               CASE WHEN f.docstring <> ''
                    THEN array_to_string(
                           list_transform(
                             string_split(trim(replace(f.docstring, chr(13), '')), chr(10)),
                             lambda l: trim('# ' || trim(l))), chr(10))
                         || chr(10)
                    ELSE '' END || f.name || ' = ' || forwarded.callee AS text
        FROM finding f
        JOIN function_parameters p ON p.function_id = f.entity_id
        CROSS JOIN LATERAL (
          SELECT regexp_extract(string_split(f.definition_text, chr(10))[-1],
                                'return\s+([A-Za-z_][A-Za-z0-9_.]*)\s*\(\s*' || p.name
                                  || '\s*\)\s*$',
                                1) AS callee) forwarded
        WHERE f.scope = 'module' AND p.is_required_by_external_contract AND forwarded.callee <> ''
          AND (EXISTS (SELECT 1 FROM import_binding_facts b
                       WHERE b.path = f.path AND b.start_line < f.definition_start_line
                         AND b.start_column = 0 AND NOT b.is_type_only
                         AND b.name = split_part(forwarded.callee, '.', 1))
               OR EXISTS (SELECT 1 FROM function_functions g
                          WHERE g.path = f.path AND g.scope = 'module'
                            AND g.definition_end_line < f.definition_start_line
                            AND g.name = split_part(forwarded.callee, '.', 1))
               OR EXISTS (SELECT 1 FROM class_classes c
                          WHERE c.path = f.path AND c.scope = 'module'
                            AND c.end_line < f.definition_start_line
                            AND c.name = split_part(forwarded.callee, '.', 1)))
    """


class ReflectiveScopeRead(Rule, id="ALL-FUNC0011"):
    """Places one callable reads or rewrites its own scope reflectively."""

    reflections: tuple[str, ...] = ("locals", "vars", "globals", "eval", "exec")

    query = """
        SELECT c.path, c.start_line, c.start_column, c.callee AS name, d.qualname
        FROM core_call c JOIN core_def d ON d.file_id = c.file_id AND d.def_id = c.function_id
        WHERE c.parses AND list_contains($reflections::VARCHAR[], c.callee)
    """
    message = "`{name}` reflectively exposes the scope of `{qualname}`"
