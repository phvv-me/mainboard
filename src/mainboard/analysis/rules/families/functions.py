"""mcmr's FunctionFact, derived from sitting_duck's syntax trees (Python only).

mcmr read each function through ruff's AST in one Rust pass. Here a function is a
`function_definition` node (its statement span includes the decorators of a `decorated_definition`
parent, and ends at its last token that is not a comment). The relations below build it up:

- `defs` is the identity, `nodes` a narrow copy of the Python syntax nodes, `lines` the source
  lines with a running count of the code ones, `imports` and `modules` what the files import;
- `parameters`, `decorators`, `controls`, `references`, `tensor_roles` are mcmr's relations of the
  same names, `bodies` holds the body block, its statements (minus a docstring) and the docstring;
- `inside` pairs each function with the nodes in its body (nested functions included, as mcmr
  walks them) that the measures read; `measures` and `tasks` aggregate them;
- `devices`, `decorations`, `classes`, `docstrings` and `forwarding` settle one group of the
  `functions` columns each.

Differences from the kernel, kept on purpose:

- a function under a module-level `if` or `try` is a function too (the kernel skipped it);
- `is_polymorphic` is the `override` decorator only (the kernel also marks any function whose name
  a class of its file shares with an ancestor or a descendant, which matters to ALL-FUNC0006 only
  for a module function named like a dispatched method);
- behavior, calls, references, recursion and receiver reads count in the places the kernel never
  walks: an `elif` test, an `assert` message, an assignment target, a call under `yield from`,
  a slice (so `behavior_operation_count` is higher and `is_first_class_reference` lower);
- a docstring has its backslash, quote, line-break, tab and carriage-return escapes resolved, and
  `\\xNN`, `\\u` and `\\N{...}` left as written.
"""

from ...catalog import Family
from . import functions_body, functions_defs, functions_sites
from .calls_sql import PythonBase


def _hop(number: int) -> str:
    """One re-export step of a decorator's target: the longest project module the target starts
    with, then what that module's imports say its next name stands for (the target stays when it
    names no project module or the module imports no such name)."""
    previous = "applied" if number == 1 else f"hop{number - 1}"
    return f"""held{number} AS (
      SELECT p.function_id, p.ordinal, p.decorator, p.target, max(length(m.module)) AS length,
             arg_max(m.ast_fact, length(m.module)) AS ast
      FROM {previous} p JOIN function_modules m ON starts_with(p.target, m.module || '.')
      GROUP BY p.function_id, p.ordinal, p.decorator, p.target
    ), hop{number} AS (
      SELECT p.function_id, p.ordinal, p.decorator,
             coalesce(b.target || substr(substr(p.target, h.length + 2),
                        length(split_part(substr(p.target, h.length + 2), '.', 1)) + 1),
                      p.target) AS target
      FROM {previous} p
      LEFT JOIN held{number} h USING (function_id, ordinal, decorator, target)
      LEFT JOIN bindings b ON b.ast = h.ast
        AND b.bound = split_part(substr(p.target, h.length + 2), '.', 1)
    )"""


DEVICES = r"""
    WITH bindings AS (
      SELECT ast, bound, any_value(target) AS target FROM function_imports GROUP BY ast, bound
    ), applied AS (
      SELECT x.function_id, x.ordinal, x.decorator,
             coalesce(b.target || substr(r.dotted, length(split_part(r.dotted, '.', 1)) + 1),
                      r.dotted) AS target
      FROM function_decorators x
      JOIN function_defs d ON d.entity_id = x.function_id,
      LATERAL (SELECT trim(split_part(x.decorator, '(', 1)) AS dotted) r
      LEFT JOIN bindings b ON b.ast = d.ast AND b.bound = split_part(r.dotted, '.', 1)
    ), @hops@
    SELECT function_id, arg_min(role, ordinal) AS device_role
    FROM (SELECT *, functions_device_role(target, decorator) AS role FROM hop3)
    WHERE role <> '' GROUP BY function_id
""".replace("@hops@", ", ".join(_hop(number) for number in (1, 2, 3)))

DECORATIONS = r"""
    WITH named AS (
      SELECT function_id, functions_decorator_name(decorator) AS name FROM function_decorators
    )
    SELECT function_id,
           bool_or(name = 'cached_property') AS cached_property, bool_or(name = 'cache') AS cache,
           bool_or(name = 'lru_cache') AS lru_cache,
           bool_or(functions_property_decorator(name)) AS is_property,
           bool_or(name IN ('abstractmethod', 'abstractproperty')) AS is_abstract,
           bool_or(name = 'overload') AS is_overload,
           bool_or(name = 'override') AS is_polymorphic,
           bool_or(functions_validator_decorator(name)) AS is_pydantic_validator,
           bool_or(NOT functions_binding_decorator(name)) AS has_framework_decorator,
           bool_or(name = 'rule') AS is_rule
    FROM named GROUP BY function_id
"""

CLASSES = r"""
    WITH bases AS (
      SELECT c.ast, c.node AS class_node,
             functions_base_name(coalesce(nullif(a.name, ''), f.name)) AS base
      FROM function_nodes c
      JOIN function_nodes l ON l.ast = c.ast AND l.parent = c.node AND l.type = 'argument_list'
      JOIN function_nodes a ON a.ast = l.ast AND a.parent = l.node
        AND a.type IN ('identifier', 'attribute', 'subscript', 'call')
      LEFT JOIN function_nodes f ON f.ast = a.ast AND f.parent = a.node AND f.sibling = 0
      WHERE c.type = 'class_definition'
    )
    SELECT ast, class_node, bool_or(base = 'Protocol') AS is_protocol_member,
           bool_or(functions_model_foundation(base)) AS is_model_method
    FROM bases GROUP BY ast, class_node
"""

DOCSTRINGS = r"""
    SELECT ast, def, functions_docstring(CASE WHEN is_raw THEN content
                                              ELSE functions_unescaped(content) END) AS docstring
    FROM (SELECT ast, def, is_raw,
                 CASE WHEN regexp_matches(bare, '^(["]{3}|[\x27]{3})')
                      THEN substr(bare, 4, length(bare) - 6)
                      ELSE substr(bare, 2, length(bare) - 2) END AS content
          FROM (SELECT ast, def, regexp_matches(literal, '^[uU]?[rR]') AS is_raw,
                       regexp_replace(literal, '^[uUrR]*', '') AS bare
                FROM function_bodies WHERE literal IS NOT NULL))
"""

FORWARDING = r"""
    WITH required AS (
      SELECT d.ast, d.node AS def, count(*) AS parameters, any_value(p.name) AS only_name
      FROM function_defs d
      JOIN function_parameters p ON p.function_id = d.entity_id AND NOT p.is_keyword_only
      LEFT JOIN function_receivers rc ON rc.ast = d.ast AND rc.def = d.node
      WHERE p.name IS DISTINCT FROM rc.receiver
      GROUP BY d.ast, d.node
    )
    SELECT b.ast, b.def, true AS returns_single_call,
           coalesce(al.kids = 3 AND a1.type = 'identifier' AND r.parameters = 1
                    AND a1.name = r.only_name, false) AS forwards_only_parameter_unchanged
    FROM function_bodies b
    JOIN function_nodes pv ON pv.ast = b.ast AND pv.parent = b.first_node AND pv.sibling = 1
    LEFT JOIN function_nodes un ON un.ast = pv.ast AND un.parent = pv.node AND un.sibling = 1
      AND pv.type = 'parenthesized_expression'
    JOIN function_nodes rv ON rv.ast = pv.ast
      AND rv.node = CASE WHEN pv.type = 'parenthesized_expression' THEN un.node ELSE pv.node END
      AND rv.type = 'call'
    LEFT JOIN function_nodes al ON al.ast = rv.ast AND al.parent = rv.node AND al.sibling = 1
    LEFT JOIN function_nodes a1 ON a1.ast = al.ast AND a1.parent = al.node AND a1.sibling = 1
    LEFT JOIN required r ON r.ast = b.ast AND r.def = b.def
    WHERE b.statements = 1 AND b.first_type = 'return_statement'
"""

FUNCTIONS = r"""
    WITH sites AS (
      SELECT function_id, count(*) AS sites, any_value(owner_class) AS owner_class,
             any_value(owner_start_line) AS owner_start_line,
             any_value(owner_start_column) AS owner_start_column,
             any_value(owner_end_line) AS owner_end_line,
             any_value(owner_end_column) AS owner_end_column
      FROM function_references GROUP BY function_id
    ), shaped AS (
      SELECT function_id, count(*) FILTER (WHERE kind = 'conditional') AS conditional_count,
             count(*) AS controls
      FROM function_controls GROUP BY function_id
    ), planned AS (
      SELECT DISTINCT b.ast, b.def
      FROM function_bodies b
      JOIN function_nodes q ON q.ast = b.ast AND q.node = b.returns
      JOIN function_nodes r ON r.ast = b.ast AND r.node BETWEEN q.node AND q.node + q.size
        AND r.type = 'identifier'
      WHERE r.name IN ('Expr', 'LazyFrame') OR ends_with(r.name, 'Query')
    ), tensor AS (
      SELECT ast, def, bool_or(wrapper IS NOT NULL) AS wrapped,
             bool_or(wrapper IS NOT NULL AND wrapper <> 'Shaped') AS states_dtype
      FROM function_tensors GROUP BY ast, def
    )
    SELECT d.fact_id, d.path, d.language, d.start_line, d.start_column, d.end_line, d.end_column,
           d.entity_id, d.name, d.scope, d.visibility,
           CASE WHEN w.cached_property THEN 'cached_property' WHEN w.cache THEN 'cache'
                WHEN w.lru_cache THEN 'lru_cache' ELSE '' END AS cache_decorator,
           coalesce(v.device_role, '') AS device_role, coalesce(doc.docstring, '') AS docstring,
           coalesce(CASE WHEN s.sites = 1 THEN s.owner_class END, '')
             AS sole_reference_owner_class,
           CASE WHEN s.sites = 1 THEN s.owner_start_line END
             AS sole_reference_owner_definition_start_line,
           CASE WHEN s.sites = 1 THEN s.owner_start_column END
             AS sole_reference_owner_definition_start_column,
           CASE WHEN s.sites = 1 THEN s.owner_end_line END
             AS sole_reference_owner_definition_end_line,
           CASE WHEN s.sites = 1 THEN s.owner_end_column END
             AS sole_reference_owner_definition_end_column,
           d.path AS definition_path, d.start_line AS definition_start_line,
           d.start_column AS definition_start_column, d.end_line AS definition_end_line,
           d.end_column AS definition_end_column, t.text AS definition_text,
           coalesce(k.created_task_count, 0) AS created_task_count,
           CASE WHEN coalesce(b.statements, 0) = 0 THEN 0
                ELSE ct.code_through - coalesce(cp.code_through, 0) END AS implementation_lines,
           coalesce(b.statements, 0) AS direct_statement_count,
           coalesce(l.loads, 0) AS reference_count, m.behavior_operation_count,
           coalesce(sh.conditional_count, 0) AS conditional_count,
           coalesce(k.gather_consumes_created_tasks, false) AS gather_consumes_created_tasks,
           coalesce(k.gather_returns_exceptions, false) AS gather_returns_exceptions,
           coalesce(k.has_task_group, false) AS has_task_group, m.reads_receiver,
           tn.ast IS NOT NULL AND (tn.wrapped
                                   OR contains(lower(coalesce(doc.docstring, '')), 'shape')
                                   OR contains(lower(coalesce(doc.docstring, '')), 'dimension'))
             AS has_tensor_shape_semantics,
           tn.ast IS NOT NULL AND (tn.states_dtype
                                   OR functions_mentions_dtype(lower(coalesce(doc.docstring, ''))))
             AS has_tensor_dtype_semantics,
           starts_with(d.name, '__') AND ends_with(d.name, '__') AS is_protocol_name,
           coalesce(b.is_async, false) AS is_async, m.is_recursive,
           coalesce(l.loads, 0) > coalesce(s.sites, 0) AS is_first_class_reference,
           coalesce(w.is_abstract, false) AS is_abstract,
           coalesce(o.is_protocol_member, false) AS is_protocol_member,
           coalesce(w.is_overload, false) AS is_overload,
           coalesce(w.is_property, false) AS is_property,
           coalesce(w.has_framework_decorator, false) OR functions_lifecycle_name(d.name)
             OR starts_with(d.name, 'visit_') OR d.name = 'generic_visit' AS is_framework_hook,
           coalesce(w.is_rule, false)
             OR (coalesce(sh.controls, 0) = 0 AND pl.def IS NOT NULL) AS is_declarative_body,
           coalesce(w.is_polymorphic, false) AS is_polymorphic,
           b.statements = 1 AND b.first_type = 'pass_statement' AS is_pass_body,
           b.statements = 1 AND b.first_type = 'raise_statement' AS is_raise_body,
           coalesce(f.returns_single_call, false) AS returns_single_call,
           coalesce(f.forwards_only_parameter_unchanged, false)
             AS forwards_only_parameter_unchanged,
           coalesce(o.is_model_method, false) AS is_model_method,
           coalesce(w.is_pydantic_validator, false) AS is_pydantic_validator,
           m.checks_raw_input_type, m.raises_validation_exception, m.constructs_owner_model
    FROM function_defs d
    JOIN function_bodies b ON b.ast = d.ast AND b.def = d.node
    JOIN function_measures m ON m.ast = d.ast AND m.def = d.node
    LEFT JOIN function_texts t ON t.ast = d.ast AND t.node = coalesce(d.wrapper, d.node)
    LEFT JOIN function_docstrings doc ON doc.ast = d.ast AND doc.def = d.node
    LEFT JOIN function_decorations w ON w.function_id = d.entity_id
    LEFT JOIN function_devices v ON v.function_id = d.entity_id
    LEFT JOIN function_classes o ON o.ast = d.ast AND o.class_node = d.owner
    LEFT JOIN sites s ON s.function_id = d.entity_id
    LEFT JOIN shaped sh ON sh.function_id = d.entity_id
    LEFT JOIN planned pl ON pl.ast = d.ast AND pl.def = d.node
    LEFT JOIN function_forwarding f ON f.ast = d.ast AND f.def = d.node
    LEFT JOIN tensor tn ON tn.ast = d.ast AND tn.def = d.node
    LEFT JOIN function_tasks k ON k.ast = d.ast AND k.def = d.node
    LEFT JOIN function_loads l ON l.ast = d.ast AND l.name = d.name
    LEFT JOIN function_lines ct ON ct.path = d.path AND ct.line = b.last_line
    LEFT JOIN function_lines cp ON cp.path = d.path AND cp.line = b.first_line - 1
"""

TENSOR_ROLES = r"""
    SELECT d.entity_id AS function_id,
           row_number() OVER (PARTITION BY d.entity_id ORDER BY t.role) - 1 AS ordinal, t.role
    FROM function_tensors t JOIN function_defs d ON d.ast = t.ast AND d.node = t.def
"""


class Functions(Family):
    family = "FunctionFact"
    verified = True
    relations = {
        "defs": functions_defs.DEFS,
        "nodes": functions_defs.NODES,
        "files": functions_defs.FILES,
        "lines": functions_defs.LINES,
        "modules": PythonBase("FunctionFact")._files(),
        "imports": functions_defs.IMPORTS,
        "calls": functions_defs.CALLS,
        "bodies": functions_defs.BODIES,
        "owners": functions_body.OWNERS,
        "texts": functions_sites.TEXTS,
        "decorators": functions_sites.DECORATORS,
        "parameters": functions_sites.PARAMETERS,
        "controls": functions_sites.CONTROLS,
        "loads": functions_sites.LOADS,
        "references": functions_sites.REFERENCES,
        "chain": functions_body.CHAIN,
        "receivers": functions_body.RECEIVERS,
        "taken": functions_body.TAKEN,
        "inside": functions_body.INSIDE,
        "roots": functions_body.ROOTS,
        "measures": functions_body.MEASURES,
        "tasks": functions_body.TASKS,
        "tensors": functions_body.TENSORS,
        "tensor_roles": TENSOR_ROLES,
        "devices": DEVICES,
        "decorations": DECORATIONS,
        "classes": CLASSES,
        "docstrings": DOCSTRINGS,
        "forwarding": FORWARDING,
        "functions": FUNCTIONS,
    }
