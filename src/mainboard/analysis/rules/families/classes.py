"""mcmr's ClassFact, derived from sitting_duck's syntax trees.

mcmr read each Python class through ruff's AST and joined the repository (who inherits it, imports
it, builds it) in a second pass. Here a class is a `class_definition` node; its bases, decorators
and methods are the nodes below it, and the repository judgments are joins over the base edges
(`classes_graph`). The relations build on one another in `classes_syntax`, `classes_behavior`,
`classes_graph`, `classes_judgments` and `classes_groups`, and the ones rules read come last. The
first relation, `files`, is the identity (one row per Python file); `facts` needs a judgment over
every class, so it follows them.

Differences from the kernel, kept on purpose:

- a class under a module-level `if` or `try` is a class too (the kernel skipped it);
- `is_protocol` is a base `Protocol` imported from typing, not any base imported from typing;
- a base imported under an alias resolves through the alias, and a class called through a package
  re-export counts as built.
"""

from ...catalog import Family, Function
from . import classes_behavior, classes_graph, classes_groups, classes_judgments, classes_syntax


class ClassesIsTest(Function):
    """Whether a path is a test file: a test directory, or a file named like one."""

    path: str

    expression = r"""(
      list_contains(['test.py', 'tests.py', 'tests.rs', 'conftest.py'],
                    regexp_extract(path, '[^/]*$'))
      OR starts_with(regexp_extract(path, '[^/]*$'), 'test_')
      OR contains(regexp_extract(path, '[^/]*$'), '_test.')
      OR contains(regexp_extract(path, '[^/]*$'), '.test.')
      OR contains(regexp_extract(path, '[^/]*$'), '.spec.')
      OR list_has_any(list_slice(string_split(path, '/'), 1, len(string_split(path, '/')) - 1),
                      ['test', 'tests', '__tests__']))"""


class ClassesVisibility(Function):
    """`public`, `internal` (one underscore) or `private` (two); dunder names are public."""

    name: str

    expression = """CASE WHEN starts_with(name, '__') AND ends_with(name, '__') THEN 'public'
      WHEN starts_with(name, '__') THEN 'private' WHEN starts_with(name, '_') THEN 'internal'
      ELSE 'public' END"""


FACTS = """
    WITH policy AS (
      SELECT EXISTS (SELECT 1 FROM import_binding_facts
                     WHERE language = 'python' AND starts_with(ltrim(declaration_text), 'from')
                       AND (module = 'patos' OR starts_with(module, 'patos.')
                            OR module = 'common.bases' OR ends_with(module, '.common.bases')))
             OR EXISTS (SELECT 1 FROM class_rows
                        WHERE (foundation_base OR is_dataclass) AND field_count = 0
                          AND NOT nested AND subclass_count > 0) AS stated)
    SELECT row_number() OVER (ORDER BY path) - 1 AS fact_order, fact_id, path, start_line,
           start_column, end_line, end_column, language,
           (SELECT stated FROM policy) AS has_approved_model_foundation_policy
    FROM class_files
"""

CLASSES = """
    SELECT r.class_id, r.fact_id, r.ordinal, r.name, r.path, c.start_line, c.start_column,
           c.end_line, c.end_column, classes_is_test(r.path) AS is_test,
           CASE WHEN r.nested THEN 'nested' ELSE 'module' END AS scope,
           classes_visibility(r.name) AS visibility, r.has_explicit_registry_name,
           r.has_instance_fields, r.field_count, r.has_inherited_fields, r.descendant_count,
           r.duplicate_component_alias_count, r.is_protocol, r.built AS is_instantiated,
           r.is_exported, r.only_cross_module_reference_is_subclass, r.is_pass_through_layer,
           r.base_is_removable_overlap, r.has_redundant_direct_base,
           r.has_noncooperative_concrete_collision, r.is_declarative_model, r.is_dataclass,
           r.has_ordinary_behavior, r.states_model_configuration,
           r.directly_inherits_pydantic_base_model, r.inherits_approved_model_foundation
    FROM class_rows r JOIN class_defs c ON c.class_id = r.class_id
"""

DIRECT_BASES = "SELECT class_id, ordinal, text AS value FROM class_bases"

DIRECT_SUBCLASSES = """
    SELECT e.parent AS class_id,
           row_number() OVER (PARTITION BY e.parent ORDER BY k.module, k.name) - 1 AS ordinal,
           k.name AS value
    FROM class_edges e JOIN class_keys k ON k.class_id = e.child
"""

CLASS_DECORATORS = "SELECT class_id, ordinal, text AS value FROM class_def_decorations"

CLASS_KEYWORDS = "SELECT class_id, ordinal, text AS value FROM class_header_items WHERE is_keyword"

METHODS = """
    SELECT m.method_id, m.fact_id, m.class_id, m.ordinal, m.name, m.path, m.start_line,
           m.start_column, m.end_line, m.end_column,
           classes_text(f.lines, m.start_line, m.start_column, m.end_line, m.end_column) AS source,
           (SELECT count(*) FROM class_regions r WHERE r.path = m.path AND r.line < m.start_line)
             AS region, m.kind, m.visibility, m.is_protocol_name,
           s.method_node IS NOT NULL AS reads_receiver_state
    FROM class_method_defs m
    JOIN class_files f ON f.fact_id = m.fact_id
    LEFT JOIN class_receiver_state s ON s.fact_id = m.fact_id AND s.method_node = m.method_node
"""

METHOD_DECORATORS = """
    SELECT m.method_id, d.ordinal, d.text AS value
    FROM class_method_defs m
    JOIN class_decorations d ON d.fact_id = m.fact_id AND d.decorated_node = m.statement_node
"""

OWNER_QUALIFIED_CALLS = "SELECT method_id, ordinal, value FROM class_owner_calls"

COUPLED_GROUPS = """
    SELECT fact_id || ':group:' || ordinal AS group_id, fact_id, ordinal, prefix, path,
           start_line, start_column, end_line, end_column, type_count, maximum_type_lines,
           coimporting_module_count
    FROM (SELECT 'classes:' || path AS fact_id, * FROM class_coupled_groups_rows)
"""

COUPLED_GROUP_SUFFIXES = """
    SELECT group_id, idx - 1 AS ordinal, substr(name, length(prefix) + 1) AS suffix
    FROM (SELECT 'classes:' || path || ':group:' || ordinal AS group_id, prefix,
                 generate_subscripts(names, 1) AS idx, unnest(names) AS name
          FROM class_coupled_groups_rows)
"""

MODEL_FILES = """
    SELECT fact_id || ':model_file' AS model_file_id, fact_id, 0 AS ordinal, path,
           1 AS start_line, 0 AS start_column, end_line, end_column, top_level_class_count,
           model_class_count, is_package_initializer
    FROM class_model_file_rows
"""

PROJECTIONS = """
    SELECT g.projection_id, g.fact_id, g.ordinal, g.root, g.path, c.start_line, c.start_column,
           c.end_line, c.end_column
    FROM class_projection_groups g
    JOIN class_nodes c ON c.fact_id = g.fact_id AND c.node_id = g.container
"""

PROJECTION_ATTRIBUTES = """
    SELECT projection_id, idx - 1 AS ordinal, value
    FROM (SELECT projection_id, generate_subscripts(attributes, 1) AS idx,
                 unnest(attributes) AS value FROM class_projection_groups)
"""

PROJECTION_OUTPUT_KEYS = """
    SELECT projection_id, idx - 1 AS ordinal, value
    FROM (SELECT g.projection_id, generate_subscripts(k.keys, 1) AS idx, unnest(k.keys) AS value
          FROM class_projection_groups g
          JOIN class_projection_keys k ON k.fact_id = g.fact_id AND k.container = g.container)
"""


class Classes(Family):
    family = "ClassFact"
    verified = True
    relations = {
        **classes_syntax.RELATIONS,
        **classes_behavior.RELATIONS,
        **classes_graph.RELATIONS,
        **classes_judgments.RELATIONS,
        **classes_groups.RELATIONS,
        "facts": FACTS,
        "classes": CLASSES,
        "direct_bases": DIRECT_BASES,
        "direct_subclasses": DIRECT_SUBCLASSES,
        "class_decorators": CLASS_DECORATORS,
        "class_keywords": CLASS_KEYWORDS,
        "methods": METHODS,
        "method_decorators": METHOD_DECORATORS,
        "owner_qualified_calls": OWNER_QUALIFIED_CALLS,
        "coupled_groups": COUPLED_GROUPS,
        "coupled_group_suffixes": COUPLED_GROUP_SUFFIXES,
        "model_files": MODEL_FILES,
        "projections": PROJECTIONS,
        "projection_attributes": PROJECTION_ATTRIBUTES,
        "projection_output_keys": PROJECTION_OUTPUT_KEYS,
    }
