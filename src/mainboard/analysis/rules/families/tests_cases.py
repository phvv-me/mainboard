"""mcmr's TestCaseGroupFact, derived from the test bodies TestFunctionFact reads.

The family reads every test, collected or not, as a vector of the literals its body states, grouped
by body shape within its file, and the loops that walk literal cases. Its relations read the
helper relations of TestFunctionFact (`test_function_shapes`, `test_function_declared`), so a run
derives both families.
"""

from ...catalog import Family

_FILE = "'testcasegroupfact:' || v.path"
_GROUP = f"{_FILE} || '/groups:' || v.grp"
_VECTOR = f"{_GROUP} || '/groups.literal_vectors:container:' || v.vector"


class TestCaseGroup(Family):
    family = "TestCaseGroupFact"
    verified = True
    relations = {
        "facts": """
            SELECT fact_order, 'testcasegroupfact:' || path AS fact_id, path, start_line,
                   start_column, end_line, end_column, language
            FROM test_function_facts
        """,
        "vectors": """
            SELECT s.path, s.body_shape, coalesce(s.literals, []) AS literals,
                   dense_rank() OVER (PARTITION BY s.path ORDER BY s.body_shape) - 1 AS grp,
                   row_number() OVER (PARTITION BY s.path, s.body_shape ORDER BY s.test) - 1 AS
                     vector
            FROM test_function_shapes s
        """,
        "loops": """
            WITH found AS (
              SELECT t.path, count(c.node_id) AS case_count,
                     EXISTS (SELECT 1 FROM test_function_nodes a
                             WHERE a.fact_id = f.fact_id AND a.type = 'assert_statement'
                               AND a.node_id > b.node_id
                               AND a.node_id <= b.node_id + b.descendant_count) AS owns_assertion,
                     row_number() OVER (PARTITION BY t.path ORDER BY f.node_id) - 1 AS ordinal
              FROM test_function_declared t
              JOIN test_function_nodes f ON f.fact_id = t.fact_id AND f.type = 'for_statement'
                AND f.node_id > t.test AND f.node_id <= t.last
              JOIN test_function_nodes k ON k.fact_id = f.fact_id AND k.parent_id = f.node_id
                AND k.type = 'in'
              JOIN test_function_nodes it ON it.fact_id = f.fact_id AND it.parent_id = f.node_id
                AND it.sibling_index = k.sibling_index + 1
                AND it.type IN ('list', 'tuple', 'expression_list')
              JOIN test_function_nodes b ON b.fact_id = f.fact_id AND b.parent_id = f.node_id
                AND b.type = 'block'
              LEFT JOIN test_function_nodes c ON c.fact_id = f.fact_id AND c.parent_id = it.node_id
                AND c.type NOT IN ('(', ')', '[', ']', ',')
              GROUP BY t.path, f.node_id, f.fact_id, b.node_id, b.descendant_count)
            SELECT 'testcasegroupfact:' || path AS fact_id, 'testcasegroupfact:' || path AS
              parent_id,
                   'testcasegroupfact:' || path || '/loops:' || ordinal AS record_id, ordinal,
                   case_count, owns_assertion
            FROM found
        """,
        "groups": f"""
            SELECT DISTINCT {_FILE} AS fact_id, {_FILE} AS parent_id, {_GROUP} AS record_id,
                   v.grp AS ordinal, v.body_shape AS normalized_syntax
            FROM test_case_group_vectors v
        """,
        "values": f"""
            SELECT {_FILE} AS fact_id, 'groups.literal_vectors' AS relation,
                   {_GROUP} AS parent_id, {_VECTOR} AS container_id, v.vector AS container_ordinal,
                   len(v.literals) AS container_length, 'container' AS entry_kind, 0 AS ordinal,
                   CAST(NULL AS VARCHAR) AS string_value
            FROM test_case_group_vectors v
            UNION ALL
            SELECT {_FILE}, 'groups.literal_vectors', {_VECTOR}, {_VECTOR}, v.vector,
                   len(v.literals), 'value', generate_subscripts(v.literals, 1) - 1,
                   unnest(v.literals)
            FROM test_case_group_vectors v
        """,
    }
