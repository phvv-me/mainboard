"""How a level flows from the atoms of `levels.py` to every key that carries one.

Levels flow, to a fixpoint, along stores (each store is its own key, and a use reads the stores
that can reach it), function returns, arguments to parameters and, for a store made under a
branch, the branch's condition. Keys name what carries a level: `s:<fact>:<node>` the value one
store binds, `r:<fact>:<function node>` what a function returns (`#<position>` one element of a
tuple or record, `.<field>` one field of a record), `b:<fact>:<node>` a branch's condition. A
field read off a record a call just built takes the level of that field alone
(`cuda_field_reads`).
"""

from ..base import CudaRelation


class CudaReturnElements(CudaRelation):
    """The values a function returns one by one when it returns a tuple (`return a, b`) or builds
    a record from arguments (`return Pair(a, b)`): `key` is `r:<fact>:<function node>#<position>`."""

    stored = True
    query = """
        WITH returns AS (
          SELECT r.fact_id, r.node_id, r.scope."function" AS function_id, e.node_id AS value_id,
                 e.type
          FROM cuda_nodes() r
          JOIN cuda_nodes() e ON e.fact_id = r.fact_id AND e.parent_id = r.node_id
            AND e.sibling_index = 1
          WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
            AND EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = r.fact_id AND d.function_id = r.scope."function")
        )
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.function_id || '#'
                          || (row_number() OVER (PARTITION BY r.fact_id, r.node_id
                                                 ORDER BY x.sibling_index) - 1) AS key,
               x.node_id AS first, x.node_id + x.descendant_count AS last
        FROM returns r
        JOIN cuda_nodes() x ON x.fact_id = r.fact_id AND x.parent_id = r.value_id
        WHERE r.type IN ('tuple', 'expression_list') AND x.type NOT IN ('(', ')', ',', 'comment')
        UNION ALL
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.function_id || '#' || a.position,
               x.node_id, x.node_id + x.descendant_count
        FROM returns r
        JOIN cuda_arguments() a ON a.fact_id = r.fact_id AND a.call_id = r.value_id
          AND a.position IS NOT NULL
        JOIN cuda_nodes() x ON x.fact_id = a.fact_id AND x.node_id = a.value_id
        WHERE r.type = 'call'
        UNION ALL
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.function_id || '.' || coalesce(f.field, a.keyword),
               x.node_id, x.node_id + x.descendant_count
        FROM returns r
        JOIN cuda_calls() c ON c.fact_id = r.fact_id AND c.node_id = r.value_id
        LEFT JOIN cuda_call_targets() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
          AND t.kind = 'class'
        JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
        JOIN cuda_nodes() x ON x.fact_id = a.fact_id AND x.node_id = a.value_id
        LEFT JOIN cuda_record_fields() f ON f.record = coalesce(t.key, c.qualified_name)
          AND f.position = a.position
        WHERE r.type = 'call' AND coalesce(f.field, a.keyword) IS NOT NULL
    """


class CudaRecordFields(CudaRelation):
    """The fields of the records a call can build, by `record` (the key of a class the project
    defines, else the dotted name of one a library does) and the `position` a positional argument
    fills: the annotated fields of the class in the order its body writes them."""

    stored = True
    query = """
        SELECT o.class_key AS record, f.name AS field, o.position
        FROM cuda_field_order() o JOIN cuda_fields() f ON f.field = o.field
        UNION
        SELECT record, field, position FROM cuda_library_results()
    """


class CudaFieldReads(CudaRelation):
    """The attributes read off a record that a call has just built, as the atoms they are:
    `warps.total` after `warps = self._prefixed(x)` takes the level of what `_prefixed` returns as
    `total` (`ref`), and `emitted.before` after `emitted = warp.ballot_prefix(f)` the library's
    level of that field (`level`), not the level of the whole record, which is the highest of its
    fields. A read is listed only when every store its name can see is such a call, so
    it hides the nodes of its object (`mask_first`..`mask_last`) and everything else reads the
    whole record."""

    stored = True
    query = """
        WITH reads AS (
          SELECT x.fact_id, x.node_id, x.name AS label, o.node_id AS object_id,
                 o.node_id + o.descendant_count AS object_last, o.type AS object_type, f.name AS field
          FROM cuda_nodes() x
          JOIN cuda_nodes() o ON o.fact_id = x.fact_id AND o.parent_id = x.node_id
            AND o.sibling_index = 0
          JOIN cuda_nodes() f ON f.fact_id = x.fact_id AND f.parent_id = x.node_id
            AND f.sibling_index = 1
          WHERE x.type = 'attribute' AND o.type IN ('identifier', 'call')
            AND EXISTS (SELECT 1 FROM cuda_device_code() d
                        WHERE d.fact_id = x.fact_id AND d.function_id = x.scope."function")
        ), sources AS (
          SELECT r.*, st.source_id AS call_id, st.kind = 'assignment' AS assigned
          FROM reads r
          JOIN cuda_reaching_stores() g ON g.fact_id = r.fact_id AND g.use_id = r.object_id
          JOIN cuda_level_stores() ls ON ls.key = g.key
          JOIN cuda_stores() st ON st.fact_id = ls.fact_id AND st.node_id = ls.node_id
          WHERE r.object_type = 'identifier'
          UNION ALL
          SELECT r.*, r.object_id, true FROM reads r WHERE r.object_type = 'call'
        ), told AS (
          SELECT s.*, l.level,
                 'r:' || t.definition_fact || ':' || t.definition_id || '.' || s.field AS ref
          FROM sources s
          LEFT JOIN cuda_calls() c ON c.fact_id = s.fact_id AND c.node_id = s.call_id AND s.assigned
          LEFT JOIN cuda_library_results() l ON l.function = c.qualified_name AND l.field = s.field
          LEFT JOIN cuda_resolved_calls() t ON t.fact_id = s.fact_id AND t.node_id = s.call_id
            AND s.assigned AND t.kind = 'function' AND t.via <> 'override'
        ), told_known AS (
          SELECT t.*, t.level IS NOT NULL
                      OR EXISTS (SELECT 1 FROM cuda_return_elements() e WHERE e.key = t.ref) AS known
          FROM told t
        ), every AS (
          SELECT *, bool_and(known) OVER (PARTITION BY fact_id, node_id) AS all_known
          FROM told_known
        )
        SELECT DISTINCT fact_id, node_id, coalesce(level, 0) AS level,
               CASE WHEN level IS NULL THEN ref END AS ref, label, object_id AS mask_first,
               object_last AS mask_last
        FROM every
        WHERE all_known
    """


class CudaLevelLinks(CudaRelation):
    """A name bound by unpacking a call (`before, total = self._prefixed(x)`) takes the level of
    the returned element at its position (`ref`), not of the whole tuple, where the callee
    returns its elements one by one."""

    stored = True
    query = """
        SELECT s.key, 'r:' || t.definition_fact || ':' || t.definition_id || '#' || o.position AS ref,
               s.name AS label
        FROM cuda_level_stores() s
        JOIN cuda_stores() o ON o.fact_id = s.fact_id AND o.node_id = s.node_id
        JOIN cuda_calls() c ON c.fact_id = o.fact_id AND c.node_id = o.source_id
        JOIN cuda_resolved_calls() t ON t.fact_id = c.fact_id AND t.node_id = c.node_id
          AND t.kind = 'function' AND t.via <> 'override'
        WHERE s.kind = 'unpack' AND o.position IS NOT NULL
          AND EXISTS (SELECT 1 FROM cuda_return_elements() e
                      WHERE e.key = 'r:' || t.definition_fact || ':' || t.definition_id
                                    || '#' || o.position)
    """


class CudaLevelSinks(CudaRelation):
    """Where a level lands: `key` takes the level of the atoms in `first`..`last` (nodes of
    `fact_id`). A store takes its value and, under a branch, the branch's condition; a function
    its returned values (and each element it returns one by one); an argument its expression
    (the parameter takes the lowest level of its arguments); a branch its condition."""

    stored = True
    query = """
        SELECT s.fact_id, s.key, x.node_id AS first, x.node_id + x.descendant_count AS last
        FROM cuda_level_stores() s
        JOIN cuda_stores() o ON o.fact_id = s.fact_id AND o.node_id = s.node_id
        JOIN cuda_nodes() x ON x.fact_id = o.fact_id AND x.node_id = o.source_id
        WHERE s.kind IN ('assignment', 'loop', 'unpack')
          AND NOT EXISTS (SELECT 1 FROM cuda_level_links() l WHERE l.key = s.key)
        UNION ALL
        SELECT fact_id, key, first, last FROM cuda_return_elements()
        UNION ALL
        SELECT s.fact_id, s.key, r.node_id, r.node_id + r.descendant_count
        FROM cuda_level_stores() s
        JOIN cuda_nodes() a ON a.fact_id = s.fact_id AND a.node_id = (
          SELECT q.parent_id FROM cuda_nodes() q WHERE q.fact_id = s.fact_id AND q.node_id = s.node_id)
        JOIN cuda_nodes() r ON r.fact_id = a.fact_id AND r.parent_id = a.node_id
          AND r.sibling_index = 2
        WHERE s.augmented AND a.type = 'augmented_assignment'
        UNION ALL
        SELECT r.fact_id, 'r:' || r.fact_id || ':' || r.scope."function", e.node_id,
               e.node_id + e.descendant_count
        FROM cuda_nodes() r
        JOIN cuda_nodes() e ON e.fact_id = r.fact_id AND e.parent_id = r.node_id
          AND e.sibling_index = 1
        WHERE r.type = 'return_statement' AND r.scope."function" IS NOT NULL
          AND EXISTS (SELECT 1 FROM cuda_device_code() d
                      WHERE d.fact_id = r.fact_id AND d.function_id = r.scope."function")
        UNION ALL
        SELECT fact_id, site, first, last FROM cuda_param_sites()
        UNION ALL
        SELECT s.fact_id, s.key, b.cond_first, b.cond_last
        FROM cuda_level_stores() s
        JOIN cuda_branches() b ON b.fact_id = s.fact_id AND b.scope_id = s.scope_id
          AND s.node_id BETWEEN b.first AND b.last
          AND NOT s.node_id BETWEEN b.cond_first AND b.cond_last
        WHERE NOT s.parameter
        UNION ALL
        SELECT fact_id, 'b:' || fact_id || ':' || branch_id, cond_first, cond_last
        FROM cuda_branches()
    """


class CudaLevelFlows(CudaRelation):
    """Each atom a sink's expression reads that a collective's or call's result does not hide
    (`label`, its `level` or the `ref` it takes, `lowers` as in the atoms)."""

    stored = True
    query = """
        SELECT s.key, a.level, a.ref, a.lowers, a.label
        FROM cuda_level_sinks() s
        JOIN cuda_level_atoms() a ON a.fact_id = s.fact_id AND a.node_id BETWEEN s.first AND s.last
        WHERE NOT EXISTS (
          SELECT 1 FROM cuda_level_atoms() m
          WHERE m.fact_id = a.fact_id AND m.mask_first IS NOT NULL AND m.node_id <> a.node_id
            AND m.node_id BETWEEN s.first AND s.last
            AND a.node_id BETWEEN m.mask_first AND m.mask_last)
        UNION ALL
        SELECT key, 0, ref, false, label FROM cuda_level_links()
    """


class CudaLevels(CudaRelation):
    """The level, 1 or 2, of every key the atoms reach, with `why` the atom that gave it. A
    parameter takes the lowest level among the arguments of all the calls that bind it, so one
    call with a uniform argument keeps it uniform."""

    stored = True
    query = """
        WITH RECURSIVE lv(key, level, why) USING KEY (key) AS (
          SELECT key, max(CASE WHEN lowers THEN least(level, 1) ELSE level END),
                 arg_max(label, (CASE WHEN lowers THEN least(level, 1) ELSE level END, label))
          FROM cuda_level_flows() WHERE ref IS NULL
          GROUP BY key
          HAVING max(CASE WHEN lowers THEN least(level, 1) ELSE level END) > 0
          UNION
          (SELECT f.key,
                  max(CASE WHEN f.lowers THEN least(d.level, 1) ELSE d.level END),
                  arg_max(d.why, (CASE WHEN f.lowers THEN least(d.level, 1) ELSE d.level END, d.why))
           FROM lv d JOIN cuda_level_flows() f ON f.ref = d.key
           GROUP BY f.key
           HAVING max(CASE WHEN f.lowers THEN least(d.level, 1) ELSE d.level END)
                  > coalesce((SELECT x.level FROM recurring.lv x WHERE x.key = f.key), 0)
           UNION ALL
           SELECT p.key, min(x.level), arg_max(x.why, (x.level, x.why))
           FROM (SELECT DISTINCT q.key FROM lv d JOIN cuda_param_sites() q ON q.site = d.key) k
           JOIN cuda_param_sites() p ON p.key = k.key
           JOIN recurring.lv x ON x.key = p.site
           GROUP BY p.key, p.sites
           HAVING count(*) = p.sites
              AND min(x.level) > coalesce((SELECT y.level FROM recurring.lv y WHERE y.key = p.key), 0))
        )
        SELECT * FROM lv
    """
