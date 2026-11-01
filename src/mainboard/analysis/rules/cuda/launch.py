"""Rules over how CUDA kernels are launched and synchronized (mcmr's CU-LAUN family).

`CU-` rules read CUDA facts only (the engine's language scope), so no rule filters by language.
"""

from ...catalog import Example, Relation, Rule

_SCALE = "__global__ void scale(float* values) {\n    values[threadIdx.x] *= 2.0f;\n}\n\n"


class CudaCallNodes(Relation):
    """CUDA call records placed at their call node."""

    query = """
    SELECT * EXCLUDE (path), node_path AS path, node_start_line AS start_line,
           node_start_column AS start_column
    FROM call_calls
"""


class RawBarrierOverCooperativeGroups(Rule, id="CU-LAUN0001", applicability="review"):
    """Raw warp and block synchronization intrinsics where cooperative groups state the same."""

    query = """
            SELECT * FROM cuda_call_nodes()
            WHERE qualified_name IN ('__syncthreads', '__syncwarp', '__ballot_sync', '__shfl_sync',
              '__shfl_down_sync', '__shfl_up_sync', '__shfl_xor_sync', '__any_sync', '__all_sync',
              '__activemask')
        """
    message = "`{qualified_name}` uses a raw CUDA synchronization intrinsic"
    # `__syncthreads()` is the block's `sync()`; with the full mask, a warp intrinsic is the same
    # member of the 32-thread tile (a call giving a width, a partial mask or another argument
    # count stays as written; arguments are read from the call's text, so a call with brackets
    # or nested calls in them is left too). The header is included once per file, above its
    # first line of code, and every edit of a file goes with its first finding so that none
    # lands without it.
    fix = r"""
        WITH calls AS (
          SELECT f.finding_key, f.path, f.start_line, f.start_column, f.node_end_line AS end_line,
                 f.node_end_column AS end_column, f.qualified_name,
                 CASE WHEN trimmed(between_parens) = '' THEN []::VARCHAR[]
                      WHEN NOT regexp_matches(between_parens, '[()<>\[\]{}"'']')
                      THEN list_transform(string_split(between_parens, ','),
                                          lambda a: trimmed(a)) END
                   AS arguments
          FROM (SELECT *, regexp_extract(node_text, '^\w+\s*\((.*)\)$', 1) AS between_parens
                FROM finding) f
        ), targets AS (
          SELECT c.*, CASE
              WHEN c.qualified_name = '__syncthreads' AND len(c.arguments) = 0
              THEN 'cooperative_groups::this_thread_block().sync()'
              WHEN c.qualified_name = '__syncwarp' AND (len(c.arguments) = 0
                   OR (len(c.arguments) = 1 AND lower(c.arguments[1]) = '0xffffffff'))
              THEN 'cooperative_groups::tiled_partition<32>('
                   || 'cooperative_groups::this_thread_block()).sync()'
              WHEN lower(c.arguments[1]) = '0xffffffff' AND len(c.arguments) = m.arity
              THEN 'cooperative_groups::tiled_partition<32>('
                   || 'cooperative_groups::this_thread_block()).' || m.member || '('
                   || array_to_string(c.arguments[2:], ', ') || ')' END AS text
          FROM calls c LEFT JOIN (VALUES
            ('__shfl_sync', 'shfl', 3), ('__shfl_down_sync', 'shfl_down', 3),
            ('__shfl_up_sync', 'shfl_up', 3), ('__shfl_xor_sync', 'shfl_xor', 3),
            ('__ballot_sync', 'ballot', 2), ('__any_sync', 'any', 2), ('__all_sync', 'all', 2)
          ) m(name, member, arity) ON m.name = c.qualified_name
        ), files AS (
          SELECT path, min(finding_key) AS finding_key FROM targets WHERE text IS NOT NULL
          GROUP BY path
        )
        SELECT f.finding_key, t.path, t.start_line, t.start_column, t.end_line, t.end_column,
               t.text
        FROM targets t JOIN files f USING (path) WHERE t.text IS NOT NULL
        UNION ALL BY NAME
        SELECT f.finding_key, f.path, c.line AS start_line, 0 AS start_column,
               c.line AS end_line, 0 AS end_column,
               '#include <cooperative_groups.h>' || chr(10) AS text
        FROM files f
        JOIN (SELECT path, min(line) AS line FROM source_lines()
              WHERE NOT regexp_matches(text, '^\s*(//.*|/\*.*|\*.*)?$') GROUP BY path) c
          USING (path)
        WHERE NOT EXISTS (SELECT 1 FROM source_lines() i
                          WHERE i.path = f.path AND contains(i.text, 'cooperative_groups.h'))
    """
    examples = (
        Example(
            bad=(
                """__global__ void scale(float* values) {
    values[threadIdx.x] *= 2.0f;
    __syncthreads();
}
"""
            ),
            good=(
                """#include <cooperative_groups.h>
__global__ void scale(float* values) {
    values[threadIdx.x] *= 2.0f;
    cooperative_groups::this_thread_block().sync();
}
"""
            ),
        ),
    )


class DefaultStreamKernelLaunch(Rule, id="CU-LAUN0002", applicability="review"):
    """Kernel launches on the default stream in a unit that otherwise uses streams."""

    query = """
            SELECT * FROM kernel_launch_facts
            WHERE unit_uses_streams AND stream IN ('', '0', 'NULL', 'nullptr', 'cudaStreamDefault')
        """
    message = (
        "`{fact_id}` launches a kernel on the default stream in a unit that otherwise uses streams"
    )
    # The launch's configuration takes the unit's one `cudaStream_t` as its stream, after a dynamic
    # shared memory size of `0` when it names none. Left alone: a unit declaring several streams
    # (or one in a container, as a pointer or an array), one declared after the launch or in
    # another function, and a launch that names the default stream outright. The CUDA grammar
    # of tree-sitter reads the configuration as a template argument list, so a launch whose
    # configuration holds `<` or `>` operators of its own is read wrongly and fixed no better.
    fix = r"""
        WITH configured AS (
          SELECT l.path, l.start_line, l.start_column, o.node_id AS launch, g.node_id AS node,
                 list(it.node_id ORDER BY it.sibling_index) AS items
          FROM finding l
          JOIN ast_nodes o ON o.path = l.path AND o.type = 'binary_expression'
            AND (o.start_line, o.start_column, o.end_line, o.end_column)
              = (l.start_line, l.start_column, l.end_line, l.end_column)
          JOIN ast_nodes i ON i.path = o.path AND i.parent_id = o.node_id AND i.sibling_index = 0
            AND i.type = 'binary_expression'
          JOIN ast_nodes t ON t.path = o.path AND t.parent_id = i.node_id
            AND t.type = 'template_function'
          JOIN ast_nodes g ON g.path = o.path AND g.parent_id = t.node_id
            AND g.type = 'template_argument_list'
          JOIN ast_nodes it ON it.path = o.path AND it.parent_id = g.node_id
            AND it.type NOT IN ('<', '>', ',')
          GROUP BY ALL
          HAVING bool_and(it.semantic_type <> 'METADATA_COMMENT')
        ), kinds AS (
          SELECT path, node_id, parent_id, count(*) OVER (PARTITION BY path) AS held
          FROM ast_nodes WHERE type = 'type_identifier' AND name = 'cudaStream_t'
        ), declarators AS (
          SELECT k.path, d.node_id AS declaration, d.type AS kind, c.type AS declarator, c.name,
                 (SELECT max(f.node_id) FROM ast_nodes f
                  WHERE f.path = k.path AND f.type = 'function_definition'
                    AND d.node_id BETWEEN f.node_id AND f.node_id + f.descendant_count) AS home
          FROM kinds k
          JOIN ast_nodes d ON d.path = k.path AND d.node_id = k.parent_id
            AND d.type IN ('declaration', 'parameter_declaration')
          JOIN ast_nodes c ON c.path = k.path AND c.parent_id = d.node_id
            AND c.node_id <> k.node_id AND (c.type = 'identifier' OR c.type LIKE '%declarator')
          WHERE k.held = 1
        ), declared AS (
          SELECT x.path, x.declaration, any_value(x.name) AS name, any_value(x.home) AS home,
                 any_value(h.node_id + h.descendant_count) AS home_end
          FROM declarators x
          LEFT JOIN ast_nodes h ON h.path = x.path AND h.node_id = x.home
          GROUP BY x.path, x.declaration
          HAVING count(*) = 1 AND bool_and(x.declarator IN ('identifier', 'init_declarator'))
             AND regexp_matches(any_value(x.name), '^[A-Za-z_][A-Za-z0-9_]*$')
             AND (any_value(x.kind) = 'declaration' OR any_value(x.home) IS NOT NULL)
        ), placed AS (
          SELECT c.path, c.start_line, c.start_column, c.node, c.items, s.name
          FROM configured c
          JOIN declared s ON s.path = c.path AND s.declaration < c.launch
            AND (s.home IS NULL OR c.launch BETWEEN s.home AND s.home_end)
          WHERE len(c.items) IN (2, 3)
        )
        SELECT f.finding_key, 'rewrite-node' AS operation, p.path, p.node,
               MAP {'CFG': p.items} AS captures,
               '<$$$CFG, ' || if(len(p.items) = 2, '0, ', '') || p.name || '>' AS template
        FROM finding f
        JOIN placed p ON p.path = f.path AND p.start_line = f.start_line
          AND p.start_column = f.start_column
    """
    examples = (
        Example(
            bad=_SCALE
            + "void run(float* d) {\n    cudaStream_t s;\n    cudaStreamCreate(&s);\n    scale<<<1, 32>>>(d);\n}\n",
            good=_SCALE
            + "void run(float* d) {\n    cudaStream_t s;\n    cudaStreamCreate(&s);\n    scale<<<1, 32, 0, s>>>(d);\n}\n",
        ),
        Example(
            bad=_SCALE
            + "void run(float* d, cudaStream_t s) {\n    scale<<<4, 128, 16>>>(d);\n}\n",
            good=_SCALE
            + "void run(float* d, cudaStream_t s) {\n    scale<<<4, 128, 16, s>>>(d);\n}\n",
        ),
        Example(
            bad=_SCALE
            + "void run(float* d) {\n    cudaStream_t a, b;\n    cudaStreamCreate(&a);\n    scale<<<1, 32>>>(d);\n}\n",
            good=_SCALE
            + "void run(float* d) {\n    cudaStream_t a, b;\n    cudaStreamCreate(&a);\n    scale<<<1, 32>>>(d);\n}\n",
        ),
        Example(
            bad=_SCALE
            + "void run(float* d, cudaStream_t s) {\n    scale<<<1, 32, 0, 0>>>(d);\n}\n",
            good=_SCALE
            + "void run(float* d, cudaStream_t s) {\n    scale<<<1, 32, 0, 0>>>(d);\n}\n",
        ),
        Example(
            bad=_SCALE
            + "void run(float* d, cudaStream_t s) {\n    scale<<<1, /* block */ 32>>>(d);\n}\n",
            good=_SCALE
            + "void run(float* d, cudaStream_t s) {\n    scale<<<1, /* block */ 32>>>(d);\n}\n",
        ),
        Example(
            bad=_SCALE
            + "void make() {\n    cudaStream_t s;\n    cudaStreamCreate(&s);\n}\n\nvoid run(float* d) {\n    scale<<<1, 32>>>(d);\n}\n",
            good=_SCALE
            + "void make() {\n    cudaStream_t s;\n    cudaStreamCreate(&s);\n}\n\nvoid run(float* d) {\n    scale<<<1, 32>>>(d);\n}\n",
        ),
    )
