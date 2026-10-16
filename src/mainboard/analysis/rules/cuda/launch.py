"""Rules over how CUDA kernels are launched and synchronized (mcmr's CU-LAUN family).

`CU-` rules read CUDA facts only (the engine's language scope), so no rule filters by language.
"""

from ...catalog import Example, Relation, Rule


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
                "__global__ void scale(float* values) {\n    values[threadIdx.x] *= 2.0f;\n"
                "    __syncthreads();\n}\n"
            ),
            good=(
                "#include <cooperative_groups.h>\n__global__ void scale(float* values) {\n"
                "    values[threadIdx.x] *= 2.0f;\n"
                "    cooperative_groups::this_thread_block().sync();\n}\n"
            ),
        ),
    )


class DefaultStreamKernelLaunch(Rule, id="CU-LAUN0002"):
    """Kernel launches on the default stream in a unit that otherwise uses streams."""

    query = """
            SELECT * FROM kernel_launch_facts
            WHERE unit_uses_streams AND stream IN ('', '0', 'NULL', 'nullptr', 'cudaStreamDefault')
        """
    message = "`{fact_id}` satisfies default stream kernel launch"
