"""Rules over how CUDA code moves memory (mcmr's CU-MEMO family)."""

from ...catalog import Rule


class SynchronousTransferInStreamScope(Rule, id="CU-MEMO0001"):
    """Blocking transfers in a translation unit that creates streams to overlap them."""

    query = """
            WITH streams AS (
              SELECT fact_id, string_agg('`' || node_path || ':' || node_start_line
                       || CASE WHEN node_end_line > node_start_line THEN '-' || node_end_line
                               ELSE '' END || '`', ', ' ORDER BY ordinal) AS stream_locations
              FROM call_calls
              WHERE qualified_name IN ('cudaStreamCreate', 'cudaStreamCreateWithFlags')
              GROUP BY fact_id
            )
            SELECT c.*, s.stream_locations FROM cuda_call_nodes() c JOIN streams s USING (fact_id)
            WHERE c.qualified_name IN ('cudaMemcpy', 'cudaMemset', 'cudaMemcpy2D', 'cudaMemcpy3D')
        """
    message = "`{qualified_name}` blocks in a scope that creates streams at {stream_locations}"
