"""Rules over avoidable cost (mcmr's PY-PERF family)."""

from ....catalog import Example, Rule

_DEVICE_DESTINATIONS = (
    "cupy.array",
    "cupy.asarray",
    "cupy.from_dlpack",
    "cudf.DataFrame.from_pandas",
    "cudf.Series.from_pandas",
    "torch.as_tensor",
    "torch.from_numpy",
    "torch.tensor",
    "torch.utils.dlpack.from_dlpack",
)
_HOST_BRIDGES = (
    "cudf.DataFrame.to_pandas",
    "cudf.Series.to_pandas",
    "cupy.asnumpy",
    "cupy.ndarray.get",
    "cupy.ndarray.toDlpack",
    "torch.Tensor.cpu",
    "torch.Tensor.numpy",
    "torch.utils.dlpack.to_dlpack",
)


class TensorInteroperabilityRoundTripCount(Rule, id="PY-PERF0001", applicability="review"):
    """Avoidable host, NumPy, or explicit DLPack tensor round trips."""

    query = """
        WITH bridges AS (
          SELECT call_id, string_agg('`' || qualified_name || '`', ', ' ORDER BY preorder) AS names
          FROM call_expressions e
          WHERE root_relation = 'argument' AND list_contains($bridges::VARCHAR[], qualified_name)
            AND NOT EXISTS (SELECT 1 FROM call_expression_ancestry a
                            WHERE a.relation = 'mapping_value'
                              AND a.descendant_expression_id = e.expression_id)
          GROUP BY call_id
        )
        SELECT c.*, c.node_path AS path, c.node_start_line AS start_line,
               c.node_start_column AS start_column, c.node_end_line AS end_line,
               c.node_end_column AS end_column, b.names AS bridge_names
        FROM call_calls c JOIN bridges b USING (call_id)
        WHERE list_contains($destinations::VARCHAR[], c.qualified_name) AND NOT c.is_shadowed
    """
    message = "`{qualified_name}` receives a value routed through {bridge_names}"
    examples = (
        Example(
            bad="""import cupy as cp
import torch


def lift(array):
    return torch.from_numpy(cp.asnumpy(array))
""",
            good="""import cupy as cp
import torch


def lift(array):
    return torch.as_tensor(array)
""",
        ),
    )
    # The bridge call that is itself an argument goes, leaving its operand, for the pairs whose
    # destination takes a device object directly (CuPy arrays and DLPack objects through the
    # CUDA array interface); `torch.from_numpy` becomes `torch.as_tensor`, which does. Bridges
    # that are methods (`x.cpu().numpy()`) are never resolved by the facts, so none is reported.
    fix = """
        WITH pairs(destination, bridge, replacement) AS (VALUES
          ('cupy.array', 'cupy.asnumpy', NULL), ('cupy.asarray', 'cupy.asnumpy', NULL),
          ('cupy.from_dlpack', 'torch.utils.dlpack.to_dlpack', NULL),
          ('torch.as_tensor', 'cupy.asnumpy', NULL),
          ('torch.from_numpy', 'cupy.asnumpy', 'as_tensor')
        ), bridged AS (
          SELECT f.finding_key, f.path, f.callee_text, f.callee_start_line, f.callee_start_column,
                 f.callee_end_line, f.callee_end_column, p.replacement,
                 e.node_start_line AS bridge_start_line,
                 e.node_start_column AS bridge_start_column,
                 e.node_end_line AS bridge_end_line, e.node_end_column AS bridge_end_column,
                 i.text AS operand
          FROM finding f
          JOIN call_expressions e
            ON e.call_id = f.call_id AND e.root_relation = 'argument' AND e.depth = 0
          JOIN pairs p ON p.destination = f.qualified_name AND p.bridge = e.qualified_name
          JOIN call_calls b ON b.node_path = e.node_path AND b.qualified_name = e.qualified_name
            AND b.node_start_line = e.node_start_line AND b.node_start_column = e.node_start_column
            AND b.node_end_line = e.node_end_line AND b.node_end_column = e.node_end_column
          JOIN call_expressions i
            ON i.call_id = b.call_id AND i.root_relation = 'argument' AND i.depth = 0
          WHERE (p.replacement IS NULL OR contains(f.callee_text, '.'))
            AND NOT b.has_starred_arguments
            AND NOT EXISTS (SELECT 1 FROM call_keywords k WHERE k.call_id = b.call_id)
            AND (SELECT count(*) FROM call_expressions x
                 WHERE x.call_id = b.call_id AND x.root_relation = 'argument'
                   AND x.depth = 0) = 1
        )
        SELECT finding_key, path, bridge_start_line AS start_line,
               bridge_start_column AS start_column, bridge_end_line AS end_line,
               bridge_end_column AS end_column, operand AS text
        FROM bridged
        UNION ALL BY NAME
        SELECT finding_key, path, callee_start_line AS start_line,
               callee_start_column AS start_column, callee_end_line AS end_line,
               callee_end_column AS end_column,
               regexp_replace(callee_text, '[^.]+$', replacement) AS text
        FROM bridged WHERE replacement IS NOT NULL
    """

    constants = {"bridges": _HOST_BRIDGES, "destinations": _DEVICE_DESTINATIONS}
