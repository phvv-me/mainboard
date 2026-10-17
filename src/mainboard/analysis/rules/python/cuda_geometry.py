"""PY-CUDA rules about launch geometry."""

from ...catalog import Example, Rule

_GPU = (
    r"\b((?:rtx|gtx)\s*\d{3,4}\w*|a100|h100|h200|gh200|gb10|gb200|b200|v100|t4|l4|l40s?|a10g?|a40|a5000|a6000|"
    r"[345]0[6789]0|ampere|hopper|ada lovelace|blackwell|volta|turing|tesla|quadro)\b"
)
_GEOMETRY = (
    r"(?i)(^|_)(blocks?|threads?|warps?|grid|tiles?|ilp|waves?|ctas?|unroll|occupancy|sms?)(_|$)"
)


class GeometryTunedOnOneDevice(Rule, id="PY-CUDA0008", applicability="review"):
    """A launch geometry constant (block, threads, tile, ILP width ...) whose comment says it was
    tuned on a named GPU: it is hard-coded where the device's own limits are available."""

    geometry: str = _GEOMETRY
    gpu: str = _GPU

    query = r"""
        WITH constants AS (
          SELECT a.fact_id, a.path, a.name, a.start_line, a.start_column, a.end_line, a.end_column,
                 regexp_replace(syntax_slice(s.content, v.start_line, v.start_column, v.end_line,
                                             v.end_column), '\s+', ' ', 'g') AS value
          FROM cuda_nodes() a
          JOIN source_files s ON s.path = a.path
          JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.parent_id = a.node_id
            AND v.sibling_index = (SELECT max(w.sibling_index) FROM cuda_nodes() w
                                   WHERE w.fact_id = a.fact_id AND w.parent_id = a.node_id)
          WHERE a.type = 'assignment' AND regexp_matches(a.name, $geometry)
            AND v.type IN ('integer', 'binary_operator')
            AND (a.scope."function" IS NULL)
        ), noted AS (
          SELECT c.*,
                 coalesce((SELECT string_agg(l.text, ' ' ORDER BY l.line)
                           FROM source_lines() l
                           WHERE l.path = c.path AND l.line < c.start_line
                             AND l.line >= c.start_line::BIGINT - 8 AND regexp_matches(l.text, '^\s*#')
                             AND NOT EXISTS (SELECT 1 FROM source_lines() m
                                             WHERE m.path = l.path AND m.line > l.line
                                               AND m.line < c.start_line
                                               AND NOT regexp_matches(m.text, '^\s*#'))), '')
                 || ' ' || coalesce((SELECT l.text FROM source_lines() l
                                     WHERE l.path = c.path AND l.line = c.start_line), '') AS note
          FROM constants c
        )
        SELECT path, start_line, start_column, end_line, end_column, name, value,
               regexp_extract(lower(note), $gpu, 1) AS device
        FROM noted
        WHERE regexp_matches(lower(note), $gpu)
    """
    message = (
        "`{name} = {value}` is a launch geometry tuned on the {device}: another GPU has other "
        "limits (SMs, shared memory, registers), so this value leaves it slower or fails to "
        "compile there. Fix: derive it from the device (SM count, occupancy) or pick it per "
        "device"
    )
    examples = (
        Example(
            bad=(
                "import cupy as cp\n\n# Threads per block, tuned on an RTX 4090.\nBLOCK = 256\n\n\n"
                "def grid(items: int) -> int:\n    return -(-items // BLOCK)\n"
            ),
            good=(
                "import cupy as cp\n\n# Threads per block.\nBLOCK = 256\n\n\n"
                "def grid(items: int) -> int:\n    return -(-items // BLOCK)\n"
            ),
        ),
    )
