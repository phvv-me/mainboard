"""PY-CUDA rules about CUDA graph capture."""

from .....catalog import Example, Rule
from ...constants import cuda_paths


class CaptureHazard(Rule, id="PY-CUDA0007", applicability="review"):
    """Code that cannot be recorded in a CUDA graph but runs under capture (a wait, a read of a
    device value, a pageable host copy), and broad exception types a recording turns into a
    silent fallback, in a function (or class) that begins a capture."""

    exempt: tuple[str, ...] = cuda_paths.MEASURING

    query = r"""
        WITH captured_scopes AS (
          SELECT fact_id, function_id AS scope_id, -1 AS first, 2147483647 AS last,
                 region_path, region_line FROM cuda_captured()
          UNION ALL
          SELECT r.fact_id, CAST(NULL AS BIGINT) AS scope_id, r.first, r.last, r.path, n.start_line
          FROM cuda_capture_regions() r
          JOIN cuda_nodes() n ON n.fact_id = r.fact_id AND n.node_id = r.first
        ), unsafe AS (
          SELECT p.path, p.start_line, p.start_column, p.end_line, p.end_column, p.shown,
                 CASE WHEN p.kind = 'explicit' THEN 'read' ELSE 'convert' END AS problem,
                 s.region_path, s.region_line
          FROM cuda_host_reads() p
          JOIN captured_scopes s ON s.fact_id = p.fact_id
            AND (s.scope_id IS NULL OR s.scope_id = p.scope_id)
            AND p.node_id BETWEEN s.first AND s.last
          WHERE p.shown IS NOT NULL
            AND NOT EXISTS (SELECT 1 FROM cuda_uncaptured() u
                            WHERE u.fact_id = p.fact_id AND p.node_id BETWEEN u.first AND u.last)
          UNION ALL
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown,
                 'barrier', s.region_path, s.region_line
          FROM cuda_calls() c
          JOIN captured_scopes s ON s.fact_id = c.fact_id
            AND (s.scope_id IS NULL OR s.scope_id = c.scope_id)
            AND c.node_id BETWEEN s.first AND s.last
          WHERE (c.method IN ('synchronize', 'sync', 'block_until_ready')
                 OR c.qualified_name IN ('torch.cuda.synchronize',
                                         'cupy.cuda.runtime.deviceSynchronize',
                                         'numba.cuda.synchronize'))
            AND NOT EXISTS (SELECT 1 FROM cuda_uncaptured() u
                            WHERE u.fact_id = c.fact_id AND c.node_id BETWEEN u.first AND u.last)
          UNION ALL
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown,
                 'pageable', s.region_path, s.region_line
          FROM cuda_calls() c
          JOIN captured_scopes s ON s.fact_id = c.fact_id
            AND (s.scope_id IS NULL OR s.scope_id = c.scope_id)
            AND c.node_id BETWEEN s.first AND s.last
          JOIN cuda_arguments() a ON a.fact_id = c.fact_id AND a.call_id = c.node_id
            AND a.position = 0
          JOIN cuda_nodes() v ON v.fact_id = a.fact_id AND v.node_id = a.value_id
          LEFT JOIN cuda_calls() inner_call ON inner_call.fact_id = v.fact_id
            AND inner_call.node_id = v.node_id
          WHERE c.qualified_name IN ('cupy.asarray', 'cupy.array')
            AND (v.type IN ('list', 'tuple', 'list_comprehension')
                 OR starts_with(coalesce(inner_call.qualified_name, ''), 'numpy.'))
            AND NOT EXISTS (SELECT 1 FROM cuda_uncaptured() u
                            WHERE u.fact_id = c.fact_id AND c.node_id BETWEEN u.first AND u.last)
        ), capturing AS (
          SELECT DISTINCT fact_id, scope_id, class_id FROM cuda_calls()
          WHERE regexp_matches(callee_text, '(^|\.)(begin_building|begin_capture|capture_begin|create_graph_builder|CUDAGraph)$')
        ), broad AS (
          SELECT x.path, x.start_line, x.start_column, x.end_line, x.end_column, x.name AS shown,
                 'broad' AS problem, CAST(NULL AS VARCHAR) AS region_path,
                 CAST(NULL AS INTEGER) AS region_line
          FROM cuda_nodes() x
          JOIN cuda_nodes() p ON p.fact_id = x.fact_id AND p.node_id = x.parent_id
          WHERE EXISTS (SELECT 1 FROM capturing c
                        WHERE c.fact_id = x.fact_id
                          AND (c.scope_id = coalesce(x.scope."function", -1)
                               OR c.class_id IS NOT NULL AND c.class_id = x.scope."class"))
            AND x.type = 'identifier' AND x.name IN ('RuntimeError', 'Exception', 'BaseException')
            AND p.type IN ('tuple', 'except_clause', 'as_pattern', 'argument_list', 'list')
        )
        SELECT f.*,
               CASE f.problem
                 WHEN 'broad' THEN 'names `' || f.shown || '` among the failures a recording turns '
                      || 'into a fallback'
                 WHEN 'read' THEN 'reads a device value to the host under capture'
                 WHEN 'convert' THEN 'turns a device value into a Python number under capture'
                 WHEN 'barrier' THEN 'waits for the device under capture'
                 ELSE 'copies pageable host memory to the device under capture' END AS what,
               CASE f.problem
                 WHEN 'broad' THEN 'any bug raised as ' || CASE WHEN regexp_matches(f.shown, '^[AEIOUaeiou]')
                      THEN 'an ' ELSE 'a ' END || f.shown || ' while recording (a stray '
                      || 'host read, a numba error) is swallowed and the pass silently runs '
                      || 'eagerly from then on'
                 ELSE 'none of these can be recorded, so the capture fails or records a graph that '
                      || 'is wrong' END AS cost,
               CASE f.problem
                 WHEN 'broad' THEN 'catch only the CUDA error types, and log the cause once when a '
                      || 'pass falls back'
                 WHEN 'pageable' THEN 'upload into a pinned buffer before capture and copy it '
                      || 'asynchronously'
                 ELSE 'do it before capture, or branch on the device (a conditional node)' END AS remedy
        FROM (SELECT * FROM unsafe UNION ALL SELECT * FROM broad) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{shown}` {what}: {cost}. Fix: {remedy}"
    examples = (
        Example(
            bad=(
                """import cupy as cp


def record(store, source, stream):
    return store.capture('pass', lambda: int(cp.cumsum(source)[0]), stream=stream)
"""
            ),
            good=(
                """import cupy as cp


def record(store, source, stream):
    return store.capture('pass', lambda: cp.cumsum(source), stream=stream)
"""
            ),
        ),
        Example(
            bad="import cupy as cp\n\n\ndef record(stream):\n    builder = stream.begin_capture()\n    try:\n        return builder\n    except RuntimeError:\n        return None\n",
            good="import cupy as cp\n\n\ndef record(stream):\n    builder = stream.begin_capture()\n    try:\n        return builder\n    except OSError:\n        return None\n",
        ),
        Example(
            bad=(
                """import cupy as cp


def branch(mode, then):
    flag = cp.zeros(1)
    if mode is not None:
        mode.when(flag, then)
    elif int(flag[0]):
        then()


def record(store, stream, mode, then):
    return store.capture('pass', lambda: branch(mode, then), stream=stream)
"""
            ),
            good=(
                """import cupy as cp


def branch(recording, then):
    flag = cp.zeros(1)
    if recording is not None:
        recording.when(flag, then)
    elif int(flag[0]):
        then()


def record(store, stream, recording, then):
    return store.capture('pass', lambda: branch(recording, then), stream=stream)
"""
            ),
        ),
    )
