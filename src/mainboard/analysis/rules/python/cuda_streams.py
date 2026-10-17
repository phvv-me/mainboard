"""PY-CUDA rules about which stream device work runs on."""

from ...catalog import Example, Rule
from .cuda_paths import MEASURING

_BLOCKING = (
    "cuMemAlloc", "cuMemAllocManaged", "cuMemAllocPitch", "cuMemFree", "cuMemFreeHost",
    "cuMemHostAlloc", "cuMemcpy", "cuMemcpy2D", "cuMemcpy3D", "cuMemcpyDtoD", "cuMemcpyDtoH",
    "cuMemcpyHtoD", "cuMemcpyPeer", "cudaFree", "cudaFreeHost", "cudaHostAlloc", "cudaMalloc",
    "cudaMalloc3D", "cudaMallocHost", "cudaMallocManaged", "cudaMallocPitch", "cudaMemcpy",
    "cudaMemcpy2D", "cudaMemcpy3D", "cudaMemcpyFromSymbol", "cudaMemcpyPeer",
    "cudaMemcpyToSymbol", "cudaMemset", "cudaMemset2D", "cudaMemset3D",
)  # fmt: skip


class StreamNotChosen(Rule, id="PY-CUDA0003", applicability="review"):
    """Device work queued on whichever stream is current: a function that reads the ambient stream
    and takes none, or a launch that ignores the stream its function holds."""

    exempt: tuple[str, ...] = MEASURING

    query = r"""
        WITH ambient AS (
          SELECT a.path, a.start_line, a.start_column, a.end_line, a.end_column, a.shown,
                 'ambient' AS problem, f.qualname AS function_name
          FROM cuda_ambient_streams() a
          LEFT JOIN cuda_definitions() f ON f.fact_id = a.fact_id AND f.node_id = a.scope_id
            AND f.kind = 'function'
          WHERE a.scope_id <> -1
            AND NOT EXISTS (SELECT 1 FROM cuda_stream_parameters() p
                            WHERE p.fact_id = a.fact_id AND p.function_id = a.scope_id)
        ), ignored AS (
          SELECT l.path, l.start_line, l.start_column, l.end_line, l.end_column, l.shown,
                 'ignored', f.qualname
          FROM cuda_launches() l
          LEFT JOIN cuda_definitions() f ON f.fact_id = l.fact_id AND f.node_id = l.scope_id
            AND f.kind = 'function'
          WHERE l.stream_kind IN ('none', 'ambient')
            AND (EXISTS (SELECT 1 FROM cuda_stream_parameters() p
                         WHERE p.fact_id = l.fact_id AND p.function_id = l.scope_id)
                 OR EXISTS (SELECT 1 FROM cuda_stream_creations() s
                            WHERE s.fact_id = l.fact_id AND s.scope_id = l.scope_id))
        )
        SELECT f.*,
               CASE f.problem
                 WHEN 'ambient' THEN 'enqueues on whichever stream is current, so `'
                      || coalesce(f.function_name, 'this code') || '` cannot be told which stream to use'
                 ELSE 'launches on the default or current stream although `'
                      || coalesce(f.function_name, 'this code') || '` holds a stream' END AS what,
               CASE f.problem
                 WHEN 'ambient' THEN 'its work cannot be ordered against another stream''s or '
                      || 'captured on a stream the caller picks'
                 ELSE 'the kernel runs outside the stream the surrounding code orders its other '
                      || 'work on' END AS cost,
               CASE f.problem
                 WHEN 'ambient' THEN 'take a `stream` parameter and enqueue on it, defaulting to '
                      || 'the current stream only where the caller names none'
                 ELSE 'pass the stream to the launch' END AS remedy
        FROM (SELECT * FROM ambient UNION ALL SELECT * FROM ignored) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{shown}` {what}: {cost}. Fix: {remedy}"
    examples = (
        Example(
            bad=(
                "import cupy as cp\n\n\ndef scale(x, k):\n"
                "    stream = cp.cuda.get_current_stream()\n"
                "    stream.synchronize()\n    return x * k\n"
            ),
            good=(
                "import cupy as cp\n\n\ndef scale(x, k, stream):\n"
                "    stream.synchronize()\n    return x * k\n"
            ),
        ),
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x, stream):\n    kernel[1, 32](x)\n",
            good="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x, stream):\n    kernel[1, 32, stream](x)\n",
        ),
        Example(
            bad=(
                "from patos.cuda.typed import kernel\n\n\n@kernel(threads=256)\ndef place(x) -> None:\n"
                "    x[0] = 1\n\n\ndef run(x, stream):\n    place[1](x)\n"
            ),
            good=(
                "from patos.cuda.typed import kernel\n\n\n@kernel(threads=256)\ndef place(x) -> None:\n"
                "    x[0] = 1\n\n\ndef run(x, stream):\n    place[1](x, stream=stream)\n"
            ),
        ),
    )


class DefaultStreamWork(Rule, id="PY-CUDA0004", applicability="review"):
    """Work on the legacy default stream, which synchronizes with every other stream: a launch or
    copy without a stream in a function that makes a stream, the legacy stream named outright, a
    device-wide barrier, or a blocking runtime copy or allocation. A stream made in another
    function does not make this one a stream user."""

    exempt: tuple[str, ...] = MEASURING
    blocking: tuple[str, ...] = _BLOCKING

    query = r"""
        WITH streamed AS (
          SELECT DISTINCT fact_id, scope_id FROM cuda_stream_creations()
        ), unstreamed AS (
          SELECT l.path, l.start_line, l.start_column, l.end_line, l.end_column, l.shown,
                 'launch' AS problem
          FROM cuda_launches() l JOIN streamed s ON s.fact_id = l.fact_id AND s.scope_id = l.scope_id
          WHERE l.framework = 'numba' AND l.stream_kind = 'none'
          UNION ALL
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown, 'copy'
          FROM cuda_calls() c JOIN streamed s ON s.fact_id = c.fact_id AND s.scope_id = c.scope_id
          WHERE (c.qualified_name = 'numba.cuda.to_device'
                 OR (c.method IN ('copy_to_device', 'copy_to_host')
                     AND EXISTS (SELECT 1 FROM cuda_types() t
                                 WHERE t.key = 'e:' || c.fact_id || ':' || c.receiver_id
                                   AND t.fw = 'numba')))
            AND NOT EXISTS (SELECT 1 FROM cuda_arguments() a
                            WHERE a.fact_id = c.fact_id AND a.call_id = c.node_id
                              AND (a.keyword = 'stream' OR a.position >= 1))
        ), named AS (
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown, 'legacy'
          FROM cuda_calls() c
          WHERE (c.method = 'launch' OR c.qualified_name IN ('cuda.core.launch',
                                                             'cuda.core.experimental.launch'))
            AND regexp_matches(c.node_text, 'LEGACY_DEFAULT_STREAM|Stream\.null\b|default_stream\(')
        ), barriers AS (
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown, 'barrier'
          FROM cuda_calls() c
          WHERE (EXISTS (SELECT 1 FROM cuda_stream_parameters() p
                         WHERE p.fact_id = c.fact_id AND p.function_id = c.scope_id)
                 OR EXISTS (SELECT 1 FROM cuda_stream_creations() s
                            WHERE s.fact_id = c.fact_id AND s.scope_id = c.scope_id))
            AND (c.qualified_name IN ('numba.cuda.synchronize', 'torch.cuda.synchronize',
                                      'cupy.cuda.runtime.deviceSynchronize')
                 OR regexp_matches(c.callee_text, '(^|\.)(cudaDeviceSynchronize|cuCtxSynchronize)$'))
            AND NOT EXISTS (SELECT 1 FROM cuda_clocks() k
                            WHERE k.fact_id = c.fact_id AND k.scope_id = c.scope_id)
        ), raw AS (
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown, 'raw'
          FROM cuda_calls() c JOIN streamed s ON s.fact_id = c.fact_id AND s.scope_id = c.scope_id
          WHERE contains(c.callee_text, 'cu')
            AND list_contains($blocking, regexp_extract(c.callee_text, '[A-Za-z0-9_]+$'))
        )
        SELECT f.*,
               CASE f.problem
                 WHEN 'launch' THEN 'launches on the legacy default stream'
                 WHEN 'copy' THEN 'copies on the legacy default stream'
                 WHEN 'legacy' THEN 'names the legacy default stream'
                 WHEN 'barrier' THEN 'waits for every stream of the device'
                 ELSE 'is a blocking runtime call that synchronizes with the legacy default stream'
               END AS what,
               CASE f.problem
                 WHEN 'barrier' THEN 'it also stalls streams that have nothing to do with this work'
                 ELSE 'the default stream orders itself against every other stream, so work on '
                      || 'the module''s streams stops overlapping with it' END AS cost,
               CASE f.problem
                 WHEN 'barrier' THEN 'wait on the one stream that matters (`stream.synchronize()`)'
                 WHEN 'raw' THEN 'use the asynchronous variant on a stream (`cudaMemcpyAsync`, '
                      || '`cudaMallocAsync`)'
                 ELSE 'pass a stream explicitly' END AS remedy
        FROM (SELECT * FROM unstreamed UNION ALL SELECT * FROM named UNION ALL SELECT * FROM barriers
              UNION ALL SELECT * FROM raw) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{shown}` {what} in code that otherwise uses streams: {cost}. Fix: {remedy}"
    examples = (
        Example(
            bad=(
                "from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\n"
                "def run(x):\n    stream = cuda.stream()\n    kernel[1, 32](x)\n"
            ),
            good=(
                "from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\n"
                "def run(x):\n    stream = cuda.stream()\n    kernel[1, 32, stream](x)\n"
            ),
        ),
        Example(
            bad="from numba import cuda\n\n\ndef run(x):\n    stream = cuda.stream()\n    cuda.synchronize()\n",
            good="from numba import cuda\n\n\ndef run(x):\n    stream = cuda.stream()\n    stream.synchronize()\n",
        ),
    )
