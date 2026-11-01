"""PY-CUDA rules about which stream device work runs on."""

from .....catalog import Example, Relation, Rule
from ...constants import cuda_paths

_BLOCKING = (
    "cuMemAlloc", "cuMemAllocManaged", "cuMemAllocPitch", "cuMemFree", "cuMemFreeHost",
    "cuMemHostAlloc", "cuMemcpy", "cuMemcpy2D", "cuMemcpy3D", "cuMemcpyDtoD", "cuMemcpyDtoH",
    "cuMemcpyHtoD", "cuMemcpyPeer", "cudaFree", "cudaFreeHost", "cudaHostAlloc", "cudaMalloc",
    "cudaMalloc3D", "cudaMallocHost", "cudaMallocManaged", "cudaMallocPitch", "cudaMemcpy",
    "cudaMemcpy2D", "cudaMemcpy3D", "cudaMemcpyFromSymbol", "cudaMemcpyPeer",
    "cudaMemcpyToSymbol", "cudaMemset", "cudaMemset2D", "cudaMemset3D",
)  # fmt: skip
# The fix rows of the findings of these problems that `CudaStreamedCalls` can give a stream, which
# it finds by where the call stands.
_STREAMED = """
    SELECT f.finding_key, 'rewrite-node' AS operation, f.path, x.node, x.captures, x.template
    FROM finding f
    JOIN cuda_streamed_calls() x ON x.path = f.path AND x.start_line = f.start_line
      AND x.start_column = f.start_column
    WHERE f.problem IN ({problems})
"""
# The fix row of a device-wide barrier, found by where the call stands and what it says: the call
# waits on the one stream its function holds instead, when that stream is its framework's (a
# parameter's framework is unknown). The row captures the callee to drop it, so a call with
# arguments (a device to wait for) keeps its leaves outside every capture and is refused.
_BARRIER = """
    SELECT f.finding_key, 'rewrite-node' AS operation, f.path, c.node_id AS node,
           MAP {'CALLEE': [c.callee_id]} AS captures, s.name || '.synchronize()' AS template
    FROM finding f
    JOIN cuda_calls() c ON c.path = f.path AND c.start_line = f.start_line
      AND c.start_column = f.start_column AND c.shown = f.shown
    JOIN cuda_owned_streams() s ON s.fact_id = c.fact_id AND s.scope_id = c.scope_id
      AND s.declared < c.node_id
      AND (s.origin = 'parameter'
           OR split_part(s.origin, '.', 1) = split_part(c.qualified_name, '.', 1))
    WHERE f.problem = 'barrier'
"""


class CudaOwnedStreams(Relation):
    """The one stream a function holds, by the name it is known as: a parameter named `stream` or
    the target a stream creation is assigned to, where it is declared (`declared`, a node id that
    precedes everything after it) and what made it (`origin`, `parameter` or the creation's
    qualified name). A function holding several, or one under another name (`streams`, `lane`,
    an attribute), has none."""

    stored = True
    query = r"""
        WITH held AS (
          SELECT fact_id, function_id AS scope_id, name, function_id AS declared,
                 'parameter' AS origin, regexp_matches(name, '(?i)(^|_)stream$') AS plain
          FROM cuda_stream_parameters()
          UNION ALL
          SELECT s.fact_id, s.scope_id, c.assigned_target, s.node_id, c.qualified_name,
                 regexp_matches(coalesce(c.assigned_target, ''), '^[A-Za-z_][A-Za-z0-9_]*$')
          FROM cuda_stream_creations() s
          JOIN cuda_calls() c ON c.fact_id = s.fact_id AND c.node_id = s.node_id
        )
        SELECT fact_id, scope_id, any_value(name) AS name, any_value(declared) AS declared,
               any_value(origin) AS origin
        FROM held GROUP BY fact_id, scope_id HAVING count(*) = 1 AND bool_and(plain)
    """


class CudaStreamedCalls(Relation):
    """The launches and copies that can be given the stream their function holds, and how.

    A numba launch `kernel[grid, block](x)` takes the stream as a third configuration item; a patos
    launch `place[1](x)` and a numba copy to the device take `stream=` after their arguments (a
    copy to the host would hand the caller data that may not have arrived). `node` is the syntax
    node the rewrite replaces (the subscript, the argument list) and `captures` and `template` are
    the `rewrite-node` row's, `$V` the kernel and `$$$CFG` the configuration. Only a call after
    the stream is declared counts, with a stream its framework takes (a numba call a parameter or a
    numba stream, a patos launch a parameter) and, for a numba launch, exactly the two items.
    """

    stored = True
    query = r"""
        SELECT l.path, l.start_line, l.start_column, c.callee_id AS node,
               MAP {'V': [any_value(v.node_id)], 'CFG': list(g.node_id ORDER BY g.sibling_index)}
                 AS captures,
               '$V[$$$CFG, ' || any_value(s.name) || ']' AS template
        FROM cuda_launches() l
        JOIN cuda_calls() c ON c.fact_id = l.fact_id AND c.node_id = l.node_id
          AND c.callee_type = 'subscript'
        JOIN cuda_owned_streams() s ON s.fact_id = l.fact_id AND s.scope_id = l.scope_id
          AND s.declared < l.node_id AND s.origin IN ('parameter', 'numba.cuda.stream')
        JOIN cuda_nodes() v ON v.fact_id = c.fact_id AND v.parent_id = c.callee_id
          AND v.sibling_index = 0
        JOIN cuda_nodes() g ON g.fact_id = c.fact_id AND g.parent_id = c.callee_id
          AND g.sibling_index >= 2 AND g.semantic_type NOT IN ('PARSER_DELIMITER', 'METADATA_COMMENT')
        WHERE l.framework = 'numba' AND l.stream_kind = 'none'
        GROUP BY l.path, l.start_line, l.start_column, c.callee_id
        HAVING count(*) = 2
        UNION ALL
        SELECT l.path, l.start_line, l.start_column, c.arguments_id, k.captures, k.template
        FROM cuda_launches() l
        JOIN cuda_calls() c ON c.fact_id = l.fact_id AND c.node_id = l.node_id
        JOIN cuda_nodes() a ON a.fact_id = c.fact_id AND a.node_id = c.arguments_id
          AND a.type = 'argument_list'
        JOIN cuda_owned_streams() s ON s.fact_id = l.fact_id AND s.scope_id = l.scope_id
          AND s.declared < l.node_id AND s.origin = 'parameter',
        keyword_appended(l.path, c.arguments_id, 'stream=' || s.name) k
        WHERE l.framework = 'patos' AND l.stream_kind = 'ambient'
        UNION ALL
        SELECT c.path, c.start_line, c.start_column, c.arguments_id, k.captures, k.template
        FROM cuda_calls() c
        JOIN cuda_nodes() a ON a.fact_id = c.fact_id AND a.node_id = c.arguments_id
          AND a.type = 'argument_list'
        JOIN cuda_owned_streams() s ON s.fact_id = c.fact_id AND s.scope_id = c.scope_id
          AND s.declared < c.node_id AND s.origin IN ('parameter', 'numba.cuda.stream'),
        keyword_appended(c.path, c.arguments_id, 'stream=' || s.name) k
        WHERE c.qualified_name = 'numba.cuda.to_device' OR c.method = 'copy_to_device'
    """


class StreamNotChosen(Rule, id="PY-CUDA0003", applicability="review"):
    """Device work queued on whichever stream is current: a function that reads the ambient stream
    and takes none, or a launch that ignores the stream its function holds."""

    exempt: tuple[str, ...] = cuda_paths.MEASURING

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
                 'ignored' AS problem, f.qualname AS function_name
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
        FROM (SELECT * FROM ambient UNION ALL BY NAME SELECT * FROM ignored) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{shown}` {what}: {cost}. Fix: {remedy}"
    # The `ignored` half: a launch of a function holding exactly one stream is given it
    # (`CudaStreamedCalls` states how, per framework). The `ambient` half needs a parameter
    # added to the function and to every caller, which is a refactor of its own, and a launch
    # whose framework or configuration is not one `CudaStreamedCalls` knows is left alone.
    fix = _STREAMED.format(problems="'ignored'")
    examples = (
        Example(
            bad=(
                """import cupy as cp


def scale(x, k):
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    return x * k
"""
            ),
            good=(
                """import cupy as cp


def scale(x, k):
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    return x * k
"""
            ),
        ),
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x, stream):\n    kernel[1, 32](x)\n",
            good="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x, stream):\n    kernel[1, 32, stream](x)\n",
        ),
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x, stream):\n    other = cuda.stream()\n    kernel[1, 32](x)\n",
            good="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x, stream):\n    other = cuda.stream()\n    kernel[1, 32](x)\n",
        ),
        Example(
            bad=(
                """from patos.cuda.typed import kernel


@kernel(threads=256)
def place(x) -> None:
    x[0] = 1


def run(x, stream):
    place[1](x)
"""
            ),
            good=(
                """from patos.cuda.typed import kernel


@kernel(threads=256)
def place(x) -> None:
    x[0] = 1


def run(x, stream):
    place[1](x, stream=stream)
"""
            ),
        ),
    )


class DefaultStreamWork(Rule, id="PY-CUDA0004", applicability="review"):
    """Work on the legacy default stream, which synchronizes with every other stream: a launch or
    copy without a stream in a function that makes a stream, the legacy stream named outright, a
    device-wide barrier, or a blocking runtime copy or allocation. A stream made in another
    function does not make this one a stream user."""

    exempt: tuple[str, ...] = cuda_paths.MEASURING
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
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown,
                 'legacy' AS problem
          FROM cuda_calls() c
          WHERE (c.method = 'launch' OR c.qualified_name IN ('cuda.core.launch',
                                                             'cuda.core.experimental.launch'))
            AND regexp_matches(c.node_text, 'LEGACY_DEFAULT_STREAM|Stream\.null\b|default_stream\(')
        ), barriers AS (
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown,
                 'barrier' AS problem
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
          SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown,
                 'raw' AS problem
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
        FROM (SELECT * FROM unstreamed UNION ALL SELECT * FROM named
              UNION ALL SELECT * FROM barriers UNION ALL SELECT * FROM raw) f
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(f.path, p))), false)
    """
    message = "`{shown}` {what} in code that otherwise uses streams: {cost}. Fix: {remedy}"
    # A launch or a host to device copy of a function holding exactly one stream is given it
    # (`CudaStreamedCalls`), and a device-wide barrier becomes a wait on that stream (`_BARRIER`).
    # Left to a person: the legacy stream named outright, the blocking runtime calls, a copy back
    # to the host (its caller would read data that may not have arrived), every call of a function
    # holding several streams, and a barrier of another framework than its stream's.
    fix = _STREAMED.format(problems="'launch', 'copy'") + "UNION ALL" + _BARRIER
    examples = (
        Example(
            bad=(
                """from numba import cuda


@cuda.jit
def kernel(x):
    x[0] = 1


def run(x):
    stream = cuda.stream()
    kernel[1, 32](x)
"""
            ),
            good=(
                """from numba import cuda


@cuda.jit
def kernel(x):
    x[0] = 1


def run(x):
    stream = cuda.stream()
    kernel[1, 32, stream](x)
"""
            ),
        ),
        Example(
            bad="from numba import cuda\n\n\ndef run(x):\n    stream = cuda.stream()\n    cuda.synchronize()\n",
            good="from numba import cuda\n\n\ndef run(x):\n    stream = cuda.stream()\n    stream.synchronize()\n",
        ),
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x):\n    stream = cuda.stream()\n    kernel[1, 32](x)\n    stream.synchronize()\n",
            good="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x):\n    stream = cuda.stream()\n    kernel[1, 32, stream](x)\n    stream.synchronize()\n",
        ),
        Example(
            bad="import torch\n\n\ndef run():\n    stream = torch.cuda.Stream()\n    torch.cuda.synchronize()\n",
            good="import torch\n\n\ndef run():\n    stream = torch.cuda.Stream()\n    stream.synchronize()\n",
        ),
        Example(
            bad="from numba import cuda\n\n\ndef run(stream):\n    other = cuda.stream()\n    cuda.synchronize()\n",
            good="from numba import cuda\n\n\ndef run(stream):\n    other = cuda.stream()\n    cuda.synchronize()\n",
        ),
        Example(
            bad="import torch\n\n\ndef run(device):\n    stream = torch.cuda.Stream()\n    torch.cuda.synchronize(device)\n",
            good="import torch\n\n\ndef run(device):\n    stream = torch.cuda.Stream()\n    torch.cuda.synchronize(device)\n",
        ),
        Example(
            bad="import torch\nfrom numba import cuda\n\n\ndef run():\n    stream = torch.cuda.Stream()\n    cuda.synchronize()\n",
            good="import torch\nfrom numba import cuda\n\n\ndef run():\n    stream = torch.cuda.Stream()\n    cuda.synchronize()\n",
        ),
        Example(
            bad="from numba import cuda\n\n\ndef run(host):\n    stream = cuda.stream()\n    return cuda.to_device(host)\n",
            good="from numba import cuda\n\n\ndef run(host):\n    stream = cuda.stream()\n    return cuda.to_device(host, stream=stream)\n",
        ),
        Example(
            bad="from numba import cuda\n\n\ndef run():\n    stream = cuda.stream()\n    device = cuda.device_array(4)\n    return device.copy_to_host()\n",
            good="from numba import cuda\n\n\ndef run():\n    stream = cuda.stream()\n    device = cuda.device_array(4)\n    return device.copy_to_host()\n",
        ),
        Example(
            bad="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x):\n    first = cuda.stream()\n    second = cuda.stream()\n    kernel[1, 32](x)\n",
            good="from numba import cuda\n\n\n@cuda.jit\ndef kernel(x):\n    x[0] = 1\n\n\ndef run(x):\n    first = cuda.stream()\n    second = cuda.stream()\n    kernel[1, 32](x)\n",
        ),
    )
