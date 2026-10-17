"""Where code names a CUDA stream, makes one, or lets the framework pick the current one."""

from .cuda_base import CudaRelation


class CudaStreamParameters(CudaRelation):
    """The parameters that carry a stream into a function: named `stream`, `streams` or `lane`,
    or annotated with a `Stream` type."""

    stored = True
    query = """
        SELECT fact_id, function_id, name
        FROM cuda_parameters()
        WHERE regexp_matches(name, '(?i)(^|_)(streams?|lane)$') OR regexp_matches(annotation, 'Stream\\b')
    """


class CudaStreamCreations(CudaRelation):
    """Calls that create a stream: `cupy.cuda.Stream(...)`, `torch.cuda.Stream(...)`,
    `numba.cuda.stream()`, `device.create_stream()`."""

    stored = True
    query = """
        SELECT fact_id, path, node_id, scope_id
        FROM cuda_calls()
        WHERE qualified_name IN ('cupy.cuda.Stream', 'torch.cuda.Stream', 'numba.cuda.stream',
                                 'cuda.core.Stream', 'cuda.core.experimental.Stream')
           OR method = 'create_stream'
           OR regexp_matches(callee_text, '(^|\\.)cuda\\.Stream$')
    """


class CudaAmbientStreams(CudaRelation):
    """Calls that read the stream that happens to be current: `get_current_stream()` and
    `torch.cuda.current_stream()`."""

    stored = True
    query = """
        SELECT fact_id, path, node_id, scope_id, start_line, start_column, end_line, end_column,
               shown, CASE WHEN callee_text LIKE '%torch%' THEN 'torch' ELSE 'cupy' END AS framework
        FROM cuda_calls()
        WHERE regexp_matches(callee_text, '(^|\\.)(get_current_stream|current_stream)$')
    """


class CudaStreamFiles(CudaRelation):
    """The files that make or take streams, where the default stream is the odd one out."""

    stored = True
    query = """
        SELECT DISTINCT fact_id FROM cuda_stream_creations()
        UNION
        SELECT DISTINCT fact_id FROM cuda_stream_parameters()
    """
