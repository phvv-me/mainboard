"""The regions of host code a clock read bounds, and whether the device has finished their work
when the clock stops.

Device work is asynchronous: a launch returns when it is queued. A timer read right after it
measures queueing, not execution, unless something waits for the device first. A fence is a
barrier (`synchronize`), a call that reaches one, or a read of a device value, which waits for
the stream it comes from.
"""

from ..base import CudaRelation


class CudaDeviceWork(CudaRelation):
    """Calls that queue work on the device. `direct` ones are the work itself: a launch, a call
    whose result is a device value, a method called on a device value (`x.fill(0)`). The rest
    call a project function that does such work, up to three calls deep. `framework` is the
    framework of the work."""

    stored = True
    query = """
        WITH RECURSIVE direct(fact_id, path, node_id, scope_id, framework) AS (
          SELECT l.fact_id, l.path, l.node_id, l.scope_id, l.framework FROM cuda_launches() l
          UNION
          SELECT c.fact_id, c.path, c.node_id, c.scope_id, t.fw
          FROM cuda_calls() c
          JOIN cuda_types() t ON t.key = 'e:' || c.fact_id || ':' || c.node_id AND t.ty = 'device'
          UNION
          SELECT c.fact_id, c.path, c.node_id, c.scope_id, t.fw
          FROM cuda_calls() c
          JOIN cuda_types() t ON t.key = 'e:' || c.fact_id || ':' || c.receiver_id
            AND t.ty = 'device'
          WHERE c.method NOT IN ('get', 'item', 'tolist', 'numpy', 'tobytes', 'cpu',
                                 'copy_to_host', 'synchronize')
        ), reach(fact_id, function_id, framework, hops) USING KEY (fact_id, function_id) AS (
          SELECT DISTINCT fact_id, scope_id, framework, 0 FROM direct WHERE scope_id <> -1
          UNION
          SELECT t.fact_id, c.scope_id, r.framework, r.hops + 1
          FROM reach r
          JOIN cuda_resolved_calls() t ON t.definition_fact = r.fact_id
            AND t.definition_id = r.function_id AND t.kind = 'function'
          JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
          WHERE r.hops < 3 AND c.scope_id <> -1 AND NOT EXISTS (
            SELECT 1 FROM recurring.reach p WHERE p.fact_id = t.fact_id AND p.function_id = c.scope_id)
        )
        SELECT fact_id, path, node_id, scope_id, framework, true AS direct FROM direct
        UNION
        SELECT c.fact_id, c.path, c.node_id, c.scope_id, r.framework, false
        FROM cuda_resolved_calls() t
        JOIN reach r ON r.fact_id = t.definition_fact AND r.function_id = t.definition_id
        JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
        WHERE t.kind = 'function'
    """


class CudaBarriers(CudaRelation):
    """Calls that wait for the device: `synchronize()` on a stream, event or device, the
    framework's device-wide barriers, and calls of project functions that reach one (up to three
    calls deep)."""

    stored = True
    query = """
        WITH RECURSIVE direct(fact_id, path, node_id, scope_id) AS (
          SELECT c.fact_id, c.path, c.node_id, c.scope_id
          FROM cuda_calls() c
          WHERE c.method IN ('synchronize', 'sync', 'block_until_ready', 'wait')
             OR c.qualified_name IN (
                  'torch.cuda.synchronize', 'cupy.cuda.runtime.deviceSynchronize',
                  'cupy.cuda.runtime.streamSynchronize', 'numba.cuda.synchronize',
                  'cuda.core.experimental.Device.sync', 'jax.block_until_ready',
                  'cuda.bindings.runtime.cudaDeviceSynchronize',
                  'cuda.bindings.driver.cuCtxSynchronize',
                  'cuda.bindings.driver.cuStreamSynchronize')
          UNION
          SELECT fact_id, path, node_id, scope_id FROM cuda_host_reads()
        ), reach(fact_id, function_id, hops) USING KEY (fact_id, function_id) AS (
          SELECT DISTINCT fact_id, scope_id, 0 FROM direct WHERE scope_id <> -1
          UNION
          SELECT t.fact_id, c.scope_id, r.hops + 1
          FROM reach r
          JOIN cuda_resolved_calls() t ON t.definition_fact = r.fact_id
            AND t.definition_id = r.function_id AND t.kind = 'function'
          JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
          WHERE r.hops < 3 AND c.scope_id <> -1 AND NOT EXISTS (
            SELECT 1 FROM recurring.reach p WHERE p.fact_id = t.fact_id AND p.function_id = c.scope_id)
        )
        SELECT * FROM direct
        UNION
        SELECT c.fact_id, c.path, c.node_id, c.scope_id
        FROM cuda_resolved_calls() t
        JOIN reach r ON r.fact_id = t.definition_fact AND r.function_id = t.definition_id
        JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
        WHERE t.kind = 'function'
    """


class CudaClocks(CudaRelation):
    """Reads of a wall clock: `time.perf_counter()` and its kin, with the statement holding the
    read and the column that statement starts at (the indentation a line inserted before it
    takes)."""

    stored = True
    query = """
        WITH statements AS (
          SELECT c.fact_id, c.path, c.node_id, c.scope_id, c.start_line, c.start_column,
                 c.end_line, c.end_column, c.shown,
                 (SELECT s.node_id FROM cuda_nodes() s
                  WHERE s.fact_id = c.fact_id AND s.node_id <= c.node_id
                    AND s.node_id + s.descendant_count >= c.node_id
                    AND s.type IN ('expression_statement', 'return_statement', 'assert_statement',
                                   'if_statement', 'while_statement', 'with_statement',
                                   'for_statement')
                  ORDER BY s.node_id DESC LIMIT 1) AS statement_id
          FROM cuda_calls() c
          WHERE c.qualified_name IN ('time.perf_counter', 'time.perf_counter_ns', 'time.time',
                                     'time.time_ns', 'time.monotonic', 'time.monotonic_ns',
                                     'time.process_time', 'timeit.default_timer')
        )
        SELECT s.*, n.start_line AS statement_line, n.start_column AS statement_column
        FROM statements s LEFT JOIN cuda_nodes() n ON n.fact_id = s.fact_id AND n.node_id = s.statement_id
    """


class CudaTimedRegions(CudaRelation):
    """The code between two consecutive clock reads of a function that does device work itself
    (`work_calls` launches, device results and device-method calls): whether a fence comes after
    the last work, including work done by the project functions it calls (`fenced`), and the
    latest work queued before the start that no fence has waited for (`leaked_work`, 0 for none)."""

    stored = True
    query = """
        WITH ordered AS (
          SELECT *, lead(node_id) OVER w AS next_id FROM cuda_clocks()
          WINDOW w AS (PARTITION BY fact_id, scope_id ORDER BY node_id)
        ), pairs AS (
          SELECT a.fact_id, a.path, a.scope_id, a.node_id AS start_id, b.node_id AS stop_id,
                 a.statement_line AS start_statement_line,
                 a.statement_column AS start_statement_column,
                 b.statement_line AS stop_statement_line,
                 b.statement_column AS stop_statement_column,
                 a.start_line, a.start_column, b.end_line, b.end_column
          FROM ordered a JOIN cuda_clocks() b ON b.fact_id = a.fact_id AND b.node_id = a.next_id
        ), inside AS (
          SELECT p.*, count(w.node_id) AS work_calls, max(w.node_id) AS last_work,
                 min(w.framework) AS framework
          FROM pairs p
          LEFT JOIN cuda_device_work() w ON w.fact_id = p.fact_id AND w.scope_id = p.scope_id
            AND w.node_id > p.start_id AND w.node_id < p.stop_id AND w.direct
          GROUP BY ALL
        ), last AS (
          SELECT i.*, coalesce((SELECT max(w.node_id) FROM cuda_device_work() w
                                WHERE w.fact_id = i.fact_id AND w.scope_id = i.scope_id
                                  AND w.node_id > i.start_id AND w.node_id < i.stop_id), 0) AS last_call
          FROM inside i
        )
        SELECT i.* EXCLUDE (last_call),
               EXISTS (SELECT 1 FROM cuda_barriers() b
                       WHERE b.fact_id = i.fact_id AND b.scope_id = i.scope_id
                         AND b.node_id >= i.last_call AND b.node_id < i.stop_id) AS fenced,
               coalesce((SELECT max(w.node_id) FROM cuda_device_work() w
                         WHERE w.fact_id = i.fact_id AND w.scope_id = i.scope_id
                           AND w.node_id < i.start_id
                           AND NOT EXISTS (SELECT 1 FROM cuda_barriers() b
                                           WHERE b.fact_id = w.fact_id AND b.scope_id = w.scope_id
                                             AND b.node_id >= w.node_id AND b.node_id < i.start_id)),
                        0) AS leaked_work
        FROM last i WHERE i.work_calls > 0
    """
