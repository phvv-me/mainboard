"""PY-CUDA rules about the host waiting on the device: reads of device values and timers."""

from typing import ClassVar

from ...catalog import Example, Rule
from .cuda_paths import MEASURING, TESTS


class HostWaitOnDevice(Rule, id="PY-CUDA0001", applicability="review"):
    """A read of a device value into Python that blocks the host until the device has finished,
    repeated per iteration or per call of a loop, or twice in one function: a scalar conversion or
    test that hides it (`int(x[0])`, `if x > 0:`, `float(x)`), a read of one value (`x.item()`),
    or a bulk copy (`x.get()`, `cupy.asnumpy(x)`, only inside a loop). A lone read outside any
    loop is not reported. Identity and membership tests (`x is not None`) read nothing."""

    exempt: tuple[str, ...] = MEASURING

    query = r"""
        WITH reads AS (
          SELECT r.*, (SELECT max(l.loop_id) FROM cuda_loops() l
                       WHERE l.fact_id = r.fact_id AND l.scope_id = r.scope_id
                         AND r.node_id BETWEEN l.first AND l.last) AS loop_id,
                 count(*) FILTER (WHERE r.kind = 'implicit' OR r.api = 'item')
                   OVER (PARTITION BY r.fact_id, r.scope_id) AS reads_here
          FROM cuda_host_reads() r
          WHERE NOT EXISTS (SELECT 1 FROM cuda_device_code() d
                            WHERE d.fact_id = r.fact_id AND d.function_id = r.scope_id)
            AND NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(r.path, p))), false)
        ), placed AS (
          SELECT r.*, h.origin_path, h.origin_line,
                 CASE WHEN r.loop_id IS NOT NULL THEN 'loop'
                      WHEN h.function_id IS NOT NULL THEN 'called'
                      WHEN r.reads_here > 1 AND (r.kind = 'implicit' OR r.api = 'item')
                      THEN 'serial' ELSE 'once' END AS context
          FROM reads r
          LEFT JOIN (SELECT * FROM cuda_hot()
                     WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(origin_path, p))), false)
                     QUALIFY row_number() OVER (PARTITION BY fact_id, function_id
                                                ORDER BY hops, origin_path, origin_line) = 1) h
            ON h.fact_id = r.fact_id AND h.function_id = r.scope_id
        )
        SELECT p.path, p.start_line, p.start_column, p.end_line, p.end_column,
               coalesce(p.shown, p.api || ' ' || syntax_slice(s.content, p.start_line,
                          p.start_column, p.end_line, p.end_column)) AS shown,
               CASE p.context
                 WHEN 'loop' THEN ' on every iteration of its loop'
                 WHEN 'called' THEN ' on every call (the function runs once per iteration of the '
                                    || 'loop at ' || p.origin_path || ':' || p.origin_line || ')'
                 WHEN 'serial' THEN ' (one of ' || p.reads_here || ' reads in this function, '
                                    || 'each its own round trip)'
                 ELSE '' END AS where_text,
               CASE WHEN p.kind = 'iterate'
                    THEN '`' || p.api || '()` walks a device array one element at a time, one '
                         || 'transfer each'
                    WHEN p.kind = 'explicit' THEN '`.' || p.api || '()` copies the array to the host'
                    WHEN p.api IN ('print', 'str', 'repr', 'format')
                    THEN '`' || p.api || '()` formats a device value, which copies it to the host'
                    WHEN p.api = 'format'
                    THEN 'formatting a device value copies it to the host'
                    WHEN p.api IN ('if', 'elif', 'while', 'assert', 'if-else', 'not', 'and/or')
                    THEN 'a Python `' || p.api || '` on a device value copies it to the host to '
                         || 'decide'
                    ELSE '`' || p.api || '()` turns a device value into a Python number by '
                         || 'copying it to the host' END AS how
        FROM placed p JOIN source_files s ON s.path = p.path
        WHERE CASE WHEN p.kind = 'explicit' AND p.api <> 'item' THEN p.context = 'loop'
                   ELSE p.context <> 'once' END
    """
    message = (
        "`{shown}` waits for the device{where_text}: {how}. Each wait is a full host-device round "
        "trip that stalls the launch queue; batch the reads into one transfer, or keep the value "
        "on the device"
    )
    examples = (
        Example(
            bad=(
                "import cupy as cp\n\n\ndef total(batches):\n    out = 0\n"
                "    for batch in batches:\n        x = cp.asarray(batch)\n"
                "        out += int(x[0])\n    return out\n"
            ),
            good="",
        ),
        Example(
            bad=(
                "import cupy as cp\n\n\ndef count_positive(batches):\n    n = 0\n"
                "    for batch in batches:\n        y = cp.asarray(batch)\n"
                "        if y.sum() > 0:\n            n += 1\n    return n\n"
            ),
            good=(
                "import cupy as cp\n\n\ndef count_positive(batches):\n"
                "    sums = cp.stack([cp.asarray(batch).sum() for batch in batches])\n"
                "    return int((sums > 0).sum())\n"
            ),
        ),
        Example(
            bad=(
                "from ..scalars import Vector\n\n\ndef search(values: Vector, key: int) -> int:\n"
                "    middle = 0\n    while values[middle] < key:\n        middle += 1\n"
                "    return middle\n"
            ),
            good=(
                "from ..scalars import Vector\nfrom ..typed import device\n\n\n@device\n"
                "def search(values: Vector, key: int) -> int:\n"
                "    middle = 0\n    while values[middle] < key:\n        middle += 1\n"
                "    return middle\n"
            ),
            path="patos/cuda/primitives/search.py",
            context={
                "pyproject.toml": '[project]\nname = "patos"\nversion = "0"\n',
                "patos/__init__.py": "",
                "patos/cuda/__init__.py": "",
                "patos/cuda/backend/__init__.py": "",
                "patos/cuda/primitives/__init__.py": "",
                "patos/cuda/backend/decorators.py": "def device(function):\n    return function\n",
                "patos/cuda/typed.py": (
                    'from .backend.decorators import device\n\n__all__ = ["device"]\n'
                ),
                "patos/cuda/scalars.py": "import cupy\n\n\nclass Vector:\n    pass\n",
            },
        ),
    )


class TimedRegionWithoutFence(Rule, id="PY-CUDA0002", applicability="safe"):
    """A timer around device work that the device has not finished: no fence between the last
    launch and the stop (the clock measures queueing), or work still queued when the clock
    starts (the region is charged for it)."""

    exempt: tuple[str, ...] = TESTS
    change: ClassVar[str] = "inserted `{fence}` before line {line}, where the {side} clock is read"

    query = r"""
        WITH regions AS (
          SELECT r.*, CASE r.framework WHEN 'patos' THEN 'cupy' ELSE r.framework END AS device
          FROM cuda_timed_regions() r
          WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(r.path, p))), false)
        ), problems AS (
          SELECT fact_id, path, stop_id AS clock_id, 'stop' AS side, work_calls, device,
                 stop_statement_line AS line, stop_statement_column AS col
          FROM regions WHERE NOT fenced
          UNION ALL
          SELECT fact_id, path, start_id, 'start', work_calls, device, start_statement_line,
                 start_statement_column
          FROM regions WHERE leaked_work > 0
        ), aliases AS (
          SELECT fact_id, qualified, min(bound) AS alias FROM cuda_names() GROUP BY ALL
        )
        SELECT c.path, c.start_line, c.start_column, c.end_line, c.end_column, c.shown, p.side,
               p.work_calls, p.line, p.col,
               CASE WHEN p.device = 'cupy' THEN a.alias || '.cuda.runtime.deviceSynchronize()'
                    WHEN p.device = 'torch' THEN a.alias || '.cuda.synchronize()'
                    WHEN p.device = 'numba' THEN a.alias || '.synchronize()' END AS fence,
               CASE p.device WHEN 'cupy' THEN 'cupy.cuda.runtime.deviceSynchronize()'
                             WHEN 'torch' THEN 'torch.cuda.synchronize()'
                             WHEN 'numba' THEN 'numba.cuda.synchronize()'
                             ELSE 'the framework''s device synchronize' END AS fence_name,
               CASE p.side WHEN 'stop'
                    THEN 'stops the clock while the device may still be running ' || p.work_calls
                         || ' call(s) of device work: a launch returns once the work is queued, '
                         || 'so the timer measures queueing, not the work'
                    ELSE 'starts the clock while device work queued earlier may still be running, '
                         || 'so the region is charged for it' END AS problem
        FROM problems p
        JOIN cuda_clocks() c ON c.fact_id = p.fact_id AND c.node_id = p.clock_id
        LEFT JOIN aliases a ON a.fact_id = p.fact_id AND a.qualified = CASE p.device
          WHEN 'cupy' THEN 'cupy' WHEN 'torch' THEN 'torch' WHEN 'numba' THEN 'numba.cuda' END
    """
    message = (
        "`{shown}` {problem}. Wait for the device first with {fence_name} so the region "
        "times the work itself"
    )
    # The fence goes on its own line above the statement that reads the clock, indented like it,
    # when the file imports the framework under a name the line can use.
    fix = """
        SELECT finding_key, path, line AS start_line, 0 AS start_column, line AS end_line,
               0 AS end_column, repeat(' ', col) || fence || chr(10) AS text
        FROM finding WHERE fence IS NOT NULL
    """
    examples = (
        Example(
            bad=(
                "import time\n\nimport cupy as cp\n\n\ndef timed(x):\n"
                "    start = time.perf_counter()\n    y = cp.cumsum(x)\n"
                "    seconds = time.perf_counter() - start\n    return y, seconds\n"
            ),
            good=(
                "import time\n\nimport cupy as cp\n\n\ndef timed(x):\n"
                "    start = time.perf_counter()\n    y = cp.cumsum(x)\n"
                "    cp.cuda.runtime.deviceSynchronize()\n"
                "    seconds = time.perf_counter() - start\n    return y, seconds\n"
            ),
        ),
        Example(
            bad='import time\n\nimport torch\n\n\ndef timed():\n    start = time.perf_counter()\n    y = torch.zeros(1024, device="cuda")\n    seconds = time.perf_counter() - start\n    return y, seconds\n',
            good='import time\n\nimport torch\n\n\ndef timed():\n    start = time.perf_counter()\n    y = torch.zeros(1024, device="cuda")\n    torch.cuda.synchronize()\n    seconds = time.perf_counter() - start\n    return y, seconds\n',
        ),
    )
