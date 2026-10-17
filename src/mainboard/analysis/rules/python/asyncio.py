"""Rules over asyncio use (mcmr's PY-ASYN family)."""

from ...catalog import Example, Relation, Rule
from ...facts import Setting


class AsyncioCallSites(Relation):
    """The `asyncio_call_sites()` relation."""

    query = """
    SELECT * REPLACE (node_path AS path), node_start_line AS start_line,
           node_start_column AS start_column
    FROM call_calls
"""


_POLICY_APIS = [
    f"asyncio.{name}"
    for name in (
        "get_event_loop_policy",
        "set_event_loop_policy",
        "AbstractEventLoopPolicy",
        "DefaultEventLoopPolicy",
        "WindowsSelectorEventLoopPolicy",
        "WindowsProactorEventLoopPolicy",
    )
]


class AsyncioRunBoundary(Rule, id="PY-ASYN0001"):
    """`asyncio.run` calls, each a synchronous event-loop boundary."""

    query = """
        SELECT path, start_line, start_column FROM core_call WHERE qualified_name = 'asyncio.run'
    """
    message = "`asyncio.run` creates a synchronous event-loop boundary"


class TaskGroupCandidate(Rule, id="PY-ASYN0002"):
    """Async functions pairing created tasks with a `gather` that raises on the first error."""

    query = """
            SELECT * FROM function_functions
            WHERE is_async AND created_task_count > 0 AND gather_consumes_created_tasks
              AND NOT gather_returns_exceptions AND NOT has_task_group
        """
    message = "`{fact_id}` satisfies task group candidate"


class DeprecatedEventLoopPolicyUsage(Rule, id="PY-ASYN0003"):
    """Event-loop policy APIs deprecated in Python 3.14."""

    python_minor: int = 14

    query = """
            SELECT path, start_line, start_column, qualified_name FROM core_call
            WHERE qualified_name IN (SELECT unnest($apis::VARCHAR[])) AND $python_minor >= 14
        """
    message = "`{qualified_name}` is an event-loop policy API deprecated in Python 3.14"

    def parameters(self) -> dict[str, Setting]:
        apis = _POLICY_APIS
        return super().parameters() | {"apis": apis}


class DefaultExecutorToThreadCandidate(Rule, id="PY-ASYN0004", applicability="review"):
    """Default-executor calls on a loop variable that can usually use `asyncio.to_thread`."""

    python_minor: int = 14

    query = """
        WITH executors AS (
          SELECT * FROM asyncio_call_sites()
          WHERE ends_with(qualified_name, '.run_in_executor') AND $python_minor >= 9
        ), shape AS (
          SELECT call_id, count(*) FILTER (WHERE root_relation = 'argument') AS arguments,
                 bool_or(root_ordinal = 0 AND text = 'None')
                   FILTER (WHERE root_relation = 'argument') AS none_first,
                 any_value(text) FILTER (WHERE root_relation = 'receiver') AS receiver
          FROM call_expressions
          WHERE depth = 0 AND call_id IN (SELECT call_id FROM executors)
          GROUP BY call_id
        )
        SELECT x.path, x.start_line, x.start_column, x.qualified_name, x.node_text
        FROM executors x JOIN shape s USING (call_id)
        WHERE s.arguments >= 2 AND s.none_first AND EXISTS (
          SELECT 1 FROM call_calls l
          WHERE l.fact_id = x.fact_id AND l.assigned_target = s.receiver
            AND l.assigned_target <> ''
            AND l.qualified_name IN ('asyncio.get_running_loop', 'asyncio.get_event_loop'))
        """
    message = (
        "`{qualified_name}` selects the default executor and can usually become "
        "`asyncio.to_thread`"
    )
    # `loop.run_in_executor(None, f, *a)` is `asyncio.to_thread(f, *a)`: the receiver, the `None`
    # and its comma go (with the blanks after it when an argument follows on the same line), the
    # rest of the call stays as written. A call with a comment before the `None` is left alone.
    # Review: `to_thread` returns a coroutine, not a future, so a caller that cancels the future
    # or adds callbacks to it needs a task.
    fix = r"""
        SELECT finding_key, path, start_line, start_column,
               start_line + len(lines) - 1 AS end_line,
               if(len(lines) = 1, start_column + strlen(opening), strlen(lines[-1]))
                 AS end_column,
               'asyncio.to_thread(' AS text, 'import asyncio' AS "import"
        FROM (
          SELECT *, string_split(opening, chr(10)) AS lines
          FROM (
            SELECT *, head || if(regexp_matches(substr(node_text, length(head) + 1),
                                                '^[ \t]+[^\s#]'),
                                 regexp_extract(substr(node_text, length(head) + 1), '^[ \t]+'),
                                 '') AS opening
            FROM (SELECT *, regexp_extract(node_text, '^[\w.]+\.run_in_executor\(\s*None\s*,')
                              AS head
                  FROM finding))
          WHERE head <> '')
    """
    examples = (
        Example(
            bad=(
                "import asyncio\n\n\nasync def read(path):\n"
                "    loop = asyncio.get_running_loop()\n"
                "    return await loop.run_in_executor(None, open, path)\n"
            ),
            good=(
                "import asyncio\n\n\nasync def read(path):\n"
                "    loop = asyncio.get_running_loop()\n"
                "    return await asyncio.to_thread(open, path)\n"
            ),
        ),
    )


class DeprecatedAsyncioCoroutineFunctionCheck(Rule, id="PY-ASYN0005", applicability="safe"):
    """Calls of `asyncio.iscoroutinefunction`, deprecated in favor of the `inspect` function."""

    python_minor: int = 14

    query = """
            SELECT * FROM asyncio_call_sites()
            WHERE qualified_name = 'asyncio.iscoroutinefunction' AND $python_minor >= 14
        """
    message = (
        "`asyncio.iscoroutinefunction` is deprecated in favor of `inspect.iscoroutinefunction`"
    )
    # The attribute form `asyncio.iscoroutinefunction` becomes `inspect.iscoroutinefunction`
    # (importing `inspect`); a bare name from `from asyncio import iscoroutinefunction` changes
    # that statement into the `inspect` one when it imports nothing else (once per statement),
    # and is left alone otherwise.
    fix = r"""
        SELECT finding_key, path, callee_start_line AS start_line,
               callee_start_column AS start_column, callee_end_line AS end_line,
               callee_end_column AS end_column, 'inspect.iscoroutinefunction' AS text,
               'import inspect' AS "import"
        FROM finding WHERE contains(callee_text, '.')
        UNION ALL BY NAME
        SELECT f.finding_key, d.path, d.start_line, d.start_column, d.end_line, d.end_column,
               regexp_replace(d.text, '^from\s+asyncio\s+import', 'from inspect import') AS text
        FROM finding f
        JOIN import_binding_facts b ON b.path = f.path AND b.name = f.callee_text
          AND b.module = 'asyncio' AND b.imported_name = 'iscoroutinefunction'
          AND b.is_sole_binding
        JOIN import_binding_nodes d ON d.fact_id = b.fact_id AND d.role = 'declaration'
        QUALIFY row_number() OVER (PARTITION BY b.fact_id
                                   ORDER BY f.start_line, f.start_column) = 1
    """
    examples = (
        Example(
            bad=(
                "import asyncio\n\n\ndef run(function):\n"
                "    if asyncio.iscoroutinefunction(function):\n"
                "        return asyncio.run(function())\n    return function()\n"
            ),
            good=(
                "import asyncio\nimport inspect\n\n\ndef run(function):\n"
                "    if inspect.iscoroutinefunction(function):\n"
                "        return asyncio.run(function())\n    return function()\n"
            ),
        ),
    )
