"""Rules over how code logs (mcmr's PY-LOGG family)."""

from functools import reduce

from ...catalog import Example, Function, Rule
from ...facts import Setting

_DIRECT_LOGGERS = (
    "logging.Logger",
    "logging.LoggerAdapter",
    "logging.getLogger",
    "logging.debug",
    "logging.info",
    "logging.warning",
    "logging.error",
    "logging.exception",
    "logging.critical",
    "logging.fatal",
    "logging.log",
    "loguru.logger",
    "structlog.get_logger",
)

# printf placeholders (`%s`, `%-8s`, `%05d`, `%.3f`, `%r` ...) as the `str.format` fields mb's
# logger fills from positional arguments; the braces already in the text are doubled first, `%%`
# is set aside and restored.
_PRINTF_FIELDS = (
    (r"%-(\d+)s", r"{:<\1}"),
    (r"%(\d+)s", r"{:>\1}"),
    (r"%\.(\d+)s", r"{:.\1}"),
    (r"%s", "{}"),
    (r"%r", "{!r}"),
    (r"%-(\d+)[di]", r"{:<\1d}"),
    (r"%(0?\d*)i", r"%\1d"),
    (r"%(0?\d*)(\.\d+)?([dfeEgGxXo])", r"{:\1\2\3}"),
)
_FORMATTED = reduce(
    lambda inner, step: f"regexp_replace({inner}, '{step[0]}', '{step[1]}', 'g')",
    _PRINTF_FIELDS,
    "replace(replace(replace(text, '{', '{{'), '}', '}}'), '%%', chr(1))",
)
_PRINTF_SPEC = r"%-?0?\d*(?:\.\d+)?[sridfeEgGxXo]"


class LoggingBracedMessage(Function):
    """A printf-style log message literal (`"wrote %s (%d rows)"`) as the `str.format` message
    mb's logger fills (`"wrote {} ({:d} rows)"`); null when a placeholder has no counterpart."""

    text: str

    expression = (
        f"CASE WHEN contains({_FORMATTED}, '%') THEN NULL "
        f"ELSE replace({_FORMATTED}, chr(1), '%') END"
    )


class LoggerBoundaryBypassCount(Rule, id="PY-LOGG0001", applicability="review"):
    """Direct logger provider calls outside the configured project boundary."""

    preferred_logger: str = "mb.logger"
    direct_logger_symbols: tuple[str, ...] = _DIRECT_LOGGERS

    query = """
        SELECT *, node_path AS path, node_start_line AS start_line,
               node_start_column AS start_column, node_end_line AS end_line,
               node_end_column AS end_column
        FROM call_calls
        WHERE list_contains($direct_logger_symbols::VARCHAR[], qualified_name)
          AND qualified_name <> $preferred_logger
          AND NOT EXISTS (SELECT 1 FROM call_module_bindings b
                          WHERE b.fact_id = call_calls.fact_id AND b.name = $preferred_logger)
    """
    message = "`{qualified_name}` bypasses the project logger `{preferred_logger}`"
    examples = (
        Example(
            bad='import logging\n\nlogger = logging.getLogger(__name__)\nlogger.info("started")\n',
            good='import logging\nfrom mb import logger\n\nlogger.info("started")\n',
        ),
    )
    # Only the shapes that move to the project logger (`from mb import logger`, structlog under
    # loguru's call shape) without changing what is logged:
    # `NAME = logging.getLogger(...)` at module level, when every read of NAME is a level call
    # (`NAME.info(...)`); `logging.getLogger(...).info(...)`; and `logging.info(...)`. Each
    # call is a single message or a printf literal whose placeholders the arguments fill,
    # rewritten as the `{}` fields that logger fills (keywords such as `exc_info` and unmatched
    # placeholders leave the finding alone). Configuring a logger (`setLevel`) is left.
    fix = r"""
        WITH levels(method, target) AS (VALUES
          ('debug', 'debug'), ('info', 'info'), ('warning', 'warning'), ('warn', 'warning'),
          ('error', 'error'), ('exception', 'exception'), ('critical', 'critical'),
          ('fatal', 'critical')
        ), shapes AS (
          SELECT f.finding_key, f.path, f.call_id, f.qualified_name, f.assigned_target,
                 f.node_text, f.node_start_line, f.node_start_column, f.node_end_line,
                 f.node_end_column, f.callee_start_line, f.callee_start_column,
                 f.callee_end_line, f.callee_end_column,
                 CASE WHEN f.qualified_name <> 'logging.getLogger' THEN 'direct'
                      WHEN f.assigned_target = '' THEN 'chained'
                      WHEN l.text = f.assigned_target || ' = ' || f.node_text THEN 'assigned'
                 END AS shape
          FROM finding f
          LEFT JOIN imports_code_lines() l ON l.path = f.path AND l.line = f.node_start_line
          WHERE f.qualified_name = 'logging.getLogger'
             OR f.qualified_name IN ('logging.debug', 'logging.info', 'logging.warning',
                                     'logging.error', 'logging.exception', 'logging.critical',
                                     'logging.fatal')
        ), uses AS (
          SELECT s.finding_key, 'direct' AS shape, s.call_id AS use_id, lv.target AS method,
                 s.callee_start_line AS head_line, s.callee_start_column AS head_column,
                 s.callee_end_line AS head_end_line, s.callee_end_column AS head_end_column
          FROM shapes s JOIN levels lv ON ends_with(s.qualified_name, '.' || lv.method)
          WHERE s.shape = 'direct'
          UNION ALL BY NAME
          SELECT s.finding_key, 'chained' AS shape, u.call_id AS use_id, lv.target AS method,
                 s.node_start_line AS head_line, s.node_start_column AS head_column,
                 s.node_end_line AS head_end_line, s.node_end_column AS head_end_column
          FROM shapes s
          JOIN call_calls u ON u.node_path = s.path AND u.callee_start_line = s.node_start_line
            AND u.callee_start_column = s.node_start_column
          JOIN levels lv ON u.callee_text = s.node_text || '.' || lv.method
          WHERE s.shape = 'chained'
          UNION ALL BY NAME
          SELECT s.finding_key, 'assigned' AS shape, u.call_id AS use_id, lv.target AS method,
                 u.callee_start_line AS head_line, u.callee_start_column AS head_column,
                 u.callee_start_line AS head_end_line,
                 u.callee_start_column + strlen(s.assigned_target) AS head_end_column
          FROM shapes s
          JOIN call_calls u ON u.node_path = s.path
          JOIN levels lv ON u.callee_text = s.assigned_target || '.' || lv.method
          WHERE s.shape = 'assigned'
        ), arguments AS (
          SELECT u.use_id, count(*) AS nargs,
                 max(CASE WHEN e.root_ordinal = 0 THEN e.text END) AS first_text,
                 max(CASE WHEN e.root_ordinal = 0 THEN e.literal_kind END) AS first_kind,
                 max(CASE WHEN e.root_ordinal = 0 THEN e.node_start_line END) AS first_start_line,
                 max(CASE WHEN e.root_ordinal = 0 THEN e.node_start_column END)
                   AS first_start_column,
                 max(CASE WHEN e.root_ordinal = 0 THEN e.node_end_line END) AS first_end_line,
                 max(CASE WHEN e.root_ordinal = 0 THEN e.node_end_column END) AS first_end_column
          FROM uses u
          JOIN call_expressions e
            ON e.call_id = u.use_id AND e.root_relation = 'argument' AND e.depth = 0
          GROUP BY u.use_id
        ), judged AS (
          SELECT u.*, a.nargs, a.first_text, a.first_start_line, a.first_start_column,
                 a.first_end_line, a.first_end_column,
                 NOT c.has_starred_arguments
                   AND NOT EXISTS (SELECT 1 FROM call_keywords k WHERE k.call_id = u.use_id)
                   AND a.nargs >= 1
                   AND (a.nargs = 1 OR (
                     a.first_kind = 'string' AND regexp_matches(a.first_text, '^[rRuU]?["'']')
                     AND NOT regexp_matches(a.first_text, '[^A-Za-z0-9_][fFbB][rR]?["'']')
                     AND len(regexp_extract_all(replace(a.first_text, '%%', ''), '%'))
                         = a.nargs - 1
                     AND len(regexp_extract_all(replace(a.first_text, '%%', ''), $printf_spec))
                         = a.nargs - 1
                     AND logging_braced_message(a.first_text) IS NOT NULL)) AS convertible
          FROM uses u
          JOIN call_calls c ON c.call_id = u.use_id
          LEFT JOIN arguments a ON a.use_id = u.use_id
        ), verdicts AS (
          SELECT s.finding_key, s.shape, s.path, s.assigned_target, s.node_start_line,
                 count(j.use_id) AS uses, coalesce(bool_and(j.convertible), true) AS convertible
          FROM shapes s LEFT JOIN judged j USING (finding_key)
          GROUP BY s.finding_key, s.shape, s.path, s.assigned_target, s.node_start_line
        ), reads AS (
          SELECT sf.path, sym.name, sym.reference_are_references_complete AS complete,
                 count(r.parent_id) AS reads
          FROM symbol_symbols sym
          JOIN symbol_facts sf USING (fact_order, fact_id)
          LEFT JOIN symbol_symbols_reference_references r ON r.parent_id = sym.record_id
          WHERE sym.scope = 'module' AND sym.name IN (SELECT assigned_target FROM shapes)
          GROUP BY sf.path, sym.name, sym.reference_are_references_complete
        ), approved AS (
          SELECT v.finding_key FROM verdicts v
          JOIN source_files c ON c.path = v.path
          LEFT JOIN reads y ON y.path = v.path AND y.name = v.assigned_target
          WHERE v.convertible AND (v.uses >= 1 OR v.shape = 'assigned')
            AND (v.shape <> 'assigned' OR (y.reads = v.uses AND (y.complete
                 OR len(regexp_extract_all(c.content, '\b' || v.assigned_target || '\b'))
                    = v.uses + 1)))
            AND (NOT regexp_matches(c.content, '\blogger\b') OR contains(c.content, $logger_import)
                 OR (v.shape = 'assigned' AND v.assigned_target = $logger_name))
        )
        SELECT j.finding_key, v.path, j.first_start_line AS start_line,
               j.first_start_column AS start_column, j.first_end_line AS end_line,
               j.first_end_column AS end_column, logging_braced_message(j.first_text) AS text
        FROM judged j
        JOIN verdicts v USING (finding_key) SEMI JOIN approved USING (finding_key)
        WHERE j.nargs >= 2
        UNION ALL BY NAME
        SELECT j.finding_key, v.path, j.head_line AS start_line, j.head_column AS start_column,
               j.head_end_line AS end_line, j.head_end_column AS end_column,
               CASE j.shape WHEN 'direct' THEN $logger_name || '.' || j.method
                            ELSE $logger_name END AS text
        FROM judged j
        JOIN verdicts v USING (finding_key) SEMI JOIN approved USING (finding_key)
        WHERE NOT (j.shape = 'assigned' AND v.assigned_target = $logger_name)
        UNION ALL BY NAME
        SELECT v.finding_key, v.path, v.node_start_line AS start_line, 0 AS start_column,
               v.node_start_line + 1 AS end_line, 0 AS end_column, '' AS text
        FROM verdicts v SEMI JOIN approved USING (finding_key)
        WHERE v.shape = 'assigned'
        UNION ALL BY NAME
        SELECT v.finding_key, v.path, $logger_import AS "import"
        FROM verdicts v SEMI JOIN approved USING (finding_key)
        WHERE v.uses >= 1
    """

    def parameters(self) -> dict[str, Setting]:
        module, _, name = self.preferred_logger.rpartition(".")
        return super().parameters() | {
            "logger_name": name,
            "logger_import": f"from {module} import {name}",
            "printf_spec": _PRINTF_SPEC,
        }
