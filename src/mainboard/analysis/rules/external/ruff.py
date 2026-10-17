"""Ruff's diagnostics as findings, for the codes of the mcmr rules that Ruff answers.

An mcmr rule that said what one Ruff rule says is dropped, and its codes go in the `[extended]`
table of `left_out.toml`. The lane runs Ruff with the project's own configuration plus
`--extend-select` of exactly those codes (the Ruff tool takes them from the `$extended` parameter
of the rules reading its views), so the project's `ignore`, per-file ignores, exclusions, `# noqa`,
target version and line length decide what is reported. It reports those codes only, and only where
the project's own Ruff run does not (the `ruff` step of the hook already does), as Ruff words them,
with Ruff's fixes: a safe fix is a safe repair, an unsafe one a repair to review. The other tables
of `left_out.toml` record why the remaining codes are not reported.
"""

import tomllib
from functools import cache
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ...catalog import Example, Rule
from ...facts import Setting
from . import wired


@cache
def mainboard_closure() -> tuple[str, ...]:
    """The canonical names of mainboard and of every distribution it requires, directly or
    through another, on this platform. A project among them cannot import `mb`: mainboard
    imports the project."""
    seen: dict[str, None] = {}
    pending = ["mainboard"]
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        seen[name] = None
        try:
            required = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        pending += [
            requirement.name
            for requirement in map(Requirement, required)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""})
        ]
    return tuple(seen)


def _kinds() -> dict[str, dict[str, str]]:
    return tomllib.loads(Path(__file__).with_name("left_out.toml").read_text())


def left_out() -> dict[str, str]:
    """Each code the lane does not report, with its kind and reason."""
    return {
        code: f"{kind}: {reason}"
        for kind, reasons in _kinds().items()
        if kind != "extended"
        for reason, codes in reasons.items()
        for code in codes.split()
    }


def extended() -> tuple[str, ...]:
    """The codes the lane adds to the project's Ruff configuration; none may be left out."""
    codes = tuple(code for codes in _kinds()["extended"].values() for code in codes.split())
    if clash := sorted(set(codes) & set(left_out())):
        raise ValueError(f"Ruff codes both extended and left out: {clash}")
    return codes


_SPAN = """
    d.path = r.path AND (r.start_line, r.start_column)
      BETWEEN (d.start_line, d.start_column) AND (d.end_line, d.end_column)
"""

# Where a Ruff code does not apply, by what the code is about. Device code cannot take
# keyword-only parameters or pass keywords, so the flags, the records built from Booleans and the
# long positional lists it takes are how it is written. A launch `kernel[grid](..., True)` passes
# positionally as the kernel's signature does. Pytest hands a test its parameters by name, so none
# is passed positionally. A parameter that may be a Boolean among other types is a value, not a
# flag. A test draws from the standard generator to check seeding or to make data, never a secret.
_SHAPED_BY_THEIR_CONTEXT = f"""
    NOT (
      r.code IN ('FBT001', 'FBT002', 'FBT003', 'PLR0917')
      AND EXISTS (SELECT 1 FROM cuda_device_code() d WHERE {_SPAN})
    ) AND NOT (
      r.code = 'FBT003'
      AND EXISTS (SELECT 1 FROM cuda_launches() d WHERE {_SPAN})
    ) AND NOT (
      r.code IN ('FBT001', 'FBT002', 'PLR0913', 'PLR0917')
      AND EXISTS (
        SELECT 1 FROM ast_nodes d JOIN ast_files f ON f.fact_id = d.fact_id
        WHERE d.type = 'function_definition' AND starts_with(d.name, 'test_') AND f.path = r.path
          AND (r.start_line, r.start_column)
            BETWEEN (d.start_line, d.start_column) AND (d.end_line, d.end_column)
          AND NOT EXISTS (
            SELECT 1 FROM ast_nodes n
            WHERE n.fact_id = d.fact_id AND n.type = 'function_definition'
              AND n.node_id BETWEEN d.node_id + 1 AND d.node_id + d.descendant_count
              AND (r.start_line, r.start_column)
                BETWEEN (n.start_line, n.start_column) AND (n.end_line, n.end_column)))
    ) AND NOT (
      r.code = 'S311' AND import_is_test_path(r.path)
    ) AND NOT (
      r.code = 'FBT001'
      AND EXISTS (
        SELECT 1 FROM ast_nodes p JOIN ast_files f ON f.fact_id = p.fact_id
        JOIN ast_nodes t ON t.fact_id = p.fact_id AND t.parent_id = p.node_id AND t.type = 'type'
        WHERE f.path = r.path AND p.type IN ('typed_parameter', 'typed_default_parameter')
          AND p.start_line = r.start_line AND p.start_column = r.start_column
          AND NOT regexp_matches(replace(t.name, ' ', ''), '^(bool|None|Optional\\[bool\\]|\\|)+$'))
    )
"""


def _diagnostics(applicability: str) -> str:
    return f"""
        SELECT r.*, CASE WHEN r.code = 'T201' THEN 'rewrote `print` as `logger.info`' END AS change
        FROM ruff_diagnostics r
        WHERE list_contains($extended::VARCHAR[], r.code) AND {applicability}
          AND {_SHAPED_BY_THEIR_CONTEXT}
    """


class RuffDiagnostics(Rule):
    """The base of the Ruff rules: every extended code, worded as Ruff words it. A rule per kind
    of fix keeps what each fix may do in its applicability. Ruff's removal of a `print` is not
    applied: a `print` statement becomes a call of the project logger that logs the same text
    (`print(a, b)` is `logger.info("{} {}", a, b)` with `from mb import logger`). A project that
    mainboard depends on, or is, cannot import `mb`, so there it is the standard library's
    (`logger = logging.getLogger(__name__)`, `logger.info("%s %s", a, b)`). A file that has a
    module-level `logger` keeps it, in the style of its own import; one with another `logger`
    is left alone. A `print` with an `end`, a `sep` or a `file` other than `sys.stderr` stays as
    it is, since only its author can place it; the logger writes to stderr and flushes, so
    `file=sys.stderr` and `flush=True` go."""

    extended: tuple[str, ...] = extended()
    preferred_logger: str = "mb.logger"

    message = "{code}: {message}"
    change = "{change}"
    fix = r"""
        WITH candidates AS (
          SELECT f.finding_key, f.path, c.call_id, c.node_text,
                 c.node_start_line, c.node_start_column, c.node_end_line, c.node_end_column,
                 NOT regexp_matches(s.content, '\blogger\b') AS fresh,
                 CASE
                   WHEN regexp_matches(s.content,
                     '(?m)^logger([ \t]*:[^=\n]+)?[ \t]*=[ \t]*(logging\.)?getLogger\(')
                     THEN '%'
                   WHEN contains(s.content, $logger_import)
                     OR regexp_matches(s.content, '(?m)^from loguru import logger\b') THEN '{}'
                   WHEN NOT regexp_matches(s.content, '\blogger\b')
                     THEN CASE WHEN list_contains($closure::VARCHAR[], f.project)
                               THEN '%' ELSE '{}' END
                 END AS style
          FROM finding f
          JOIN call_calls c ON c.node_path = f.path AND c.qualified_name = 'builtins.print'
           AND c.callee_start_line = f.start_line AND c.callee_start_column = f.start_column
          JOIN source_files s ON s.path = f.path
          JOIN source_lines() first ON first.path = f.path AND first.line = c.node_start_line
          JOIN source_lines() last ON last.path = f.path AND last.line = c.node_end_line
          WHERE f.code = 'T201' AND NOT c.has_starred_arguments
            AND regexp_matches(left(first.text, c.node_start_column), '^[ \t]*$')
            AND strlen(rtrim(last.text)) = c.node_end_column
            AND NOT EXISTS (
              SELECT 1 FROM call_keywords k
              WHERE k.call_id = c.call_id AND NOT (
                (k.name = 'file' AND regexp_matches(c.node_text, '[(,] ?file=sys\.stderr[,)]'))
                OR (k.name = 'flush' AND regexp_matches(c.node_text, '[(,] ?flush=True[,)]'))))
        ), prints AS (
          SELECT * FROM candidates WHERE style IS NOT NULL
        ), elements AS (
          SELECT p.finding_key, p.style, e.call_id, e.root_ordinal, e.text, e.literal_kind,
                 regexp_matches(e.text, '^"[^"\\{}]*"$|^''[^''\\{}]*''$')
                   AND NOT (p.style = '%' AND contains(e.text, '%')) AS is_literal
          FROM prints p
          LEFT JOIN call_expressions e ON e.call_id = p.call_id
           AND e.root_relation = 'argument' AND e.depth = 0
        ), arguments AS (
          SELECT finding_key, count(call_id) AS count, min(text) AS only,
                 coalesce(bool_and(literal_kind = 'string'
                                   AND regexp_matches(text, '^[rRuU]?["'']')
                                   OR regexp_matches(text, '^[fF]("[^"]*"|''[^'']*'')$')),
                          false) AS is_text,
                 coalesce(string_agg(
                   CASE WHEN is_literal THEN substr(text, 2, length(text) - 2)
                        WHEN style = '%' THEN '%s' ELSE '{}' END,
                   ' ' ORDER BY root_ordinal), '') AS template,
                 string_agg(CASE WHEN NOT is_literal THEN text END, ', ' ORDER BY root_ordinal)
                   AS rest
          FROM elements
          GROUP BY finding_key
        ), rewritten AS (
          SELECT p.*,
                 'logger.info(' || CASE
                   WHEN a.count = 0 THEN '""'
                   WHEN a.count = 1 AND a.is_text THEN a.only
                   ELSE '"' || a.template || '"' || coalesce(', ' || a.rest, '') END || ')' AS text
          FROM prints p JOIN arguments a USING (finding_key)
        )
        SELECT finding_key, path, node_start_line AS start_line, node_start_column AS start_column,
               node_end_line AS end_line, node_end_column AS end_column, text
        FROM rewritten
        UNION ALL BY NAME
        SELECT finding_key, path, $logger_import AS "import" FROM rewritten
        WHERE fresh AND style = '{}'
        UNION ALL BY NAME
        SELECT finding_key, path,
               unnest(['import logging', 'logger = logging.getLogger(__name__)']) AS "import"
        FROM rewritten WHERE fresh AND style = '%'
        UNION ALL BY NAME
        SELECT f.finding_key, e.path, e.start_line, e.start_column, e.end_line, e.end_column,
               e.text, e."import"
        FROM finding f JOIN ruff_edits e USING (fact_id)
        WHERE f.fix_applicability IN ('safe', 'unsafe') AND f.code NOT IN ('T201', 'T203')
    """

    def parameters(self) -> dict[str, Setting]:
        module, _, name = self.preferred_logger.rpartition(".")
        return super().parameters() | {
            "logger_import": f"from {module} import {name}",
            "closure": list(mainboard_closure()),
        }


class RuffSafeFix(RuffDiagnostics, id="PY-RUFF0001", applicability="safe"):
    """Ruff diagnostics whose fix Ruff calls safe, one that keeps the code's behavior."""

    query = _diagnostics("fix_applicability = 'safe'")
    examples = (
        Example(
            bad=(
                "def run(value):\n    if value:\n        return 1\n    else:\n        return 2\n"
            ),
            good="def run(value):\n    if value:\n        return 1\n    return 2\n",
        ),
    )


class RuffReview(RuffDiagnostics, id="PY-RUFF0002", applicability="review"):
    """Ruff diagnostics without a safe fix: one with a fix that may change behavior, shown for
    review, and those with none."""

    query = _diagnostics("coalesce(fix_applicability, 'none') <> 'safe'")
    examples = (
        Example(
            bad='"""Module."""\n\n\ndef run(host):\n    print("started", host)\n    return 1\n',
            good=(
                '"""Module."""\nfrom mb import logger\n\n\ndef run(host):\n'
                '    logger.info("started {}", host)\n    return 1\n'
            ),
        ),
        Example(
            bad='import sys\n\n\ndef run():\n    print("started", file=sys.stderr, flush=True)\n',
            good='import sys\nfrom mb import logger\n\n\ndef run():\n    logger.info("started")\n',
        ),
        Example(
            bad='"""Module."""\n\n\ndef run(host):\n    print("started", host)\n    return 1\n',
            good=(
                '"""Module."""\nimport logging\n\nlogger = logging.getLogger(__name__)\n\n\n'
                'def run(host):\n    logger.info("started %s", host)\n    return 1\n'
            ),
            context={"pyproject.toml": '[project]\nname = "patos"\nversion = "0"\n'},
        ),
        Example(
            bad='import logging\n\n\ndef run(host):\n    print("done 100%", host)\n',
            good=(
                "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
                'def run(host):\n    logger.info("%s %s", "done 100%", host)\n'
            ),
            context={"pyproject.toml": '[project]\nname = "mainboard"\nversion = "0"\n'},
        ),
        Example(
            bad=(
                "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
                'def run(host):\n    print("started", host)\n'
            ),
            good=(
                "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
                'def run(host):\n    logger.info("started %s", host)\n'
            ),
            context={"pyproject.toml": '[project]\nname = "lab"\nversion = "0"\n'},
        ),
    )


if not wired():
    for rule in (RuffSafeFix, RuffReview):
        Rule.defined.pop(rule.id)
