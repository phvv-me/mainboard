# Reading a project's facts directly: the views every rule reads, queried by hand.

import json
from pathlib import Path

from ....render import Output, rows
from ...engine.facts import Query
from ...engine.lint import Lint, selected

_COMPACT = Output()


def sql(
    query: str,
    root: Path = Path(),
    *,
    language: str | None = None,
    limit: int = 200,
    output: Output = _COMPACT,
) -> None:
    """Rows of QUERY over ROOT's fact views, under a header line of the selected columns.

    Every rule's families are derived, so any view a rule or its fix reads answers: `core_import`,
    `core_def`, `placed_symbols()`, `source_files` and the rest. Every selected column is
    printed, an empty one too.

    Args:
        query: one DuckDB SELECT over the fact views.
        root: the project whose facts are read.
        language: read through that language's narrowed views (`python`); every fact when unset.
        limit: the most rows printed.
    """
    root = root.resolve()
    with Lint(root, selected(())) as lint:
        facts = lint.facts(root)
        facts.ensure(Query(sql=query))
        result = facts.cursor(language).execute(query)
        columns = [column for column, *_ in result.description]
        fetched = result.fetchmany(limit)
    # A cell JSON cannot hold, a date or a decimal, prints as its text.
    payloads = [
        json.loads(json.dumps(dict(zip(columns, row, strict=True)), default=str))
        for row in fetched
    ]
    rows(payloads, mode=output.mode, fields=output.projection(columns), title="sql")
