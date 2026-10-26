# Reading a project's facts directly: the views every rule reads, queried by hand.

import json
from pathlib import Path

from ....log import logger
from ...engine.facts import Query
from ...engine.lint import Lint, selected


def sql(
    query: str,
    root: Path = Path(),
    *,
    language: str | None = None,
    as_json: bool = False,
    limit: int = 200,
) -> None:
    """Rows of QUERY over ROOT's fact views, tab-separated under a header line.

    Every rule's families are derived, so any view a rule or its fix reads answers: `core_import`,
    `core_def`, `placed_symbols()`, `source_files` and the rest.

    Args:
        query: one DuckDB SELECT over the fact views.
        root: the project whose facts are read.
        language: read through that language's narrowed views (`python`); every fact when unset.
        as_json: one JSON record per row instead.
        limit: the most rows printed.
    """
    root = root.resolve()
    with Lint(root, selected(())) as lint:
        facts = lint.facts(root)
        facts.ensure(Query(sql=query))
        result = facts.cursor(language).execute(query)
        columns = [column for column, *_ in result.description]
        rows = result.fetchmany(limit)
    if as_json:
        logger.result(
            "\n".join(
                json.dumps(dict(zip(columns, row, strict=True)), default=str) for row in rows
            )
        )
        return
    logger.result("\n".join(["\t".join(columns), *("\t".join(map(str, row)) for row in rows)]))
