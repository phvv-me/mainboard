"""Rules over SQLAlchemy and SQLModel operations (mcmr's PY-SQLA family)."""

from ...catalog import Example, Function, Relation, Rule


class SqlalchemyOperations(Relation):
    """The `sqlalchemy_operations()` relation."""

    query = """
    SELECT *, node_span_path AS path, node_span_start_line AS start_line,
           node_span_start_column AS start_column, node_span_end_line AS end_line,
           node_span_end_column AS end_column
    FROM query_operations
"""


class SqlalchemyTextEndLine(Function):
    """The line on which `text` ends when it starts on `start_line`."""

    start_line: int
    text: str

    expression = "start_line::BIGINT + length(text) - length(replace(text, chr(10), ''))"


class SqlalchemyTextEndColumn(Function):
    """The column (in bytes, the unit of the kernel's spans) at which `text` ends when it starts
    at `start_column`."""

    start_column: int
    text: str

    expression = (
        r"CASE WHEN contains(text, chr(10)) THEN strlen(regexp_extract(text, '[^\n]*$')) "
        "ELSE start_column::BIGINT + strlen(text) END"
    )


_HERO = """from sqlmodel import Field, Session, SQLModel, select


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)


def load(session: Session, hero_id: int):
"""


def _segment_edit(segment: str, text: str, *, where: str = "TRUE") -> str:
    """The edit rows replacing an operation's `segment` span with `text`, for the findings that
    have the segment and satisfy `where`."""
    return f"""
        SELECT finding_key, {segment}_span_path AS path,
               {segment}_span_start_line AS start_line,
               {segment}_span_start_column AS start_column,
               {segment}_span_end_line AS end_line, {segment}_span_end_column AS end_column,
               '{text}' AS text
        FROM finding WHERE {segment}_id IS NOT NULL AND {where}
    """


_PLAIN_EXECUTE = """
    execute_segment_id IS NOT NULL AND scalars_segment_id IS NOT NULL
    AND selected_expression_count = 1 AND EXISTS (
      SELECT 1 FROM call_calls c
      WHERE c.node_path = execute_segment_span_path
        AND c.callee_end_line = execute_segment_span_end_line
        AND c.callee_end_column = execute_segment_span_end_column
        AND NOT c.has_starred_arguments
        AND NOT EXISTS (SELECT 1 FROM call_keywords k WHERE k.call_id = c.call_id)
        AND (SELECT count(*) FROM call_expressions a WHERE a.call_id = c.call_id
             AND a.root_relation = 'argument' AND a.depth = 0) = 1)
"""


class AsyncSessionExpirationPolicy(Rule, id="PY-SQLA0001", applicability="review"):
    """Require explicit non-expiring SQLAlchemy async session factories."""

    query = """
        SELECT * FROM sqlalchemy_operations()
        WHERE kind = 'async_sessionmaker' AND expire_on_commit AND NOT has_unknown_keywords
    """
    message = (
        "`async_sessionmaker` retains commit expiration and can trigger implicit I/O after commit"
    )
    # The keyword goes right after the last argument, so no line is added or moved; a call that
    # already ends in a comma on its own line keeps it (formatters then explode one per line). A
    # factory with a comment, or one that states `expire_on_commit=True`, is a decision to read.
    fix = r"""
        SELECT finding_key, path, line AS start_line, column_ AS start_column, line AS end_line,
          column_ AS end_column,
          CASE last WHEN '(' THEN 'expire_on_commit=False'
            WHEN ',' THEN ' expire_on_commit=False'
                          || CASE WHEN contains(gap, chr(10)) THEN ',' ELSE '' END
            ELSE ', expire_on_commit=False' END AS text
        FROM (
          SELECT finding_key, path, right(body, 1) AS last, gap,
            sqlalchemy_text_end_line(start_line, body) AS line,
            sqlalchemy_text_end_column(start_column, body) AS column_
          FROM (SELECT *, regexp_replace(node_text, '(?s)\s*\)$', '') AS body,
                       regexp_extract(node_text, '(?s)(\s*)\)$', 1) AS gap
                FROM finding
                WHERE NOT contains(node_text, '#')
                  AND NOT contains(node_text, 'expire_on_commit')))
    """
    examples = (
        Example(
            bad=(
                "from sqlalchemy.ext.asyncio import async_sessionmaker\n\n"
                "sessions = async_sessionmaker(engine)\n"
            ),
            good=(
                "from sqlalchemy.ext.asyncio import async_sessionmaker\n\n"
                "sessions = async_sessionmaker(engine, expire_on_commit=False)\n"
            ),
        ),
    )


class SessionCommitInsideLoop(Rule, id="PY-SQLA0002"):
    """Find SQLAlchemy session commits nested inside loops."""

    query = (
        "SELECT * FROM sqlalchemy_operations() WHERE kind = 'session_commit' AND is_inside_loop"
    )
    message = "Session commit runs inside a loop and opens one transaction per iteration"


class SqlmodelExecuteScalarsApi(Rule, id="PY-SQLA0003", applicability="safe"):
    """Prefer SQLModel `exec` for exact scalar selections."""

    query = """
        SELECT * FROM sqlalchemy_operations()
        WHERE kind = 'execute_scalars' AND framework = 'sqlmodel' AND NOT has_execution_options
    """
    message = "SQLModel `execute(...).scalars()` bypasses the typed `exec` API"
    # `execute(select(X)).scalars()` is `exec(select(X))` for one selected entity and one plain
    # argument: rename the method and drop the `.scalars()` segment together.
    fix = (
        _segment_edit("execute_segment", "exec", where=_PLAIN_EXECUTE)
        + "UNION ALL BY NAME"
        + _segment_edit("scalars_segment", "", where=_PLAIN_EXECUTE)
    )
    examples = (
        Example(
            bad=_HERO + "    return session.execute(select(Hero)).scalars().all()\n",
            good=_HERO + "    return session.exec(select(Hero)).all()\n",
        ),
    )


class SqlmodelRedundantScalars(Rule, id="PY-SQLA0004", applicability="safe"):
    """Remove scalar extraction repeated after SQLModel `exec`."""

    query = """
        SELECT * FROM sqlalchemy_operations()
        WHERE kind = 'exec_scalars' AND framework = 'sqlmodel' AND selected_expression_count = 1
    """
    message = "`scalars()` repeats scalar extraction already performed by SQLModel `exec`"
    fix = _segment_edit("scalars_segment", "")
    examples = (
        Example(
            bad=_HERO + "    return session.exec(select(Hero)).scalars().all()\n",
            good=_HERO + "    return session.exec(select(Hero)).all()\n",
        ),
    )


class SqlmodelPrimaryKeyGet(Rule, id="PY-SQLA0005", applicability="review"):
    """Prefer `Session.get` for provable SQLModel primary-key lookups."""

    query = """
        SELECT * FROM sqlalchemy_operations()
        WHERE kind = 'primary_key_first' AND framework = 'sqlmodel'
          AND selected_expression_count = 1 AND has_primary_key_equality
          AND NOT has_execution_options
    """
    message = "Primary-key selection can use the session identity-map-aware `get` API"
    # `session.exec(select(M).where(M.id == key)).first()` is `session.get(M, key)` except that
    # `get` answers from the identity map without a query when the row is already loaded, so
    # the fix is for review. Only the plain chain over one model, one `==` and a simple key
    # (no operator the comparison could be mistaken for) is rewritten.
    fix = r"""
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               parts.session || '.get(' || parts.model || ', ' || parts.key || ')' AS text
        FROM (
          SELECT *, regexp_extract(node_text,
            '(?s)^([A-Za-z_][\w.]*)\.exec\(\s*select\(\s*([A-Za-z_][\w.]*)\s*\)\s*\.where\('
            || '\s*([A-Za-z_][\w.]*)\.\w+\s*==\s*(.+?)\s*,?\s*\)\s*,?\s*\)\s*\.first\(\s*\)$',
            ['session', 'model', 'owner', 'key']) AS parts
          FROM finding WHERE NOT contains(node_text, '#'))
        WHERE parts.model = parts.owner
          AND NOT regexp_matches(parts.key,
            '==|!=|<|>|:=|\b(and|or|not|if|else|in|is|lambda|await)\b')
    """
    examples = (
        Example(
            bad=_HERO
            + "    return session.exec(select(Hero).where(Hero.id == hero_id)).first()\n",
            good=_HERO + "    return session.get(Hero, hero_id)\n",
        ),
    )
