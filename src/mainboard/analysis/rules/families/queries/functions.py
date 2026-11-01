"""The SQL functions the QueryFact relations call on the text of a syntax tree."""

from ....catalog import Function


class QueryDotted(Function):
    """The text when it is a plain dotted name (`session`, `self.db`), NULL otherwise."""

    text: str

    expression = r"CASE WHEN regexp_matches(text, '^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$') THEN text END"


class QueryResolved(Function):
    """A written name with its head replaced by the origin the file's imports give it."""

    origin: str
    written: str

    expression = r"""
        CASE WHEN origin IS NULL THEN written WHEN NOT contains(written, '.') THEN origin
             ELSE origin || '.' || substr(written, instr(written, '.') + 1) END
    """


class QuerySession(Function):
    """Whether a resolved name is `sqlmodel`'s `Session` or `AsyncSession`."""

    resolved: str

    expression = r"""
        coalesce(starts_with(resolved, 'sqlmodel.')
                 AND regexp_extract(resolved, '[^.]*$') IN ('Session', 'AsyncSession'), false)
    """
