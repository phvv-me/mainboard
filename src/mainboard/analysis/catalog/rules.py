"""What a rule is (see the package's docstring), and what its fix makes."""

import re
from collections.abc import Iterable
from datetime import date
from string import Formatter
from typing import ClassVar, Literal

from patos import FrozenModel

from ..constants import LANGUAGES
from ..edits import Edit
from ..engine.facts import Query, Setting
from ..findings import Hit

type Applicability = Literal["safe", "review", "display", "none"]


class Repair(list[Edit]):
    """The edits one finding's fix makes, with where the finding stood and what the fix changed
    (`change`, a line an agent can read to pick up from the fix)."""

    def __init__(self, edits: Iterable[Edit], *, path: str, line: int, change: str) -> None:
        super().__init__(edits)
        self.path, self.line, self.change = path, line, change


class Example(FrozenModel):
    """A snippet the rule must flag (`bad`) and what its fix must turn it into (`good`; None when
    the fix removes the file), held in the project file `path` (`lint test` picks one from the
    rule's language when empty) beside the project's other files, `context` (path: text)."""

    bad: str
    good: str | None = ""
    path: str = ""
    context: dict[str, str] = {}


class RuleProvenance(FrozenModel):
    """A dated guidance review, separate from the toolchain actually used for analysis."""

    reviewed_on: date
    method: str
    review_context: str
    inspected: tuple[str, ...]
    limits: str


class Rule(FrozenModel):
    """A relation whose rows are findings; its fields are the settings `query` reads as `$name`."""

    id: ClassVar[str]
    applicability: ClassVar[Applicability]
    query: ClassVar[str]
    message: ClassVar[str | None] = None
    fix: ClassVar[str | None] = None
    change: ClassVar[str | None] = None
    examples: ClassVar[tuple[Example, ...]] = ()
    guidance: ClassVar[str] = ""
    references: ClassVar[tuple[str, ...]] = ()
    provenance: ClassVar[RuleProvenance | None] = None
    constants: ClassVar[dict[str, Setting]] = {}
    defined: ClassVar[dict[str, type[Rule]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Setting) -> None:
        if "id" in cls.__dict__:
            Rule.defined[cls.id] = cls

    @classmethod
    def fixes(cls) -> Literal["none", "sql", "method"]:
        """How the rule fixes: SQL over its findings, a `repair` method, or not at all."""
        if cls.fix is not None:
            return "sql"
        return "method" if cls.repair is not Rule.repair else "none"

    @classmethod
    def summary(cls) -> str:
        """The docstring's first line."""
        return (cls.__doc__ or "").strip().splitlines()[0]

    @property
    def language(self) -> str | None:
        """The language the id's prefix names (`PY-` Python ...); `ALL-` rules read every one."""
        return LANGUAGES.get(self.id.split("-", 1)[0])

    def __init_subclass__(cls, *, id: str = "", applicability: Applicability = "none") -> None:
        """A subclass without an `id` is a base its family's rules share, not a rule."""
        super().__init_subclass__()
        if id:
            cls.id, cls.applicability = id, applicability

    def _bound(self, sql: str) -> Query:
        """`sql` with the parameters it names bound."""
        parameters = self.parameters().items()
        return Query(
            sql=sql,
            parameters={
                name: value for name, value in parameters if re.search(rf"\${name}\b", sql)
            },
        )

    def _change(self) -> str:
        """The change template as one SQL expression, a missing value read as empty text."""
        if self.change is None:
            return "CAST(NULL AS VARCHAR)"
        parts = self._parts(self.change)
        return f"concat({', '.join(parts)})" if parts else "''"

    def _message(self) -> str:
        """The template as one SQL expression.

        `{column}` reads the row and `{setting}` is the setting's value.
        """
        if self.message is None:
            return "message"
        return " || ".join(self._parts(self.message)) or "''"

    def _parts(self, template: str) -> list[str]:
        """The SQL expressions whose concatenation is `template`.

        `{column}` reads the row and `{setting}` is the setting's value.
        """
        settings = self.model_dump()

        def quoted(text: str) -> str:
            return "'" + text.replace("'", "''") + "'"

        return [
            part
            for literal, field, _, _ in Formatter().parse(template)
            for part in (
                quoted(literal) if literal else "",
                quoted(str(settings[field]))
                if field in settings
                else f'CAST("{field}" AS VARCHAR)'
                if field
                else "",
            )
            if part
        ]

    def changes(self) -> str:
        """The statement giving each numbered finding's key, place and change text.

        The change text is NULL when the rule states none.
        """
        return (
            f"SELECT finding_key, path, start_line, {self._change()} AS change FROM finding_rows"
        )

    def edits(self, found: str | None = None) -> Query | None:
        """The fix's edit rows over this rule's findings, when the fix is SQL.

        Every `finding` row carries a `finding_key`, and edit rows sharing one are applied together
        or not at all. `found` names a table `finding_rows()` filled, which `finding` then reads
        instead of running the query again.
        """
        if (declared := type(self).fix) is None:
            return None
        fix = declared.strip()
        finding = (
            f"finding AS (SELECT * FROM {found})"
            if found
            else f"finding AS (SELECT *, row_number() OVER () AS finding_key FROM ({self.sql()}))"
        )
        # A fix with its own common table expressions joins `finding` to them.
        opening = re.match(r"(?is)with(\s+recursive)?\s+", fix)
        sql = (
            f"WITH{opening.group(1) or ''} {finding}, {fix[opening.end() :]}"
            if opening
            else f"WITH {finding} {fix}"
        )
        return self._bound(sql)

    def finding_rows(self) -> str:
        """The statement filling the temporary table `finding_rows` with numbered findings.

        Both `edits("finding_rows")` and `changes()` read it.
        """
        return (
            "CREATE OR REPLACE TEMP TABLE finding_rows AS "
            f"SELECT *, row_number() OVER () AS finding_key FROM ({self.sql()})"
        )

    def parameters(self) -> dict[str, Setting]:
        """What the SQL reads as `$name`: the settings and the class's `constants`.

        A rule deriving a value from its settings extends this.
        """
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (self.model_dump() | self.constants).items()
        }

    def repair(self, hit: Hit) -> list[Edit] | None:
        """The edits a Python fix makes for one finding, None when it cannot.

        A rule fixing in Python overrides it.
        """
        return None

    def rows(self) -> Query:
        """The finding query with the parameters it names bound."""
        return self._bound(self.sql())

    def sql(self) -> str:
        """The finding query; a rule composing its SQL from its settings overrides this."""
        return self.query

    def statement(self, *columns: str) -> str:
        """The query the check runs, one row per finding.

        Each row holds the placement and message, then `columns` (SQL expressions over the
        finding's row).
        """
        extra = "".join(f", {column}" for column in columns)
        return (
            f"SELECT path, start_line, start_column, {self._message()} AS message{extra} "
            f"FROM ({self.sql()})"
        )
