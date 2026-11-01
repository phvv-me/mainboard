"""What a rule is (see the package's docstring), and what its fix makes."""

import re
from collections.abc import Iterable
from datetime import date
from string import Formatter
from typing import TYPE_CHECKING, ClassVar, Literal

from pathspec import GitIgnoreSpec
from patos import FrozenModel
from pydantic.fields import FieldInfo

from ..constants import LANGUAGES
from ..edits import Edit
from ..engine.facts import Query, Setting
from ..findings import Hit

if TYPE_CHECKING:
    from ..operations.nodes import Rewrite

type Applicability = Literal["safe", "review", "display", "none"]


class Repair(list[Edit]):
    """The edits one finding's fix makes, with where the finding stood and what the fix changed
    (`change`, a line an agent can read to pick up from the fix)."""

    def __init__(self, edits: Iterable[Edit], *, path: str, line: int, change: str) -> None:
        super().__init__(edits)
        self.path, self.line, self.change = path, line, change


class Example(FrozenModel):
    """A snippet the rule must flag (`bad`) and what its fix must turn it into (`good`; None when
    the fix removes the file; equal to `bad` when the fix must refuse and leave the file as it is),
    held in the project file `path` (`lint test` picks one from the rule's language when empty)
    beside the project's other files, `context` (path: text)."""

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
    # Patterns with templates that rewrite the node a finding names, in place of a `fix`.
    rewrites: ClassVar[tuple[Rewrite, ...]] = ()
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
    def fixes(cls) -> Literal["none", "sql", "method", "parts"]:
        """How the rule fixes: SQL over its findings, a `repair` method, or not at all.

        Declared `rewrites` are SQL too (`__init_subclass__` states it). A bundle fixes through its
        parts.
        """
        if cls.fix is not None:
            return "sql"
        return "method" if cls.repair is not Rule.repair else "none"

    @classmethod
    def summary(cls) -> str:
        """The docstring's first line."""
        return (cls.__doc__ or "").strip().splitlines()[0]

    @property
    def fixers(self) -> dict[str, Rule]:
        """The rules whose fix runs, by the table `staged()` fills with their findings."""
        return {"finding_rows": self} if type(self).fixes() != "none" else {}

    @property
    def language(self) -> str | None:
        """The language the id's prefix names (`PY-` Python ...); `ALL-` rules read every one."""
        return LANGUAGES.get(self.id.split("-", 1)[0])

    def __init_subclass__(cls, *, id: str = "", applicability: Applicability = "none") -> None:
        """A subclass without an `id` is a base its family's rules share, not a rule.

        One declaring `rewrites` has them as its fix, so it declares no `fix` beside them. Its
        query yields the `node` each finding rewrites; a finding with none is left alone.
        """
        super().__init_subclass__()
        if id:
            cls.id, cls.applicability = id, applicability
        if cls.__dict__.get("rewrites"):
            if "fix" in cls.__dict__:
                raise TypeError(f"{cls.__name__} declares both `fix` and `rewrites`")
            cls.fix = cls.rewrites[0].fix(cls.rewrites)
            cls.constants = cls.constants | cls.rewrites[0].bindings(cls.rewrites)

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
        or not at all. `found` names a table `staged()` filled, which `finding` then reads instead
        of running the query again.
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

    def staged(self) -> list[Query]:
        """The statements filling `finding_rows` with numbered findings.

        Both `edits(table)` and `changes()` read what they fill.
        """
        return [
            self._bound(
                "CREATE OR REPLACE TEMP TABLE finding_rows AS "
                f"SELECT *, row_number() OVER () AS finding_key FROM ({self.sql()})"
            )
        ]

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


class Part(Rule):
    """A rule with no id of its own, held by a `Bundle`: its settings, query and message as when
    it stood alone, and the gitignore-style lines of the paths it leaves out. It fixes its own
    findings as a rule does, with `fix` or `repair` over its own rows."""

    exclude: tuple[str, ...] = ()

    def branch(self, name: str) -> str:
        """The part's findings as a SELECT of the four finding columns."""
        return (
            f"SELECT path, start_line, start_column, {self._message()} AS message "
            f"FROM ({self.kept(name)})"
        )

    def excluded(self) -> list[dict[str, str | bool]]:
        """The gitignore lines of `exclude` in order, each an RE2 `regex` and its `include`.

        `include` is False for a `!` line.
        """
        return [
            {"regex": pattern.regex.pattern, "include": bool(pattern.include)}
            for pattern in GitIgnoreSpec.from_lines(self.exclude).patterns
            if pattern.regex
        ]

    def kept(self, name: str) -> str:
        """The part's whole finding rows with its excluded paths left out.

        Its parameters take the prefix `name__`, so two parts that set `maximum` do not meet. A
        row is left out as `GitIgnoreSpec.match_file` would: the last matching line decides, except
        that a `!` line matching only through a parent folder (pathspec's group 1, `ps_d`) does not
        overrule a line that matched the file itself.
        """
        own = self.parameters()
        sql = re.sub(r"\$(\w+)", lambda m: f"${name}__{m[1]}" if m[1] in own else m[0], self.sql())
        return (
            f"SELECT * FROM ({sql}) "
            "WHERE NOT list_reduce([{'include': line.include, "
            "'folder': regexp_extract(path, line.regex, 1) = '/'} "
            f"FOR line IN ${name}__excluded::STRUCT(regex VARCHAR, include BOOLEAN)[] "
            "IF regexp_matches(path, line.regex)], "
            "lambda out, hit: CASE WHEN hit.include OR NOT hit.folder OR out.folder "
            "THEN hit ELSE out END, {'include': false, 'folder': true}).include"
        )

    def listed(self, table: str) -> str:
        """The numbered rows of `table` as `finding_rows` holds them: place, message and change."""
        return (
            "SELECT finding_key, path, start_line, start_column, "
            f"{self._message()} AS message, {self._change()} AS change FROM {table}"
        )


class Bundle(Rule):
    """A rule whose findings are those of its parts, one `Part` field each.

    A project configures a part in a table of its own under the bundle's:
    `[tool.mcmr.rules."ID".part_name]` holds that part's settings and its `exclude`. A part that
    declares a fix repairs its own findings, from its own whole rows, and the bundle then fixes
    through its parts.
    """

    @classmethod
    def fields(cls) -> dict[str, FieldInfo]:
        """Each part's settings as `part.setting`."""
        return {
            f"{part}.{name}": field
            for part, owner in cls.model_fields.items()
            for name, field in type(owner.default).model_fields.items()
        }

    @classmethod
    def fixes(cls) -> Literal["none", "sql", "method", "parts"]:
        """`parts` when a part fixes its own findings, else `none`."""
        fixed = any(
            type(field.default).fixes() != "none"
            for field in cls.model_fields.values()
            if isinstance(field.default, Part)
        )
        return "parts" if fixed else "none"

    @property
    def fixers(self) -> dict[str, Rule]:
        """Each part that fixes, by the table `staged()` fills with its findings."""
        return {
            self._table(name): part
            for name, part in self.parts().items()
            if type(part).fixes() != "none"
        }

    @staticmethod
    def _table(name: str) -> str:
        """The table holding the findings of the part `name`."""
        return f"finding_rows__{name}"

    def _change(self) -> str:
        """The `change` column `staged()` fills, each part having stated its own."""
        return "change"

    def parameters(self) -> dict[str, Setting]:
        """Every part's parameters under its prefix, with the patterns of its `exclude`."""
        return {
            f"{name}__{key}": value
            for name, part in self.parts().items()
            for key, value in (part.parameters() | {"excluded": part.excluded()}).items()
        }

    def parts(self) -> dict[str, Part]:
        """The parts by field name."""
        return {name: value for name, value in self if isinstance(value, Part)}

    def sql(self) -> str:
        return "\nUNION ALL\n".join(part.branch(name) for name, part in self.parts().items())

    def staged(self) -> list[Query]:
        """One table of whole rows per part, then `finding_rows` over all of them.

        A finding's key holds its part's ordinal in the high 32 bits and its row number below, so
        no two parts share one.
        """
        parts = self.parts()
        tables = [
            self._bound(
                f"CREATE OR REPLACE TEMP TABLE {self._table(name)} AS SELECT *, "
                f"({ordinal}::BIGINT << 32) + row_number() OVER () AS finding_key "
                f"FROM ({part.kept(name)})"
            )
            for ordinal, (name, part) in enumerate(parts.items())
        ]
        union = " UNION ALL ".join(part.listed(self._table(name)) for name, part in parts.items())
        return [*tables, Query(sql=f"CREATE OR REPLACE TEMP TABLE finding_rows AS {union}")]
