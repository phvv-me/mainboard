"""Rules over how a subclass overrides the members of its base (mcmr's ALL-OVER family)."""

import ast

from ... import edits
from ...catalog import Example, Relation, Rule
from ...edits import Edit
from ...findings import Hit, Span

_PLACEHOLDER = r"^(?:_+$|_[a-zA-Z0-9_]*[a-zA-Z0-9]$|dummy|ignored_|unused_)"
_MANGLED = "starts_with(name, '__') AND NOT ends_with(name, '__')"


def _tail(dotted: str) -> str:
    """The last dotted segment of `dotted`, SQL: `a.b.c` reads `c`."""
    return rf"regexp_replace({dotted}, '^.*\.', '')"


_ABSTRACT = "('abstractmethod', 'abstractproperty', 'abstractclassmethod', 'abstractstaticmethod')"
_FLAGS = rf"""flags AS (
      SELECT fact_id, parent_id, split_part(relation, '.', 1) AS side,
             bool_or(tail = 'classmethod') AS is_classmethod,
             bool_or(ends_with(string_value, '.setter')) AS is_setter,
             bool_or(tail IN ('property', 'cached_property')
                     OR regexp_matches(string_value, '\.(setter|getter|deleter)$'))
               AS is_read_as_data,
             bool_or(tail = 'final') AS is_final
      FROM (SELECT *, {_tail("string_value")} AS tail FROM override_values
            WHERE relation IN ('declared.decorators', 'inherited.decorators'))
      GROUP BY ALL)"""


class OverridesPairs(Relation):
    """The `overrides_pairs()` relation."""

    query = f"""
    WITH {_FLAGS}
    SELECT fact_id, name, i.record_id AS inherited_record_id, d.record_id AS declared_record_id,
           i.parameters_present AS inherited_callable, d.parameters_present AS declared_callable,
           i.asynchronous AS inherited_asynchronous, d.asynchronous AS declared_asynchronous,
           coalesce(fi.is_classmethod, false) AS inherited_is_classmethod,
           coalesce(fd.is_classmethod, false) AS declared_is_classmethod,
           coalesce(fd.is_setter, false) AS declared_is_setter,
           coalesce(fi.is_read_as_data, false) AS inherited_is_read_as_data,
           coalesce(fd.is_read_as_data, false) AS declared_is_read_as_data,
           coalesce(fi.is_final, false) AS inherited_is_final
    FROM override_inherited i JOIN override_declared d USING (fact_id, name)
    LEFT JOIN flags fi
      ON fi.fact_id = i.fact_id AND fi.parent_id = i.record_id AND fi.side = 'inherited'
    LEFT JOIN flags fd
      ON fd.fact_id = d.fact_id AND fd.parent_id = d.record_id AND fd.side = 'declared'
"""


class OverridesCallablePairs(Relation):
    """The `overrides_callable_pairs()` relation: each overridden method both classes declare with
    parameters, special methods, mangled names and property setters aside."""

    query = """
    SELECT fact_id, name AS method, inherited_record_id AS inherited,
           declared_record_id AS declared, inherited_is_classmethod, declared_is_classmethod
    FROM overrides_pairs()
    WHERE inherited_callable AND declared_callable AND NOT starts_with(name, '__')
      AND NOT declared_is_setter
"""


class OverridesNumbered(Relation):
    """The `overrides_numbered()` relation: both sides' parameters of each pair, stacked and
    numbered within their kind."""

    query = """
    WITH stacked AS (
      SELECT p.fact_id, p.method, p.inherited, p.declared, 'inherited' AS side,
             p.inherited_is_classmethod AS is_classmethod, q.ordinal, q.name, q.kind, q.has_default
      FROM overrides_callable_pairs() p JOIN override_inherited_parameters q
        ON q.fact_id = p.fact_id AND q.parent_id = p.inherited
      UNION ALL
      SELECT p.fact_id, p.method, p.inherited, p.declared, 'declared', p.declared_is_classmethod,
             q.ordinal, q.name, q.kind, q.has_default
      FROM overrides_callable_pairs() p JOIN override_declared_parameters q
        ON q.fact_id = p.fact_id AND q.parent_id = p.declared
    )
    SELECT *, row_number() OVER (PARTITION BY fact_id, inherited, declared, side, kind
                                 ORDER BY ordinal) AS position
    FROM stacked
"""


class OverridesSlots(Relation):
    """The `overrides_slots()` relation: per pair and parameter slot (a position within a kind, or
    a keyword name), what the base held there and what the subclass answers, whether the slot is
    unmatched and whether the subclass binds it under another name."""

    query = rf"""
    WITH slots AS (
      SELECT *, side = 'declared' AS is_declared,
             bool_or(side = 'declared') OVER (
               PARTITION BY fact_id, inherited, declared, kind, name) AS answered_by_name,
             bool_or(side = 'declared' AND kind = 'var_positional') OVER (
               PARTITION BY fact_id, inherited, declared) AS has_varargs,
             bool_or(side = 'declared' AND kind = 'var_keyword') OVER (
               PARTITION BY fact_id, inherited, declared) AS has_kwargs
      FROM overrides_numbered()
      WHERE NOT (kind = 'positional_or_keyword' AND is_classmethod AND position = 1)
    ), held AS (
      SELECT *,
             CASE WHEN kind = 'keyword_only' THEN name ELSE CAST(position AS VARCHAR) END AS slot,
             NOT is_declared AND NOT coalesce(
               CASE WHEN kind IN ('positional_only', 'positional_or_keyword') THEN has_varargs
                    WHEN kind = 'keyword_only' THEN has_kwargs ELSE false END
               AND NOT answered_by_name, false) AS held
      FROM slots
    ), compared AS (
      SELECT fact_id, method, inherited, declared, kind, slot, bool_or(held) AS held,
             bool_or(is_declared) AS answered,
             bool_or(is_declared AND has_default) AS answer_default,
             bool_or(NOT is_declared AND has_default) AS held_default,
             max(CASE WHEN held THEN name END) AS held_name,
             max(CASE WHEN is_declared THEN name END) AS answered_name
      FROM held GROUP BY ALL
    )
    SELECT *, (held AND NOT answered) OR (NOT held AND answered AND NOT answer_default)
                AS unmatched,
           kind = 'positional_or_keyword' AND held AND answered AND held_name <> answered_name
             AND NOT regexp_matches(held_name, '{_PLACEHOLDER}')
             AND NOT regexp_matches(answered_name, '{_PLACEHOLDER}') AS renamed
    FROM compared
"""


class OverridesSignatures(Relation):
    """The `overrides_signatures()` relation."""

    query = """
    WITH optional AS (
      SELECT fact_id, inherited, declared,
             count(*) FILTER (WHERE side = 'inherited' AND has_default) AS inherited_optional,
             count(*) FILTER (WHERE side = 'declared' AND has_default) AS declared_optional
      FROM overrides_numbered() WHERE kind IN ('positional_only', 'positional_or_keyword')
      GROUP BY ALL
    ), measured AS (
      SELECT fact_id, inherited, declared,
        bool_or(unmatched AND kind = 'positional_or_keyword') AS positions_changed,
        bool_or(unmatched AND kind = 'positional_only') AS slots_changed,
        bool_or(unmatched AND kind = 'keyword_only') AS keywords_changed,
        bool_or(kind IN ('var_positional', 'var_keyword') AND held AND NOT answered)
          AS variadics_removed,
        count(*) FILTER (WHERE renamed) AS renamed_total,
        bool_or(kind = 'var_positional' AND answered) AS declared_varargs
      FROM overrides_slots()
      GROUP BY ALL
    )
    SELECT fact_id, method, inherited, declared, differing_arguments, renamed_parameters,
           CAST(differing_arguments = 0 AND renamed_parameters = 0 AND NOT declared_varargs
                AND declared_optional < inherited_optional AS BIGINT)
             AS required_what_the_base_defaulted
    FROM (SELECT p.fact_id, p.method, p.inherited, p.declared,
                 coalesce(CAST(m.positions_changed OR m.slots_changed OR m.keywords_changed
                               AS BIGINT) + CAST(m.variadics_removed AS BIGINT), 0)
                   AS differing_arguments,
                 CASE WHEN m.positions_changed THEN 0 ELSE coalesce(m.renamed_total, 0) END
                   AS renamed_parameters,
                 coalesce(m.declared_varargs, false) AS declared_varargs,
                 coalesce(o.inherited_optional, 0) AS inherited_optional,
                 coalesce(o.declared_optional, 0) AS declared_optional
          FROM overrides_callable_pairs() p
          LEFT JOIN measured m USING (fact_id, inherited, declared)
          LEFT JOIN optional o USING (fact_id, inherited, declared))
"""


class OverridesRenames(Relation):
    """The `overrides_renames()` relation: each parameter an override binds under another name
    than the base used, in a pair whose positions otherwise line up."""

    query = """
    SELECT fact_id, method, held_name AS base_name, answered_name AS own_name
    FROM (SELECT *, bool_or(unmatched AND kind = 'positional_or_keyword')
                      OVER (PARTITION BY fact_id, inherited, declared) AS positions_changed
          FROM overrides_slots())
    WHERE renamed AND NOT positions_changed
"""


def _debt(selected: str, measure: str = "count(*)") -> str:
    """The facts whose `selected` rows (or their `measure`) add up past zero."""
    return f"""
        SELECT f.path, f.start_line, f.start_column, f.fact_id, v.value
        FROM override_facts f
        JOIN (SELECT fact_id, {measure} AS value FROM ({selected}) GROUP BY fact_id) v
          USING (fact_id)
        WHERE v.value > 0
    """


def _members(pairs: str, fields: str) -> str:
    """Per fact, the subclass methods of `pairs` (a relation of `fact_id, method, ...`) as one
    list of structs: the method's first line, column and source, and the `fields` the fix reads."""
    return f"""
        SELECT r.fact_id, list({{'method': r.method, 'line': m.start_line,
                                'column': m.start_column, 'source': m.source,
                                {fields}}}) AS members
        FROM ({pairs}) r
        JOIN override_facts f USING (fact_id)
        JOIN class_classes c ON c.path = f.path AND c.start_line = f.start_line
        JOIN class_methods m ON m.class_id = c.class_id AND m.name = r.method
        GROUP BY r.fact_id
    """


class _Method:
    """A method parsed where it sits in its file, for edits at the places `ast` reports."""

    def __init__(self, path: str, member: dict[str, int | str]) -> None:
        """member: a `_members` struct."""
        self.path, self.line = path, int(member["line"])
        self.text = " " * int(member["column"]) + str(member["source"])
        self.lines = self.text.splitlines()
        self.module = "class _:\n" + self.text
        wrapper = ast.parse(self.module).body[0]
        function = wrapper.body[0] if isinstance(wrapper, ast.ClassDef) else wrapper
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            raise TypeError("a method is a function")
        self.function = function

    def at(self, line: int, offset: int) -> Span:
        """The empty span at an `ast` position, whose columns count UTF-8 bytes as the file's do
        (source line one is `class _:`)."""
        return Span(
            path=self.path,
            start_line=self.line + line - 2,
            start_column=offset,
            end_line=self.line + line - 2,
            end_column=offset,
        )

    def span(self, node: ast.expr | ast.arg) -> Span:
        """The span of one `ast` node."""
        first, last = (
            self.at(node.lineno, node.col_offset),
            self.at(node.end_lineno or node.lineno, node.end_col_offset or node.col_offset),
        )
        return first.model_copy(
            update={"end_line": last.start_line, "end_column": last.start_column}
        )

    def token(self, node: ast.expr | ast.arg, length: int) -> Span:
        """The span of the `length` characters a node opens with (a parameter's name, not its
        annotation)."""
        first = self.at(node.lineno, node.col_offset)
        return first.model_copy(update={"end_column": first.start_column + length})

    def arguments(self) -> list[ast.arg]:
        """Every named parameter in declaration order."""
        arguments = self.function.args
        return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]


def _names(function: ast.AST) -> set[str]:
    """Every name `function` reads, binds or declares anywhere, parameters included."""
    return {
        *(node.id for node in ast.walk(function) if isinstance(node, ast.Name)),
        *(node.arg for node in ast.walk(function) if isinstance(node, ast.arg)),
        *(
            name
            for node in ast.walk(function)
            if isinstance(node, ast.Global | ast.Nonlocal)
            for name in node.names
        ),
    }


class InheritedAttributeHidesAMethod(Rule, id="ALL-OVER0010"):
    """Methods a subclass writes under a name an ancestor already binds to data."""

    query = _debt(
        f"""SELECT fact_id FROM overrides_pairs()
            WHERE NOT inherited_callable AND declared_callable AND NOT ({_MANGLED})""",
    )
    message = "inherited attribute hides a method is {value} for `{fact_id}`"


def _super_call(path: str, member: dict[str, int | str]) -> list[Edit] | None:
    """`super().__init__()` as a line of its own before the first statement of an initializer
    (after its docstring); None when that statement shares a line with the signature."""
    method = _Method(path, member)
    body = method.function.body
    has_docstring = (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )
    if has_docstring and len(body) == 1:
        line, indent = (body[0].end_lineno or body[0].lineno) + 1, body[0].col_offset
    elif method.lines[body[has_docstring].lineno - 2][: body[has_docstring].col_offset].strip():
        return None
    else:
        line, indent = body[has_docstring].lineno, body[has_docstring].col_offset
    start = Span(
        path=path,
        start_line=method.line + line - 2,
        start_column=0,
        end_line=method.line + line - 2,
        end_column=0,
    )
    return [edits.replace(start, " " * indent + "super().__init__()\n")]


class SubclassInitializerSkipsItsBase(Rule, id="ALL-OVER0006", applicability="review"):
    """Subclasses that write their own initializer and never run the one above them."""

    query = f"""
        SELECT d.*, r.members
        FROM ({
        _debt(
            rf'''SELECT fact_id FROM override_facts f
             WHERE depth = 1
               AND EXISTS (SELECT 1 FROM override_inherited i
                           WHERE i.fact_id = f.fact_id AND i.name = '__init__'
                             AND NOT EXISTS (SELECT 1 FROM override_values v
                                             WHERE v.relation = 'inherited.decorators'
                                               AND v.fact_id = i.fact_id
                                               AND v.parent_id = i.record_id
                                               AND {_tail("string_value")} IN {_ABSTRACT}))
               AND EXISTS (SELECT 1 FROM override_declared d
                           WHERE d.fact_id = f.fact_id AND d.name = '__init__')
               AND NOT EXISTS (SELECT 1 FROM override_values
                               WHERE relation = 'initializer_calls' AND fact_id = f.fact_id
                                 AND string_value IN ('super', {_tail("f.base")}))''',
        )
    }) d
        LEFT JOIN ({
        _members(
            '''SELECT f.fact_id, '__init__' AS method,
                      NOT EXISTS (
                        SELECT 1 FROM override_inherited i
                        JOIN override_inherited_parameters q
                          ON q.fact_id = i.fact_id AND q.parent_id = i.record_id
                        WHERE i.fact_id = f.fact_id AND i.name = '__init__' AND q.ordinal > 0
                          AND q.kind IN ('positional_only', 'positional_or_keyword',
                                         'keyword_only')
                          AND NOT q.has_default) AS bare
               FROM override_facts f''',
            "'bare': r.bare",
        )
    }) r USING (fact_id)
    """
    message = "subclass initializer skips its base is {value} for `{fact_id}`"
    examples = (
        Example(
            bad=(
                "class Base:\n"
                "    def __init__(self):\n"
                "        self.items = []\n"
                "\n"
                "\n"
                "class Child(Base):\n"
                "    def __init__(self):\n"
                "        self.count = 0\n"
            ),
            good=(
                "class Base:\n"
                "    def __init__(self):\n"
                "        self.items = []\n"
                "\n"
                "\n"
                "class Child(Base):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.count = 0\n"
            ),
        ),
    )

    def repair(self, hit: Hit) -> list[Edit] | None:
        """The base initializer runs first, when it asks for no argument the subclass would have
        to invent."""
        members = [member for member in hit.records("members") if member["bare"]]
        made = [_super_call(str(hit.path), member) for member in members]
        return (
            None if not made or None in made else [edit for part in made if part for edit in part]
        )


class InitializerCalledOnAStranger(Rule, id="ALL-OVER0007"):
    """Initializers a subclass runs on a class it does not actually inherit from."""

    query = _debt(
        rf"""SELECT c.fact_id
             FROM override_values c JOIN override_facts f USING (fact_id)
             WHERE c.relation = 'initializer_calls' AND f.depth = 1 AND c.string_value <> 'super'
               AND NOT EXISTS (SELECT 1 FROM override_values b
                               WHERE b.relation = 'base_names' AND b.fact_id = c.fact_id
                                 AND b.string_value = c.string_value)
               AND coalesce((SELECT string_value FROM override_values b
                             WHERE b.relation = 'base_names' AND b.fact_id = c.fact_id
                               AND b.ordinal = 0), '') = {_tail("f.base")}""",
    )
    message = "initializer called on a stranger is {value} for `{fact_id}`"


class OverridingMethodChangesItsCallProtocol(Rule, id="ALL-OVER0004"):
    """Overrides that change how a caller has to reach the member, not just what it does."""

    query = _debt(
        f"""SELECT fact_id FROM overrides_pairs()
            WHERE inherited_callable AND declared_callable AND NOT ({_MANGLED})
              AND (inherited_is_read_as_data <> declared_is_read_as_data
                   OR inherited_asynchronous <> declared_asynchronous)""",
    )
    message = "overriding method changes its call protocol is {value} for `{fact_id}`"


class AbstractMemberLeftUnimplemented(Rule, id="ALL-OVER0005"):
    """Promises a base made that the concrete subclass below it never kept."""

    abstract_bases: tuple[str, ...] = ("ABC", "ABCMeta", "Protocol")

    query = _debt(
        f"""WITH promised AS (
              SELECT fact_id, parent_id, split_part(relation, '.', 1) AS side
              FROM override_values
              WHERE relation IN ('declared.decorators', 'inherited.decorators')
                AND {_tail("string_value")} IN {_ABSTRACT}
            ), exempt AS (
              SELECT fact_id FROM override_values
              WHERE relation = 'ancestor_names'
                AND list_contains($abstract_bases::VARCHAR[], string_value)
              UNION
              SELECT fact_id FROM promised WHERE side = 'declared')
            SELECT i.fact_id FROM override_inherited i
            WHERE i.fact_id NOT IN (SELECT fact_id FROM exempt)
              AND EXISTS (SELECT 1 FROM promised p WHERE p.side = 'inherited'
                          AND p.fact_id = i.fact_id AND p.parent_id = i.record_id)
              AND NOT EXISTS (SELECT 1 FROM override_declared d
                              WHERE d.fact_id = i.fact_id AND d.name = i.name)""",
    )
    message = "abstract member left unimplemented is {value} for `{fact_id}`"


class OverridingMethodAcceptsDifferentArguments(Rule, id="ALL-OVER0001"):
    """Overriding methods that stopped accepting what the method they replace accepts."""

    query = _debt(
        "SELECT fact_id, differing_arguments FROM overrides_signatures()",
        "sum(differing_arguments)",
    )
    message = "overriding method accepts different arguments is {value} for `{fact_id}`"


def _rename(path: str, member: dict[str, int | str]) -> list[Edit] | None:
    """A method's parameter `old` renamed `new` in its signature and body; None when the new name
    is already used there or a nested scope binds the old one (the reads there may be its own)."""
    method = _Method(path, member)
    old, new = str(member["old"]), str(member["new"])
    parameter = next((a for a in method.arguments() if a.arg == old), None)
    body = [node for statement in method.function.body for node in ast.walk(statement)]
    scopes = [
        node
        for node in body
        if isinstance(node, ast.Lambda | ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    bound = [
        *(a.arg for scope in scopes for a in ast.walk(scope.args) if isinstance(a, ast.arg)),
        *(
            name.id
            for node in body
            if isinstance(node, ast.comprehension)
            for name in ast.walk(node.target)
            if isinstance(name, ast.Name)
        ),
    ]
    if parameter is None or new in _names(method.function) or old in bound:
        return None
    reads = [node for node in body if isinstance(node, ast.Name) and node.id == old]
    renamed: list[ast.arg | ast.Name] = [parameter, *reads]
    return [edits.replace(method.token(node, len(old.encode())), new) for node in renamed]


class OverridingMethodRenamesAParameter(Rule, id="ALL-OVER0002", applicability="review"):
    """Positions an overriding method binds under a name the base never used."""

    query = f"""
        SELECT d.*, r.members
        FROM ({
        _debt(
            "SELECT fact_id, renamed_parameters FROM overrides_signatures()",
            "sum(renamed_parameters)",
        )
    }) d
        LEFT JOIN ({
        _members("SELECT * FROM overrides_renames()", "'old': r.own_name, 'new': r.base_name")
    }) r USING (fact_id)
    """
    message = "overriding method renames a parameter is {value} for `{fact_id}`"
    examples = (
        Example(
            bad=(
                "class Base:\n"
                "    def run(self, alpha):\n"
                "        return alpha\n"
                "\n"
                "\n"
                "class Child(Base):\n"
                "    def run(self, value):\n"
                "        return value + 1\n"
            ),
            good=(
                "class Base:\n"
                "    def run(self, alpha):\n"
                "        return alpha\n"
                "\n"
                "\n"
                "class Child(Base):\n"
                "    def run(self, alpha):\n"
                "        return alpha + 1\n"
            ),
        ),
    )

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Each renamed parameter takes the base's name; all of a class's or none (a partial fix
        would change the finding's count)."""
        renamed = [_rename(str(hit.path), member) for member in hit.records("members")]
        if not renamed or None in renamed:
            return None
        return [edit for edits_ in renamed if edits_ for edit in edits_]
