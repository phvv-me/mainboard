"""Rules over how a subclass overrides the members of its base (mcmr's ALL-OVER family)."""

import ast
from collections.abc import Mapping

from .....catalog import Example, Part, Rule
from .....edits import Edit, Replace
from .....findings import Hit, Span
from .relations import tail

_MANGLED = "starts_with(name, '__') AND NOT ends_with(name, '__')"
_ABSTRACT = "('abstractmethod', 'abstractproperty', 'abstractclassmethod', 'abstractstaticmethod')"


def _debt(selected: str, *, measure: str = "count(*)") -> str:
    """The facts whose `selected` rows (or their `measure`) add up past zero."""
    return f"""
        SELECT f.path, f.start_line, f.start_column, f.fact_id, v.value
        FROM override_facts f
        JOIN (SELECT fact_id, {measure} AS value FROM ({selected}) GROUP BY fact_id) v
          USING (fact_id)
        WHERE v.value > 0
    """


def _members(pairs: str, *, fields: str) -> str:
    """Per fact, the subclass methods of `pairs` as one list of structs.

    `pairs` is a relation of `fact_id, method, ...`, and each struct holds the method's first line,
    column and source, and the `fields` the fix reads.
    """
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

    def __init__(self, path: str, member: Mapping[str, int | str]) -> None:
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

    def arguments(self) -> list[ast.arg]:
        """Every named parameter in declaration order."""
        arguments = self.function.args
        return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]

    def at(self, line: int, *, offset: int) -> Span:
        """The empty span at an `ast` position, in the file's UTF-8 byte columns.

        Source line one is `class _:`.
        """
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
            self.at(node.lineno, offset=node.col_offset),
            self.at(node.end_lineno or node.lineno, offset=node.end_col_offset or node.col_offset),
        )
        return first.model_copy(
            update={"end_line": last.start_line, "end_column": last.start_column}
        )

    def token(self, node: ast.expr | ast.arg, length: int) -> Span:
        """The span of the `length` characters a node opens with.

        For a parameter that is its name, not its annotation.
        """
        first = self.at(node.lineno, offset=node.col_offset)
        return first.model_copy(update={"end_column": first.start_column + length})


class InheritedAttributeHidesAMethod(Part):
    """Methods a subclass writes under a name an ancestor already binds to data."""

    query = _debt(
        f"""SELECT fact_id FROM overrides_pairs()
            WHERE NOT inherited_callable AND declared_callable AND NOT ({_MANGLED})""",
    )
    message = "inherited attribute hides a method is {value} for `{fact_id}`"


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
                                               AND {tail("string_value")} IN {_ABSTRACT}))
               AND EXISTS (SELECT 1 FROM override_declared d
                           WHERE d.fact_id = f.fact_id AND d.name = '__init__')
               AND NOT EXISTS (SELECT 1 FROM override_values
                               WHERE relation = 'initializer_calls' AND fact_id = f.fact_id
                                 AND string_value IN ('super', {tail("f.base")}))''',
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
            fields="'bare': r.bare",
        )
    }) r USING (fact_id)
    """
    message = "subclass initializer skips its base is {value} for `{fact_id}`"
    examples = (
        Example(
            bad=(
                """class Base:
    def __init__(self):
        self.items = []


class Child(Base):
    def __init__(self):
        self.count = 0
"""
            ),
            good=(
                """class Base:
    def __init__(self):
        self.items = []


class Child(Base):
    def __init__(self):
        super().__init__()
        self.count = 0
"""
            ),
        ),
    )

    @staticmethod
    def _super_call(path: str, member: dict[str, int | str]) -> list[Edit] | None:
        """`super().__init__()` on its own line before an initializer's first statement.

        It goes after the docstring, and is None when that statement shares a line with the
        signature.
        """
        method = _Method(path, member)
        body = method.function.body
        has_docstring = (
            isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        if has_docstring and len(body) == 1:
            line, indent = (body[0].end_lineno or body[0].lineno) + 1, body[0].col_offset
        elif method.lines[body[has_docstring].lineno - 2][
            : body[has_docstring].col_offset
        ].strip():
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
        return [Replace(span=start, text=" " * indent + "super().__init__()\n")]

    def repair(self, hit: Hit) -> list[Edit] | None:
        """The base initializer runs first, when it asks for no argument the subclass lacks.

        An argument the subclass would have to invent stops the fix.
        """
        members = [member for member in hit.records("members") if member["bare"]]
        made = [
            SubclassInitializerSkipsItsBase._super_call(str(hit.path), member)
            for member in members
        ]
        return (
            None if not made or None in made else [edit for part in made if part for edit in part]
        )


class InitializerCalledOnAStranger(Part):
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
                               AND b.ordinal = 0), '') = {tail("f.base")}""",
    )
    message = "initializer called on a stranger is {value} for `{fact_id}`"


class OverridingMethodChangesItsCallProtocol(Part):
    """Overrides that change how a caller has to reach the member, not just what it does."""

    query = _debt(
        f"""SELECT fact_id FROM overrides_pairs()
            WHERE inherited_callable AND declared_callable AND NOT ({_MANGLED})
              AND (inherited_is_read_as_data <> declared_is_read_as_data
                   OR inherited_asynchronous <> declared_asynchronous)""",
    )
    message = "overriding method changes its call protocol is {value} for `{fact_id}`"


class AbstractMemberLeftUnimplemented(Part):
    """Promises a base made that the concrete subclass below it never kept."""

    abstract_bases: tuple[str, ...] = ("ABC", "ABCMeta", "Protocol")

    query = _debt(
        f"""WITH promised AS (
              SELECT fact_id, parent_id, split_part(relation, '.', 1) AS side
              FROM override_values
              WHERE relation IN ('declared.decorators', 'inherited.decorators')
                AND {tail("string_value")} IN {_ABSTRACT}
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


class OverridingMethodAcceptsDifferentArguments(Part):
    """Overriding methods that stopped accepting what the method they replace accepts."""

    query = _debt(
        "SELECT fact_id, differing_arguments FROM overrides_signatures()",
        measure="sum(differing_arguments)",
    )
    message = "overriding method accepts different arguments is {value} for `{fact_id}`"


class OverridingMethodRenamesAParameter(Rule, id="ALL-OVER0002", applicability="review"):
    """Positions an overriding method binds under a name the base never used."""

    query = f"""
        SELECT d.*, r.members
        FROM ({
        _debt(
            "SELECT fact_id, renamed_parameters FROM overrides_signatures()",
            measure="sum(renamed_parameters)",
        )
    }) d
        LEFT JOIN ({
        _members(
            "SELECT * FROM overrides_renames()", fields="'old': r.own_name, 'new': r.base_name"
        )
    }) r USING (fact_id)
    """
    message = "overriding method renames a parameter is {value} for `{fact_id}`"
    examples = (
        Example(
            bad=(
                """class Base:
    def run(self, alpha):
        return alpha


class Child(Base):
    def run(self, value):
        return value + 1
"""
            ),
            good=(
                """class Base:
    def run(self, alpha):
        return alpha


class Child(Base):
    def run(self, alpha):
        return alpha + 1
"""
            ),
        ),
    )

    @staticmethod
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

    @staticmethod
    def _rename(path: str, member: dict[str, int | str]) -> list[Edit] | None:
        """A method's parameter `old` renamed `new` in its signature and body.

        It is None when the new name is already used there or a nested scope binds the old one,
        since the reads there may be its own.
        """
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
        if (
            parameter is None
            or new in OverridingMethodRenamesAParameter._names(method.function)
            or old in bound
        ):
            return None
        reads = [node for node in body if isinstance(node, ast.Name) and node.id == old]
        renamed: list[ast.arg | ast.Name] = [parameter, *reads]
        return [Replace(span=method.token(node, len(old.encode())), text=new) for node in renamed]

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Each renamed parameter takes the base's name, all of a class's or none.

        A partial fix would change the finding's count.
        """
        renamed = [
            OverridingMethodRenamesAParameter._rename(str(hit.path), member)
            for member in hit.records("members")
        ]
        if not renamed or None in renamed:
            return None
        return [edit for edits_ in renamed if edits_ for edit in edits_]
