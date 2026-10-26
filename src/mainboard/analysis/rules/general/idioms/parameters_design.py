"""Rules over function parameter lists (mcmr's ALL-PARA family)."""

import ast
import io
import re
import tokenize
from collections.abc import Mapping, Sequence

from .... import edits
from ....catalog import Example, Relation, Rule
from ....edits import Edit
from ....engine.facts import Setting
from ....findings import Hit, Span

_BOOLEAN = ("bool", "boolean", "Boolean", "_Bool", "BOOL")
_OPENING, _CLOSING = frozenset("([{"), frozenset(")]}")


class ParametersBooleans(Relation):
    """The `parameters_booleans()` relation."""

    query = f"""
    SELECT *, type_name IN {_BOOLEAN} OR has_boolean_annotation AS is_boolean,
           row_number() OVER (PARTITION BY function_id ORDER BY ordinal) - 1 AS argument_index
    FROM function_parameters WHERE NOT is_receiver
"""


class ParametersSwappablePairs(Relation):
    """The `parameters_swappable_pairs()` relation: adjacent positional parameters sharing one
    declared type, by the function, the pair's names and where the second sits in a call."""

    query = """
    SELECT function_id, left_name, right_name, type_name, right_index
    FROM (SELECT function_id, name AS left_name, type_name,
                 lead(name) OVER next AS right_name, lead(type_name) OVER next AS right_type,
                 lead(argument_index) OVER next AS right_index
          FROM parameters_booleans() WHERE NOT is_keyword_only
          WINDOW next AS (PARTITION BY function_id ORDER BY ordinal))
    WHERE type_name <> '' AND type_name = right_type
"""


class ParametersKeywordPlan(Relation):
    """The `parameters_keyword_plan()` relation: for each function with a positional parameter
    that is a Boolean or half of a swappable pair, the first such parameter (where `*` goes) and,
    when making the rest keyword-only could break nothing outside what the facts see, every call
    to name the arguments of.

    The calls must be all of the function's uses: every reference resolved to a call or an import
    (so none is a callback), no call of that name left unresolved anywhere, no string literal of
    the repository spelling its name (a dispatch by name passes positionally), a method's name
    held by no other method (an override would keep the old names) and a signature with no `*` or
    `/`.
    """

    query = r"""
    WITH triggers AS (
      SELECT function_id, argument_index, name FROM parameters_booleans()
      WHERE is_boolean AND NOT is_keyword_only
      UNION ALL
      SELECT function_id, right_index, right_name FROM parameters_swappable_pairs()
    ), bounds AS (
      SELECT function_id, min(argument_index) AS boundary_index,
             arg_min(name, argument_index) AS boundary_name
      FROM triggers GROUP BY function_id
    ), unresolved AS (
      SELECT DISTINCT regexp_extract(qualified_name, '[^.:]+$') AS name FROM call_calls
      WHERE starts_with(target_id, 'python:unresolved-symbol:')
         OR starts_with(target_id, 'python:attribute:')
    ), functions AS (
      SELECT b.function_id, b.boundary_index, b.boundary_name, f.name, f.definition_path,
             f.definition_start_line, f.definition_start_column, f.definition_text, d.qualname
      FROM bounds b
      JOIN function_functions f ON f.entity_id = b.function_id
      JOIN symbol_reach_declarations d
        ON d.span_path = f.path AND d.span_start_line = f.definition_start_line
       AND d.kind IN ('function', 'method')
      WHERE f.scope IN ('module', 'method') AND NOT starts_with(f.name, '__')
        AND ends_with(f.definition_path, '.py')
        AND NOT EXISTS (SELECT 1 FROM function_decorators x
                        WHERE x.function_id = f.entity_id
                          AND x.decorator NOT IN ('staticmethod', 'classmethod'))
        AND NOT regexp_matches(regexp_extract(f.definition_text,
              '(?s)^(?:@[^\n]*\n\s*)*(?:async\s+)?def\s+\w+\s*\(.*?\)\s*(?:->[^\n]*?)?:', 0),
              '[(,]\s*[*/]')
        AND d.call_count + d.import_count = d.own_file_references + d.other_file_references
        AND f.name NOT IN (SELECT name FROM unresolved)
        AND f.name NOT IN (SELECT word FROM reach_quoted_words())
        AND (f.scope = 'module' OR (SELECT count(*) FROM function_functions g
                                    WHERE g.name = f.name AND g.scope = 'method') = 1)
    ), sites AS (
      SELECT fn.function_id,
             list({'path': c.node_path, 'line': c.node_start_line,
                   'column': c.node_start_column,
                   'text': c.node_text}
                  ORDER BY c.node_path, c.node_start_line, c.node_start_column) AS calls
      FROM functions fn
      JOIN call_calls c ON c.target_id IN ('python:function:' || fn.qualname,
                                           'python:method:' || fn.qualname)
      GROUP BY fn.function_id
    ), names AS (
      SELECT function_id, list(name ORDER BY argument_index) AS names
      FROM parameters_booleans() GROUP BY function_id
    )
    SELECT fn.*, coalesce(s.calls, []) AS calls, n.names
    FROM functions fn LEFT JOIN sites s USING (function_id) JOIN names n USING (function_id)
"""


class SwappableParameterPair(Rule, id="ALL-PARA0001", applicability="review"):
    """Adjacent positional parameters of one type a caller can silently swap. Device code takes
    its arguments by position, pytest passes a test's by name, a `Protocol` method only
    describes the signature of what implements it, and a method overriding a project base takes
    the base's signature, which is reported once at the base, so none of them is reported."""

    query = """
        SELECT f.path, f.start_line, f.start_column, f.name, d.left_name, d.right_name,
               d.type_name, p.boundary_index, p.boundary_name, p.definition_path,
               p.definition_start_line, p.definition_start_column, p.definition_text, p.calls,
               p.names
        FROM parameters_swappable_pairs() d
        JOIN function_functions f ON f.entity_id = d.function_id
        LEFT JOIN parameters_keyword_plan() p ON p.function_id = d.function_id
        WHERE NOT starts_with(f.name, 'test_') AND NOT f.is_protocol_member
          AND NOT EXISTS (SELECT 1 FROM cuda_device_code() c
                          WHERE c.path = f.path AND c.name = f.name
                            AND f.start_line <= c.start_line AND c.end_line <= f.end_line)
          AND NOT EXISTS (SELECT 1 FROM override_links l
                          JOIN override_members m ON m.class_id = l.class_id
                          JOIN override_members b ON b.class_id = l.base_id AND b.name = m.name
                          WHERE l.path = f.path AND m.name = f.name AND m.line = f.start_line
                            AND m.is_callable AND b.is_callable)
    """
    message = (
        "`{name}` takes `{left_name}` and `{right_name}` next to each other and both are "
        "`{type_name}`, so a caller can transpose them silently"
    )
    examples = (
        Example(
            bad=(
                """def blend(left: float, right: float) -> float:
    return left + right


value = blend(1.0, 2.0)
"""
            ),
            good=(
                """def blend(left: float, *, right: float) -> float:
    return left + right


value = blend(1.0, right=2.0)
"""
            ),
        ),
    )

    @staticmethod
    def _argument_starts(text: str) -> list[tuple[int, int]] | None:
        """Where each positional argument of the call `text` opens, as (line, column).

        It is None when an argument could not take a name in front of it (`*xs`, `**kw`, a bare
        generator expression, an assignment expression).
        """
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type
            not in {
                tokenize.NL,
                tokenize.COMMENT,
                tokenize.NEWLINE,
                tokenize.ENDMARKER,
                tokenize.INDENT,
            }
        ]
        starts: list[tuple[int, int]] = []
        depth, expect = 0, True
        for number in range(SwappableParameterPair._opening(tokens) + 1, len(tokens) - 1):
            token = tokens[number]
            top = depth == 0
            if top and (token.string in {"for", ":="} or (expect and token.string in {"*", "**"})):
                return None
            named = token.type == tokenize.NAME and tokens[number + 1].string == "="
            if top and expect and named:
                break
            if top and expect:
                starts.append(token.start)
            depth += token.type == tokenize.OP and token.string in _OPENING
            depth -= token.type == tokenize.OP and token.string in _CLOSING
            expect = top and token.type == tokenize.OP and token.string == ","
        return starts

    @staticmethod
    def _declared(hit: Hit) -> list[Edit] | None:
        """The `*, ` before the parameter names start from, None when no function is read.

        An empty edit at the `def` comes first, so a second finding of the same function does not
        add a second `*`.
        """
        column, first = int(hit.definition_start_column or 0), int(hit.definition_start_line or 1)
        wrapper = "class _:\n" if column else ""
        outer = ast.parse(wrapper + " " * column + str(hit.definition_text)).body[0]
        function = outer.body[0] if isinstance(outer, ast.ClassDef) else outer
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            return None
        parameter = next(
            a
            for a in (*function.args.posonlyargs, *function.args.args)
            if a.arg == hit.boundary_name
        )
        shift = first - 1 - bool(column)
        return [
            edits.replace(
                _point(str(hit.definition_path), line=shift + node.lineno, column=node.col_offset),
                text,
            )
            for node, text in ((function, ""), (parameter, "*, "))
        ]

    @staticmethod
    def _named(hit: Hit, call: Mapping[str, int | str]) -> list[Edit] | None:
        """`name=` before each positional argument of `call` from the boundary on.

        None when the call cannot take the names.
        """
        names, text = hit.sequence("names"), str(call["text"])
        starts = SwappableParameterPair._argument_starts(text)
        if starts is None or len(starts) > len(names):
            return None
        lines = text.split("\n")
        return [
            edits.replace(
                _point(
                    str(call["path"]),
                    line=int(call["line"]) + row - 1,
                    column=len(lines[row - 1][:offset].encode())
                    + (int(call["column"]) if row == 1 else 0),
                ),
                f"{names[index]}=",
            )
            for index, (row, offset) in list(enumerate(starts))[int(hit.boundary_index or 0) :]
        ]

    @staticmethod
    def _opening(tokens: Sequence[tokenize.TokenInfo]) -> int:
        """Where the bracket the last of `tokens` closes opens."""
        opens: list[int] = []
        opener: dict[int, int] = {}
        for number, token in enumerate(tokens):
            if token.type == tokenize.OP and token.string in _OPENING:
                opens.append(number)
            elif token.type == tokenize.OP and token.string in _CLOSING:
                opener[number] = opens.pop()
        return opener[len(tokens) - 1]

    def repair(self, hit: Hit) -> list[Edit] | None:
        """Everything from the pair's second parameter on becomes keyword-only, named at calls.

        Only a function whose every use the facts resolved (`parameters_keyword_plan()`) is fixed.
        `*, ` goes before that parameter and `name=` before each positional argument from there on
        at every call; nothing is fixed when a call cannot take the names.
        """
        if hit.boundary_name is None or hit.calls is None:
            return None
        result = SwappableParameterPair._declared(hit)
        for call in hit.records("calls"):
            if result is None or (found := SwappableParameterPair._named(hit, call)) is None:
                return None
            result += found
        return result


class ConfigurationObjectParameter(Rule, id="ALL-PARA0002"):
    """Parameters a callable only reads attributes from."""

    minimum_reads: int = 2
    configuration_markers: tuple[str, ...] = ("config", "configuration", "options", "settings")

    query = """
        WITH attributes AS (
          SELECT fact_id, parent_id AS record_id, count(*) AS attribute_count,
                 string_agg(string_value, '`, `' ORDER BY string_value) AS attributes
          FROM (SELECT DISTINCT fact_id, parent_id, string_value FROM parameter_values
                WHERE relation = 'parameters.attribute_reads')
          GROUP BY ALL)
        SELECT coalesce(p.span_path, f.path) AS path,
               coalesce(p.span_start_line, f.start_line) AS start_line,
               coalesce(p.span_start_column, f.start_column) AS start_column,
               p.owner, p.name, p.annotation, a.attributes
        FROM parameter_parameters p
        JOIN parameter_facts f USING (fact_order, fact_id)
        JOIN attributes a USING (fact_id, record_id)
        WHERE p.all_uses_known AND p.operations_length = 0 AND a.attribute_count >= $minimum_reads
          AND regexp_matches(p.annotation, $marker)
    """
    message = "`{owner}` takes `{name}` as a whole `{annotation}` and reads only `{attributes}`"

    def parameters(self) -> dict[str, Setting]:
        marker = "(?i)" + "|".join(re.escape(marker) for marker in self.configuration_markers)
        return super().parameters() | {"marker": marker}


class BooleanParameterCount(Rule, id="ALL-PARA0004"):
    """The flags one callable takes, wherever a caller passes them."""

    maximum: int = 1

    query = """
        WITH flags AS (
          SELECT function_id, count(*) AS flag_count,
                 string_agg(CASE WHEN type_name <> ''
                                 THEN '`' || name || '` as `' || type_name || '`'
                                 WHEN has_boolean_annotation
                                 THEN '`' || name || '` with a Boolean annotation'
                                 ELSE '`' || name || '` with a Boolean default' END,
                            ', ' ORDER BY ordinal) AS descriptions
          FROM parameters_booleans() WHERE is_boolean OR has_boolean_default
          GROUP BY function_id HAVING count(*) > $maximum)
        SELECT f.path, f.start_line, f.start_column,
               '`' || f.name || '` takes ' || v.flag_count
                 || CASE WHEN v.flag_count = 1 THEN ' Boolean parameter, which creates '
                         ELSE ' Boolean parameters, which create ' END
                 || CAST(2 ** v.flag_count AS BIGINT) || ' behavior combinations, with '
                 || v.descriptions AS message
        FROM flags v JOIN function_functions f ON f.entity_id = v.function_id
    """


def _point(path: str, *, line: int, column: int) -> Span:
    """The empty span at `line` and `column` of `path`, where an edit inserts."""
    return Span(path=path, start_line=line, start_column=column, end_line=line, end_column=column)
