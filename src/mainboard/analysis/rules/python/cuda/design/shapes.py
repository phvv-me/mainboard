"""PY-CUDA rules about the shape of GPU code: records on the device and casts that change
nothing."""

from .....catalog import Example, Rule
from ...constants import cuda_examples, cuda_paths


class RecordOnDeviceIsNamedTuple(Rule, id="PY-CUDA0010", applicability="review"):
    """A `NamedTuple` used as a value inside device code (built there, or typed on a device
    function's parameter or return): the project's device record is `Struct`, one record concept
    instead of two."""

    exempt: tuple[str, ...] = cuda_paths.TESTS
    change = "rewrote `{name}({base})` as `{name}(Struct)`"

    query = r"""
        WITH tuples AS (
          SELECT d.key AS class_key, d.fact_id, d.path, d.name, d.start_line, d.start_column,
                 d.end_line, d.end_column, b.written, ba.node_id AS base_id,
                 ba.start_line AS base_line, ba.start_column AS base_column,
                 ba.end_line AS base_end_line, ba.end_column AS base_end_column
          FROM cuda_definitions() d
          JOIN cuda_bases() b ON b.class_key = d.key AND b.qualified = 'typing.NamedTuple'
          JOIN cuda_nodes() ar ON ar.fact_id = d.fact_id AND ar.parent_id = d.node_id
            AND ar.type = 'argument_list'
          JOIN cuda_nodes() ba ON ba.fact_id = d.fact_id AND ba.parent_id = ar.node_id
            AND ba.name = b.written
        ), built AS (
          SELECT t.key AS class_key, c.path, c.start_line, c.shown, f.qualname AS function_name
          FROM cuda_resolved_calls() t
          JOIN cuda_calls() c ON c.fact_id = t.fact_id AND c.node_id = t.node_id
          JOIN cuda_device_code() f ON f.fact_id = c.fact_id AND f.function_id = c.scope_id
          WHERE t.kind = 'class'
          UNION
          SELECT h.definition, a.path, a.start_line, a.annotation, f.qualname
          FROM cuda_annotations() a
          JOIN cuda_device_code() f ON f.fact_id = a.fact_id AND f.function_id = a.owner_id
          JOIN cuda_chains() h ON h.fact_id = a.fact_id AND h.chain = cuda_annotation_head(a.annotation)
          WHERE a.site IN ('parameter', 'return') AND h.definition IS NOT NULL
        ), used AS (
          SELECT class_key, min(path || ':' || start_line) AS first_use,
                 arg_min(function_name, path || ':' || start_line) AS function_name
          FROM built GROUP BY class_key
        )
        SELECT t.path, t.start_line, t.start_column, t.end_line, t.end_column, t.name,
               t.written AS base, t.base_line, t.base_column, t.base_end_line, t.base_end_column,
               u.first_use, u.function_name
        FROM tuples t JOIN used u ON u.class_key = t.class_key
        WHERE EXISTS (SELECT 1 FROM cuda_names() n
                      WHERE regexp_matches(n.qualified, '^patos\.cuda\.typed(\.\w+)*\.Struct$'))
          AND NOT coalesce(len($exempt) > 0 AND regexp_matches(t.path, '(?:' || array_to_string($exempt, ')|(?:') || ')'), false)
    """
    message = (
        "`{name}` is a `NamedTuple` used on the device (`{function_name}`, {first_use}): the device "
        "record of this project is `Struct`, so a value that crosses the kernel boundary has two "
        "record concepts, each with its own typing and calling rules. Fix: declare `{name}` as a "
        "`Struct`"
    )
    fix = """
        SELECT finding_key, path, base_line AS start_line, base_column AS start_column,
               base_end_line AS end_line, base_end_column AS end_column, 'Struct' AS text,
               'from patos.cuda.typed import Struct' AS import
        FROM finding
    """
    examples = (
        Example(
            bad=(
                """from typing import NamedTuple

from patos.cuda.typed import Struct, device


class Span(NamedTuple):
    start: int
    end: int


class Spans(Struct):
    starts: int

    @device
    def get(self, index: int) -> Span:
        return Span(index, index)
"""
            ),
            good=(
                """from patos.cuda.typed import Struct, device


class Span(Struct):
    start: int
    end: int


class Spans(Struct):
    starts: int

    @device
    def get(self, index: int) -> Span:
        return Span(index, index)
"""
            ),
        ),
    )


class RedundantScalarCast(Rule, id="PY-CUDA0013", applicability="safe"):
    """A scalar cast in device code that something already does: the value has that kind (a
    variable, an array element or a record field declared with it), the parameter, the declared
    variable, the array element or the return annotation (each element of a tuple return, each
    branch of a conditional) converts to it on its own, or the operator that meets it with an
    operand of that kind does (an arithmetic operator, an augmented assignment, a comparison,
    `min` or `max`), under the C integer rules patos types device arithmetic by."""

    exempt: tuple[str, ...] = cuda_paths.TESTS
    change = "dropped the cast `{shown}`, keeping `{argument}`"

    query = r"""
        WITH callers AS (
          SELECT a.fact_id, a.value_id AS cast_id, a.call_id, a.position, a.keyword
          FROM cuda_arguments() a
        ), hits AS (
          SELECT c.*, 'kind' AS why
          FROM cuda_casts() c
          JOIN cuda_expression_kinds() k ON k.fact_id = c.fact_id AND k.node_id = c.argument_id
            AND k.kind = c.kind
          UNION
          SELECT c.*, 'parameter'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN callers a ON a.fact_id = q.fact_id AND a.cast_id = q.root_id
          JOIN cuda_resolved_calls() r ON r.fact_id = a.fact_id AND r.node_id = a.call_id
            AND r.kind = 'function' AND r.via <> 'override'
          JOIN cuda_definitions() f ON f.fact_id = r.definition_fact AND f.node_id = r.definition_id
          JOIN cuda_parameters() p ON p.fact_id = f.fact_id AND p.function_id = f.node_id
            AND (p.name = a.keyword
                 OR p.position = a.position + 1
                    + CASE WHEN f.class_id IS NOT NULL AND f.outer_id IS NULL
                                AND NOT EXISTS (SELECT 1 FROM cuda_decorators() x
                                                WHERE x.fact_id = f.fact_id
                                                  AND x.definition_id = f.node_id
                                                  AND x.written = 'staticmethod')
                           THEN 1 ELSE 0 END)
          JOIN cuda_declared_kinds() k ON k.fact_id = f.fact_id AND k.function_id = f.node_id
            AND k.site = 'parameter' AND k.name = p.name AND k.kind = c.kind
          UNION
          SELECT c.*, 'declaration'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_stores() s ON s.fact_id = q.fact_id AND s.source_id = q.root_id
            AND s.kind = 'assignment'
          JOIN cuda_declared_kinds() k ON k.fact_id = s.fact_id AND k.function_id = s.scope_id
            AND k.name = s.name AND k.kind = c.kind AND k.site IN ('parameter', 'local')
          UNION
          SELECT c.*, 'return'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_nodes() r ON r.fact_id = c.fact_id AND r.type = 'return_statement'
            AND r.scope."function" = c.scope_id
          JOIN cuda_nodes() v ON v.fact_id = r.fact_id AND v.parent_id = r.node_id
            AND v.node_id = q.root_id
          JOIN cuda_declared_kinds() k ON k.fact_id = c.fact_id AND k.function_id = c.scope_id
            AND k.site = 'return' AND k.kind = c.kind
          UNION
          SELECT c.*, 'tuple'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_nodes() x ON x.fact_id = q.fact_id AND x.node_id = q.root_id
          JOIN cuda_nodes() t ON t.fact_id = x.fact_id AND t.node_id = x.parent_id
            AND t.type IN ('expression_list', 'tuple')
          JOIN cuda_nodes() r ON r.fact_id = t.fact_id AND r.node_id = t.parent_id
            AND r.type = 'return_statement' AND r.scope."function" = c.scope_id
          JOIN cuda_return_kinds() e ON e.fact_id = c.fact_id AND e.function_id = c.scope_id
            AND e.kind = c.kind
            AND e.position = (SELECT count(*) FROM cuda_nodes() b
                              WHERE b.fact_id = t.fact_id AND b.parent_id = t.node_id
                                AND b.sibling_index < x.sibling_index
                                AND b.type NOT IN ('(', ',', ')'))
          UNION
          SELECT c.*, 'element'
          FROM cuda_casts() c
          JOIN cuda_cast_roots() q ON q.fact_id = c.fact_id AND q.node_id = c.node_id
          JOIN cuda_nodes() a ON a.fact_id = c.fact_id AND a.type = 'assignment'
          JOIN cuda_nodes() l ON l.fact_id = a.fact_id AND l.parent_id = a.node_id
            AND l.sibling_index = 0 AND l.type = 'subscript'
          JOIN cuda_nodes() b ON b.fact_id = l.fact_id AND b.parent_id = l.node_id AND b.sibling_index = 0
          JOIN cuda_name_uses() u ON u.fact_id = b.fact_id AND u.node_id = b.node_id
          JOIN cuda_declared_kinds() k
            ON 'v:' || k.fact_id || ':' || k.function_id || ':' || k.name = u.variable
           AND k.element_kind = c.kind
          WHERE q.root_id = (SELECT max(r.node_id) FROM cuda_nodes() r
                             WHERE r.fact_id = a.fact_id AND r.parent_id = a.node_id)
            AND a.node_id < q.root_id AND q.root_id <= a.node_id + a.descendant_count
          UNION
          SELECT c.*, 'operator'
          FROM cuda_casts() c
          JOIN cuda_expression_kinds() inner_kind ON inner_kind.fact_id = c.fact_id
            AND inner_kind.node_id = c.argument_id
          JOIN cuda_operations() o ON o.fact_id = c.fact_id AND o.operand_id = c.node_id
          JOIN cuda_expression_kinds() other_kind ON other_kind.fact_id = o.fact_id
            AND other_kind.node_id = o.other_id AND other_kind.kind = c.kind
          WHERE CASE o.meeting
                  WHEN 'combined' THEN cuda_combined(c.kind, inner_kind.kind)
                  ELSE cuda_operated(o.operator, c.kind, inner_kind.kind) END = c.kind
        )
        SELECT h.path, h.start_line, h.start_column, h.end_line, h.end_column, h.shown, h.kind,
               CASE h.why
                 WHEN 'kind' THEN 'the value is already `' || h.kind || '`'
                 WHEN 'parameter' THEN 'the call converts the argument to the parameter''s `'
                      || h.kind || '`'
                 WHEN 'declaration' THEN 'the assignment converts to the declared `' || h.kind || '`'
                 WHEN 'return' THEN 'the return annotation converts to `' || h.kind || '`'
                 WHEN 'tuple' THEN 'the return annotation converts this element to `' || h.kind || '`'
                 WHEN 'operator' THEN 'the operator converts to the `' || h.kind
                      || '` of the other operand'
                 ELSE 'the store into the array converts to its `' || h.kind || '` element' END
                 AS reason, arg.start_line AS argument_line, arg.start_column AS argument_column,
               arg.end_line AS argument_end_line, arg.end_column AS argument_end_column,
               syntax_slice(s.content, arg.start_line, arg.start_column, arg.end_line,
                            arg.end_column) AS argument,
               arg.type IN ('binary_operator', 'conditional_expression', 'comparison_operator',
                            'boolean_operator', 'lambda', 'not_operator', 'unary_operator')
               AND enclosing.type IN ('binary_operator', 'unary_operator', 'comparison_operator',
                                   'boolean_operator', 'not_operator', 'attribute', 'subscript',
                                   'conditional_expression', 'await') AS needs_parentheses
        FROM hits h
        JOIN cuda_nodes() arg ON arg.fact_id = h.fact_id AND arg.node_id = h.argument_id
        JOIN cuda_nodes() site ON site.fact_id = h.fact_id AND site.node_id = h.node_id
        JOIN cuda_nodes() enclosing ON enclosing.fact_id = site.fact_id AND enclosing.node_id = site.parent_id
        JOIN source_files s ON s.path = h.path
        WHERE NOT coalesce(list_bool_or(list_transform($exempt, lambda p: regexp_matches(h.path, p))), false)
    """
    message = "`{shown}` is a redundant cast: {reason}, so dropping it changes nothing. Fix: write `{argument}`"
    fix = """
        SELECT finding_key, path, start_line, start_column, end_line, end_column,
               CASE WHEN needs_parentheses THEN '(' || argument || ')' ELSE argument END AS text
        FROM finding
    """
    examples = (
        Example(
            bad=(
                """from patos.cuda.scalars import i32
from patos.cuda.typed import device


@device
def twice(value: i32) -> i32:
    return i32(value) * 2
"""
            ),
            good=(
                """from patos.cuda.scalars import i32
from patos.cuda.typed import device


@device
def twice(value: i32) -> i32:
    return value * 2
"""
            ),
        ),
        Example(
            bad="from patos.cuda.scalars import i32\nfrom patos.cuda.typed import device\n\n\n@device\ndef bump(value: i32) -> i32:\n    total: i32 = i32(value + 1)\n    return total\n",
            good="from patos.cuda.scalars import i32\nfrom patos.cuda.typed import device\n\n\n@device\ndef bump(value: i32) -> i32:\n    total: i32 = value + 1\n    return total\n",
        ),
        Example(
            bad=(
                """from patos.cuda.scalars import i16, i64, u8, u32, u64
from patos.cuda.typed import device


@device
def widen(wide: u64, narrow: u8, step: u32, small: i16, big: i64) -> None:
    total = wide + u64(narrow)
    least = min(big, i64(small))
    step += u32(narrow)
"""
            ),
            good=(
                """from patos.cuda.scalars import i16, i64, u8, u32, u64
from patos.cuda.typed import device


@device
def widen(wide: u64, narrow: u8, step: u32, small: i16, big: i64) -> None:
    total = wide + narrow
    least = min(big, small)
    step += narrow
"""
            ),
        ),
        Example(
            bad=(
                """from patos.cuda.scalars import u32
from patos.cuda.typed import Struct, device


class Header(Struct):
    mask: u32


@device
def low(header: Header, wide: bool) -> tuple[u32, u32]:
    return u32(header.mask) if wide else header.mask, u32(header.mask)
"""
            ),
            good=(
                """from patos.cuda.scalars import u32
from patos.cuda.typed import Struct, device


class Header(Struct):
    mask: u32


@device
def low(header: Header, wide: bool) -> tuple[u32, u32]:
    return header.mask if wide else header.mask, header.mask
"""
            ),
        ),
        Example(
            bad=(
                """from ..typed import device, u32


@device
def keep(value: u32) -> u32:
    return u32(value)
"""
            ),
            good=(
                """from ..typed import device, u32


@device
def keep(value: u32) -> u32:
    return value
"""
            ),
            path="patos/cuda/primitives/keep.py",
            context={
                **cuda_examples.PATOS_PACKAGE,
                "patos/cuda/dec.py": "def device(function):\n    return function\n",
                "patos/cuda/scalars.py": (
                    "from numba import cuda\nimport numpy as np\n\nu32 = np.uint32\n"
                ),
                "patos/cuda/typed.py": (
                    'from .dec import device\nfrom .scalars import u32\n\n__all__ = ["device", "u32"]\n'
                ),
            },
        ),
    )
