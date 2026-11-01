from collections.abc import Sequence

from patos import FrozenModel

from mainboard.analysis.operations.nodes import Dialect
from mainboard.analysis.operations.nodes.dialects import C, Cpp, Cuda, Python, Rust, TypeScript


class Samples(FrozenModel):
    """Text that exercises one dialect's atoms and binding parents on parsed code.

    wrap: a file around one expression, where `{}` stands.
    atoms: an expression of each atom type.
    contexts: an expression holding a lone hole `{}` under each binding parent.
    plain: expressions of no atom type, which some binding parent must tear apart.
    loose: places that bind nothing tighter, `{}` standing for an expression there.
    everyday: expressions that mean the same in every loose place.
    exempt: the (atom type, binding parent) pairs that mean nothing in the language.
    """

    wrap: str
    atoms: dict[str, str]
    contexts: dict[str, str]
    plain: Sequence[str]
    loose: Sequence[str]
    everyday: Sequence[str]
    exempt: frozenset[tuple[str, str]] = frozenset()


_C_ATOMS = {
    "identifier": "x",
    "field_expression": "a.b",
    "call_expression": "f(1)",
    "subscript_expression": "a[0]",
    "number_literal": "1",
    "string_literal": '"s"',
    "concatenated_string": '"a" "b"',
    "char_literal": "'c'",
    "true": "true",
    "false": "false",
    "null": "NULL",
    "parenthesized_expression": "(a + b)",
    "compound_literal_expression": "(T){1}",
}
_C = Samples(
    wrap="void g() { int q = {}; }\n",
    atoms=_C_ATOMS,
    contexts={
        "field_expression": "{}.a",
        "call_expression": "{}(1)",
        "subscript_expression": "{}[0]",
        "binary_expression": "{} * 2",
        "unary_expression": "!{}",
        "pointer_expression": "*{}",
        "update_expression": "++{}",
        "cast_expression": "(int){}",
        "sizeof_expression": "sizeof {}",
    },
    plain=("a + b", "a || b", "!a", "a ? b : c", "a = 1", "(int)a", "a++", "&a"),
    loose=("{}", "f({})", "c ? {} : 2", "(x = {})"),
    everyday=("a + b", "a || b", "!a", "a ? b : c", "a = 1", "(int)a", "a++", "&a"),
    # `1.a` is no field access of anything: the lexer reads the float `1.` and an identifier.
    exempt=frozenset({("number_literal", "field_expression")}),
)
_CPP = _C.model_copy(
    update={"atoms": {**_C_ATOMS, "qualified_identifier": "a::b", "this": "this"}}
)

# Each dialect with the text that exercises it; CUDA parses with the C++ grammar.
SAMPLES: dict[type[Dialect], Samples] = {
    Python: Samples(
        wrap="x = {}\n",
        atoms={
            "identifier": "x",
            "attribute": "a.b",
            "call": "f(1)",
            "subscript": "a[0]",
            "string": '"s"',
            "concatenated_string": '"a" "b"',
            "float": "1.5",
            "true": "True",
            "false": "False",
            "none": "None",
            "parenthesized_expression": "(a + b)",
            "list": "[1, 2]",
            "dictionary": "{1: 2}",
            "set": "{1, 2}",
            "tuple": "(1, 2)",
        },
        contexts={
            "attribute": "{}.a",
            "call": "{}(1)",
            "subscript": "{}[0]",
            "binary_operator": "{} * 2",
            "unary_operator": "-{}",
            "comparison_operator": "{} < 2",
            "boolean_operator": "{} and 2",
            "not_operator": "not {}",
            "await": "await {}",
        },
        plain=("a + b", "a if b else c", "a or b", "not a", "a < b"),
        loose=(
            "f({})",
            "x = {}",
            "f(k={})",
            "x[{}]",
            "({}, c)",
            "[{} for c in y]",
            "d = [{}]",
        ),
        everyday=("a + b", "a if b else c", "a or b", "not a", "a < b", "-a", "a ** b", "await a"),
    ),
    TypeScript: Samples(
        wrap="const q = {};\n",
        atoms={
            "identifier": "x",
            "member_expression": "a.b",
            "call_expression": "f(1)",
            "subscript_expression": "a[0]",
            "string": "'s'",
            "template_string": "`t`",
            "true": "true",
            "false": "false",
            "null": "null",
            "undefined": "undefined",
            "this": "this",
            "parenthesized_expression": "(a + b)",
            "array": "[1, 2]",
            "object": "{a: 1}",
            "regex": "/re/g",
        },
        contexts={
            "member_expression": "{}.a",
            "call_expression": "{}(1)",
            "subscript_expression": "{}[0]",
            "binary_expression": "{} * 2",
            "unary_expression": "-{}",
            "await_expression": "await {}",
            "non_null_expression": "{}!",
            "as_expression": "{} as T",
            "satisfies_expression": "{} satisfies T",
            "update_expression": "{}++",
        },
        plain=("a + b", "a ? b : c", "a || b", "!a", "a = 1", "await a"),
        loose=("{}", "f({})", "[{}]", "x = {}", "c ? {} : 2"),
        everyday=("a + b", "a ? b : c", "a || b", "!a", "a = 1", "await a"),
    ),
    Rust: Samples(
        wrap="fn main() { let q = {}; }\n",
        atoms={
            "identifier": "x",
            "scoped_identifier": "a::b",
            "field_expression": "a.b",
            "call_expression": "f(1)",
            "index_expression": "a[0]",
            "string_literal": '"s"',
            "raw_string_literal": 'r"s"',
            "integer_literal": "1",
            "float_literal": "1.5",
            "boolean_literal": "true",
            "char_literal": "'c'",
            "parenthesized_expression": "(a + b)",
            "array_expression": "[1, 2]",
            "tuple_expression": "(1, 2)",
            "macro_invocation": "vec![1]",
            "self": "self",
            "unit_expression": "()",
            "try_expression": "a?",
            "await_expression": "a.await",
        },
        contexts={
            "field_expression": "{}.a",
            "call_expression": "{}(1)",
            "index_expression": "{}[0]",
            "binary_expression": "{} * 2",
            "unary_expression": "-{}",
            "reference_expression": "&{}",
            "type_cast_expression": "{} as T",
            "try_expression": "{}?",
            "await_expression": "{}.await",
        },
        plain=("a + b", "a || b", "-a", "a as T", "&a", "|x| x", "a = 1"),
        loose=("{}", "f({})", "[{}]", "({}, 1)"),
        everyday=("a + b", "a || b", "-a", "a as T", "&a", "|x| x", "a = 1"),
    ),
    C: _C,
    Cpp: _CPP,
    Cuda: _CPP,
}
