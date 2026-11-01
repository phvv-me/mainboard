# What a node operation must know of one language's syntax, found by the name the facts give it.

from typing import ClassVar

from patos import FrozenModel, Registry


class Dialect(Registry, FrozenModel):
    """What node operations must know of one language's syntax, found by the facts' name for it.

    atoms: expression types needing no parentheses wherever a lone expression lands.
    binding: parent types binding tighter than a bare expression, which wraps a lone expression
        that is no atom.
    blocks: bodies that must hold a statement, which `filler` fills once emptied.
    escaping: parent types whose braces read a doubled `{` as text, so a lone expression opening
        with `{` (a dictionary or set display) stays in parentheses there.
    expando: what `$` becomes in a pattern, a letter its language reads as an identifier.
    filler: the statement an emptied block holds, empty when the language needs none.
    grammar: the tree-sitter grammar the language parses with, which is not always its name (CUDA
        is C++ to tree-sitter).
    strings: string types whose inner lines are content, which a re-indent leaves alone.
    wrappers: types a definition is deleted with.

    A slot a parent type shares with looser syntax is not modeled: a call is no constructor of a
    TypeScript `new`, and a conditional or a lambda is no condition of a Python conditional.
    """

    atoms: ClassVar[frozenset[str]]
    binding: ClassVar[frozenset[str]]
    blocks: ClassVar[frozenset[str]] = frozenset()
    escaping: ClassVar[frozenset[str]] = frozenset()
    expando: ClassVar[str] = "µ"
    filler: ClassVar[str] = ""
    grammar: ClassVar[str]
    language: ClassVar[str]
    strings: ClassVar[frozenset[str]]
    wrappers: ClassVar[frozenset[str]] = frozenset()


class Python(Dialect):
    """No `integer` among the atoms: `1.a` lexes as a float, so `1` before an attribute needs
    parentheses."""

    atoms = frozenset(
        {
            "attribute",
            "call",
            "concatenated_string",
            "dictionary",
            "false",
            "float",
            "identifier",
            "list",
            "none",
            "parenthesized_expression",
            "set",
            "string",
            "subscript",
            "true",
            "tuple",
        }
    )
    binding = frozenset(
        {
            "attribute",
            "await",
            "binary_operator",
            "boolean_operator",
            "call",
            "comparison_operator",
            "not_operator",
            "subscript",
            "unary_operator",
        }
    )
    blocks = frozenset({"block"})
    escaping = frozenset({"interpolation"})
    filler = "pass"
    grammar = "python"
    language = "python"
    strings = frozenset({"string"})
    wrappers = frozenset({"decorated_definition"})


class Rust(Dialect):
    """No wrappers: an attribute is a sibling item before the definition, not a node around it."""

    atoms = frozenset(
        {
            "array_expression",
            "await_expression",
            "boolean_literal",
            "call_expression",
            "char_literal",
            "field_expression",
            "float_literal",
            "identifier",
            "index_expression",
            "integer_literal",
            "macro_invocation",
            "parenthesized_expression",
            "raw_string_literal",
            "scoped_identifier",
            "self",
            "string_literal",
            "try_expression",
            "tuple_expression",
            "unit_expression",
        }
    )
    binding = frozenset(
        {
            "await_expression",
            "binary_expression",
            "call_expression",
            "field_expression",
            "index_expression",
            "reference_expression",
            "try_expression",
            "type_cast_expression",
            "unary_expression",
        }
    )
    grammar = "rust"
    language = "rust"
    strings = frozenset({"raw_string_literal", "string_literal"})


class TypeScript(Dialect):
    """No `number` among the atoms (`1.a` is a syntax error) and no `non_null_expression` (`-x!`
    groups as `(-x)!`)."""

    atoms = frozenset(
        {
            "array",
            "call_expression",
            "false",
            "identifier",
            "member_expression",
            "null",
            "object",
            "parenthesized_expression",
            "regex",
            "string",
            "subscript_expression",
            "template_string",
            "this",
            "true",
            "undefined",
        }
    )
    binding = frozenset(
        {
            "as_expression",
            "await_expression",
            "binary_expression",
            "call_expression",
            "member_expression",
            "non_null_expression",
            "satisfies_expression",
            "subscript_expression",
            "unary_expression",
            "update_expression",
        }
    )
    grammar = "typescript"
    language = "typescript"
    strings = frozenset({"string", "template_string"})
    wrappers = frozenset({"ambient_declaration", "export_statement"})


class CFamily(Dialect):
    """What C, C++ and CUDA share. A pointer, cast, update or `sizeof` expression binds tighter
    than a binary operator, so each is a binding parent beside the unary and postfix ones."""

    atoms = frozenset(
        {
            "call_expression",
            "char_literal",
            "compound_literal_expression",
            "concatenated_string",
            "false",
            "field_expression",
            "identifier",
            "null",
            "number_literal",
            "parenthesized_expression",
            "string_literal",
            "subscript_expression",
            "true",
        }
    )
    binding = frozenset(
        {
            "binary_expression",
            "call_expression",
            "cast_expression",
            "field_expression",
            "pointer_expression",
            "sizeof_expression",
            "subscript_expression",
            "unary_expression",
            "update_expression",
        }
    )
    strings = frozenset({"string_literal"})
    wrappers = frozenset({"linkage_specification"})


class C(CFamily):
    grammar = "c"
    language = "c"


class Cpp(CFamily):
    atoms = CFamily.atoms | {"qualified_identifier", "this"}
    grammar = "cpp"
    language = "cpp"
    strings = CFamily.strings | {"raw_string_literal"}
    wrappers = CFamily.wrappers | {"template_declaration"}


class Cuda(Cpp):
    """CUDA files parse with the C++ grammar, which reads `__global__` and `<<<` as errors."""

    language = "cuda"
