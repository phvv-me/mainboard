"""The vocabulary of the syntax relations: which tree-sitter node types are statements and which are
expressions, with the kind each stands for, the wrappers that carry no meaning, and the groups of
types a stage asks about. `fill` writes them into a SQL template."""

_EXPRESSION = "expression"
_OPERATION = "operation"
_STATEMENT = "statement"
_COLLECTION = "collection"
_COMPREHENSION = "comprehension"

_STATEMENTS = {
    "function_definition": "callable",
    "class_definition": "type",
    "return_statement": "return",
    "raise_statement": "raise",
    "if_statement": "branch",
    "for_statement": "loop",
    "while_statement": "loop",
    "try_statement": "guard",
    "with_statement": "scope",
    "import_statement": "import",
    "import_from_statement": "import",
    "future_import_statement": "import",
    "assignment": "binding",
    "augmented_assignment": "binding",
    "expression_statement": "effect",
    "pass_statement": _STATEMENT,
    "break_statement": _STATEMENT,
    "continue_statement": _STATEMENT,
    "global_statement": _STATEMENT,
    "nonlocal_statement": _STATEMENT,
    "assert_statement": _STATEMENT,
    "delete_statement": _STATEMENT,
    "match_statement": _STATEMENT,
    "type_alias_statement": _STATEMENT,
    "print_statement": _STATEMENT,
    "exec_statement": _STATEMENT,
}

_EXPRESSIONS = {
    "identifier": "name",
    "attribute": "member",
    "call": "call",
    "string": "text",
    "concatenated_string": "text",
    "integer": "literal",
    "float": "literal",
    "true": "literal",
    "false": "literal",
    "none": "literal",
    "ellipsis": _EXPRESSION,
    "list": _COLLECTION,
    "set": _COLLECTION,
    "tuple": _COLLECTION,
    "dictionary": _COLLECTION,
    "expression_list": _COLLECTION,
    "pattern_list": _COLLECTION,
    "tuple_pattern": _COLLECTION,
    "list_pattern": _COLLECTION,
    "list_comprehension": _COMPREHENSION,
    "set_comprehension": _COMPREHENSION,
    "dictionary_comprehension": _COMPREHENSION,
    "generator_expression": _COMPREHENSION,
    "lambda": "callable",
    "binary_operator": _OPERATION,
    "boolean_operator": _OPERATION,
    "comparison_operator": _OPERATION,
    "unary_operator": _OPERATION,
    "not_operator": _OPERATION,
    "union_type": _OPERATION,
    "subscript": "index",
    "generic_type": "index",
    "await": "await",
    "conditional_expression": _EXPRESSION,
    "yield": _EXPRESSION,
    "list_splat": _EXPRESSION,
    "list_splat_pattern": _EXPRESSION,
    "named_expression": _EXPRESSION,
    "slice": _EXPRESSION,
}

# Nodes that carry no meaning of their own: their children belong to the nearest kept ancestor.
_TRANSPARENT = (
    "module",
    "block",
    "parenthesized_expression",
    "argument_list",
    "pair",
    "dictionary_splat",
    "keyword_argument",
    "as_pattern",
    "with_item",
    "with_clause",
    "decorated_definition",
    "decorator",
    "elif_clause",
    "else_clause",
    "except_clause",
    "except_group_clause",
    "finally_clause",
    "case_clause",
    "for_in_clause",
    "if_clause",
    "interpolation",
    "type",
    "type_parameter",
    "constrained_type",
    "parameters",
    "default_parameter",
    "typed_parameter",
    "typed_default_parameter",
)

# What can stand as an operand, to number the operands of a node: `a.b` has the object first.
_OPERANDS = (
    *_EXPRESSIONS,
    "parenthesized_expression",
    "pair",
    "dictionary_splat",
    "keyword_argument",
    "as_pattern",
    "with_item",
    "assignment",
)

# What tree-sitter nests around a star it misreads: `[*a.b()]` is `call(attribute(list_splat(a)))`.
_CHAINED = (
    "attribute",
    "call",
    "subscript",
    "binary_operator",
    "boolean_operator",
    "comparison_operator",
    "conditional_expression",
)

# Statements that end where their last simple statement ends: tree-sitter lets a trailing comment
# at the end of a block stretch every compound statement around it.
_SIMPLE = (
    "expression_statement",
    "return_statement",
    "raise_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "global_statement",
    "nonlocal_statement",
    "assert_statement",
    "delete_statement",
    "import_statement",
    "import_from_statement",
    "future_import_statement",
    "print_statement",
    "exec_statement",
    "type_alias_statement",
)

_COMPOUND = (
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
)


def _list(names: tuple[str, ...] | list[str]) -> str:
    """A SQL list of string literals."""
    return "(" + ", ".join(f"'{name}'" for name in names) + ")"


def fill(sql: str) -> str:
    """The placeholders of a SQL template, replaced by the vocabulary they name."""
    return (
        sql.replace(
            "@VOCABULARY@",
            ", ".join(
                f"('{type}', '{kind}')" for type, kind in {**_STATEMENTS, **_EXPRESSIONS}.items()
            ),
        )
        .replace("@ALIVE@", _list([*_STATEMENTS, *_EXPRESSIONS, *_TRANSPARENT]))
        .replace("@OPERANDS@", _list(_OPERANDS))
        .replace("@CHAINED@", _list(_CHAINED))
        .replace("@FACT_TYPES@", _list(["decorated_definition", *_COMPOUND[:2], *_SIMPLE]))
        .replace("@SIMPLE@", _list(_SIMPLE))
        .replace("@COMPOUND@", _list(_COMPOUND))
    )
