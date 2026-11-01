"""Each dialect is found by the language the facts name, and what it says of its grammar holds on
parsed text: the types it names occur, its expando is an identifier and, for CUDA, a function
definition holds its name."""

import ast
from collections.abc import Callable, Mapping

import pytest
from duckdb import DuckDBPyConnection

from mainboard.analysis.operations.nodes import Dialect, Node, Site
from mainboard.analysis.operations.nodes.dialects import C, Cpp, Cuda, Python, Rust, TypeScript
from mainboard.analysis.providers.ducks import Ducks

_FILES = [
    pytest.param(Python, "module.py", id="python"),
    pytest.param(TypeScript, "module.ts", id="typescript"),
    pytest.param(Rust, "lib.rs", id="rust"),
    pytest.param(C, "unit.c", id="c"),
    pytest.param(Cpp, "unit.cpp", id="cpp"),
    pytest.param(Cuda, "kernel.cu", id="cuda"),
]
# An assignment of a sum of two holes, set in each language's smallest file.
_PATTERNS: dict[type[Dialect], str] = {
    Python: "w = $X + $$$Y\n",
    TypeScript: "const w = $X + $$$Y;\n",
    Rust: "fn m() { let w = $X + $$$Y; }\n",
    C: "void m() { int w = $X + $$$Y; }\n",
    Cpp: "void m() { int w = $X + $$$Y; }\n",
    Cuda: "void m() { int w = $X + $$$Y; }\n",
}
# A string interpolating a name, in each language whose braces escape a doubled `{`.
_ESCAPING: dict[type[Dialect], str] = {Python: 'f"{x}"\n'}
# Every function the CUDA corpus file defines, in document order: the C and C++ ones, then each
# kernel in the forms it takes (launch bounds, static, template, namespace, member, out of line).
_DEFINED = [
    "add",
    "h",
    "twice",
    "bounded",
    "st",
    "tmpl",
    "inner",
    "method",
    "out_of_line",
    "launch",
]


@pytest.mark.parametrize(("dialect", "path"), _FILES)
def test_a_dialect_is_found_by_the_language_the_facts_name(
    sites: Mapping[str, Mapping[int, Site]], dialect: type[Dialect], path: str
) -> None:
    root = sites[path][0]

    found = root.dialect

    assert (type(found), found.language) == (dialect, dialect.language)


@pytest.mark.parametrize("dialect", list(_PATTERNS), ids=lambda dialect: dialect.language)
def test_a_dialect_names_the_grammar_the_facts_parse_its_language_with(
    dialect: type[Dialect],
) -> None:
    assert dialect.grammar == Ducks.LANGUAGES[dialect.language][0]


@pytest.mark.parametrize(("dialect", "path"), _FILES)
def test_every_type_a_dialect_names_is_a_node_type_its_corpus_file_holds(
    sites: Mapping[str, Mapping[int, Site]], dialect: type[Dialect], path: str
) -> None:
    held = {node.type for node in sites[path][0].tree}
    named = dialect.atoms | dialect.binding | dialect.blocks | dialect.strings | dialect.wrappers

    assert named - held == set()


@pytest.mark.parametrize("dialect", list(_PATTERNS), ids=lambda dialect: dialect.language)
def test_the_expando_parses_as_one_identifier_in_the_dialects_grammar(
    grammar: Callable[[type[Dialect], str], list[Node]], dialect: type[Dialect]
) -> None:
    pattern = _PATTERNS[dialect].replace("$", dialect.expando)

    holes = {
        (pattern.encode()[node.start : node.end].decode(), node.type)
        for node in grammar(dialect, pattern)
        if pattern.encode()[node.start : node.end].startswith(dialect.expando.encode())
        and node.type == "identifier"
    }

    assert holes == {
        (f"{dialect.expando}X", "identifier"),
        (f"{dialect.expando * 3}Y", "identifier"),
    }


@pytest.mark.parametrize("dialect", list(_ESCAPING), ids=lambda dialect: dialect.language)
def test_every_type_a_dialect_escapes_in_is_a_node_type_its_grammar_parses(
    grammar: Callable[[type[Dialect], str], list[Node]], dialect: type[Dialect]
) -> None:
    held = {node.type for node in grammar(dialect, _ESCAPING[dialect])}

    assert (bool(dialect.escaping), dialect.escaping - held) == (True, set())


def test_every_dialect_that_escapes_a_brace_has_a_string_to_parse() -> None:
    escaping = {dialect for dialect in Dialect.implementations() if dialect.escaping}

    assert escaping == set(_ESCAPING)


def test_a_cuda_function_definition_holds_its_name_in_every_declaration_form(
    cursor: DuckDBPyConnection,
) -> None:
    names = cursor.execute(
        "SELECT name FROM ast_nodes WHERE path = 'kernel.cu' AND type = 'function_definition' "
        "ORDER BY node_id"
    ).fetchall()

    assert [name for (name,) in names] == _DEFINED


def test_a_bodiless_kernel_prototype_names_its_kernel_on_its_declarator_not_its_declaration(
    cursor: DuckDBPyConnection,
) -> None:
    named = cursor.execute(
        "SELECT p.name, d.name FROM ast_nodes p JOIN ast_nodes d "
        "ON d.path = p.path AND d.parent_id = p.node_id AND d.type = 'function_declarator' "
        "WHERE p.path = 'kernel.cu' AND p.type = 'declaration' AND d.name = 'decl_only'"
    ).fetchall()

    assert named == [("__global__", "decl_only")]


@pytest.mark.parametrize(("dialect", "path"), _FILES)
def test_every_string_type_a_dialect_names_has_a_node_spanning_lines_in_its_corpus_file(
    sites: Mapping[str, Mapping[int, Site]], dialect: type[Dialect], path: str
) -> None:
    tree = sites[path][0].tree

    spanning = {
        node.type for node in tree if node.type in dialect.strings and node.line < node.end_line
    }

    assert spanning == dialect.strings


def test_python_needs_a_statement_in_an_emptied_block_and_its_filler_is_one() -> None:
    with pytest.raises(SyntaxError):
        ast.parse("def f():\n")

    assert ast.parse(f"def f():\n    {Python.filler}\n").body


@pytest.mark.parametrize(
    ("dialect", "program"),
    [
        pytest.param(TypeScript, "function f() {}\n", id="typescript"),
        pytest.param(Rust, "fn f() {}\n", id="rust"),
        pytest.param(C, "void f() {}\n", id="c"),
        pytest.param(Cpp, "void f() {}\n", id="cpp"),
        pytest.param(Cuda, "void f() {}\n", id="cuda"),
    ],
)
def test_an_emptied_block_parses_cleanly_where_a_dialect_names_no_block(
    grammar: Callable[[type[Dialect], str], list[Node]], dialect: type[Dialect], program: str
) -> None:
    nodes = grammar(dialect, program)

    assert (any(node.type == "ERROR" for node in nodes), dialect.blocks, dialect.filler) == (
        False,
        frozenset(),
        "",
    )


def test_a_rust_attribute_is_a_sibling_of_its_item_so_rust_has_no_wrapper(
    grammar: Callable[[type[Dialect], str], list[Node]],
) -> None:
    nodes = grammar(Rust, "#[inline]\nfn f() {}\n")

    parents = {node.type: node.parent for node in nodes if node.type.endswith("_item")}

    assert (len(set(parents.values())), Rust.wrappers) == (1, frozenset())
