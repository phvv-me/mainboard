"""A dialect's atoms and binding parents are checked on parsed code: an atom keeps its meaning with
no parentheses under every binding parent, and each binding parent tears apart some expression
that is no atom."""

import ast
from collections.abc import Callable
from functools import partial

import pytest

from mainboard.analysis.operations.nodes import Dialect, Node
from mainboard.analysis.operations.nodes.dialects import C, Cpp, Cuda, Python, Rust, TypeScript

from .constants import SAMPLES

_DIALECTS = [
    pytest.param(Python, id="python"),
    pytest.param(TypeScript, id="typescript"),
    pytest.param(Rust, id="rust"),
    pytest.param(C, id="c"),
    pytest.param(Cpp, id="cpp"),
    pytest.param(Cuda, id="cuda"),
]
_HOLE = "Name(id='__hole__', ctx=Load())"


def _is_intact(
    parse: Callable[[str], list[Node]],
    program: str,
    *,
    sample: str,
    parent: str | None = None,
    kind: str | None = None,
) -> bool:
    """Whether `sample`, set where `program` has `{}`, parses as one node without a syntax error.

    That node has type `kind` and stands under a node of type `parent`, when they are given.
    """
    start = len(program.split("{}")[0].encode())
    nodes = parse(program.replace("{}", sample))
    types: dict[int | None, str] = {node.node: node.type for node in nodes}
    spans = [node for node in nodes if (node.start, node.end) == (start, start + len(sample))]
    return not any(node.type == "ERROR" or node.start == node.end for node in nodes) and any(
        parent in (None, types.get(node.parent)) and kind in (None, node.type) for node in spans
    )


def _is_same_meaning(context: str, *, sample: str) -> bool:
    """Whether CPython reads `sample` in `context` as the one expression it is alone."""
    marked = ast.dump(ast.parse(context.replace("{}", "__hole__")))
    expected = marked.replace(_HOLE, ast.dump(ast.parse(sample, mode="eval").body))
    return ast.dump(ast.parse(context.replace("{}", sample))) == expected


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_every_atom_and_binding_parent_a_dialect_names_has_a_sample(
    dialect: type[Dialect],
) -> None:
    samples = SAMPLES[dialect]

    assert (set(samples.atoms), set(samples.contexts)) == (dialect.atoms, dialect.binding)


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_an_atom_survives_unparenthesized_under_every_binding_parent(
    grammar: Callable[[type[Dialect], str], list[Node]], dialect: type[Dialect]
) -> None:
    samples, parse = SAMPLES[dialect], partial(grammar, dialect)

    torn = [
        (kind, parent)
        for kind, sample in samples.atoms.items()
        for parent, context in samples.contexts.items()
        if (kind, parent) not in samples.exempt
        and not _is_intact(
            parse, samples.wrap.replace("{}", context), sample=sample, parent=parent, kind=kind
        )
    ]

    assert torn == []


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_every_binding_parent_tears_apart_some_expression_that_is_no_atom(
    grammar: Callable[[type[Dialect], str], list[Node]], dialect: type[Dialect]
) -> None:
    samples, parse = SAMPLES[dialect], partial(grammar, dialect)

    loose = [
        parent
        for parent, context in samples.contexts.items()
        if all(
            _is_intact(parse, samples.wrap.replace("{}", context), sample=sample, parent=parent)
            for sample in samples.plain
        )
    ]

    assert loose == []


@pytest.mark.parametrize("dialect", _DIALECTS)
def test_an_everyday_expression_parses_alike_where_a_dialect_binds_nothing_tighter(
    grammar: Callable[[type[Dialect], str], list[Node]], dialect: type[Dialect]
) -> None:
    samples, parse = SAMPLES[dialect], partial(grammar, dialect)

    torn = [
        (context, sample)
        for context in samples.loose
        for sample in samples.everyday
        if not _is_intact(parse, samples.wrap.replace("{}", context), sample=sample)
    ]

    assert torn == []


def test_a_python_atom_keeps_its_meaning_unparenthesized_under_every_binding_parent() -> None:
    samples = SAMPLES[Python]

    changed = [
        (kind, parent)
        for kind, sample in samples.atoms.items()
        for parent, context in samples.contexts.items()
        if not _is_same_meaning(context, sample=sample)
    ]

    assert changed == []


def test_every_python_binding_parent_changes_the_meaning_of_some_expression_that_is_no_atom() -> (
    None
):
    samples = SAMPLES[Python]

    loose = [
        parent
        for parent, context in samples.contexts.items()
        if all(_is_same_meaning(context, sample=sample) for sample in samples.plain)
    ]

    assert loose == []


def test_python_reads_an_everyday_expression_alike_where_it_binds_nothing_tighter() -> None:
    samples = SAMPLES[Python]

    changed = [
        (context, sample)
        for context in samples.loose
        for sample in samples.everyday
        if not _is_same_meaning(context, sample=sample)
    ]

    assert changed == []


def test_python_has_no_integer_atom_since_an_integer_before_an_attribute_is_a_syntax_error() -> (
    None
):
    with pytest.raises(SyntaxError):
        ast.parse("1.a")

    assert "integer" not in Python.atoms


def test_typescript_has_no_number_atom_since_a_number_before_a_member_is_a_syntax_error(
    grammar: Callable[[type[Dialect], str], list[Node]],
) -> None:
    samples = SAMPLES[TypeScript]
    program = samples.wrap.replace("{}", samples.contexts["member_expression"])

    intact = _is_intact(partial(grammar, TypeScript), program, sample="1")

    assert (intact, "number" in TypeScript.atoms) == (False, False)
