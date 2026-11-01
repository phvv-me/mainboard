"""A pattern unifies with the code it describes, its holes take what they stand for with the
comments they own, and a template puts the captures back where its holes stand."""

from collections.abc import Callable, Sequence

import pytest
from duckdb import DuckDBPyConnection
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mainboard.analysis.operations.nodes import Node, Rewrite, Tree
from mainboard.analysis.operations.nodes.dialects import Python
from mainboard.analysis.operations.nodes.patterns import Template
from mainboard.analysis.operations.nodes.rewrite import Unifier

from .builders import swallows

type Parse = Callable[[str], list[Node]]
type Fields = dict[str, str | tuple[str, ...] | dict[str, tuple[str, ...]]]

_EXPRESSIONS = [
    "x",
    "a.b",
    "f(x)",
    "f(x, y)",
    "[1, 2]",
    "{1: 2}",
    "(a, b)",
    "a + b",
    "not a",
    "f(g(x))",
    "a[0]",
    '"s"',
]
_BODIES = [("a = 1",), ("b = f(x)  # note",), ("# lone", "c = 2"), ("d += 1",)]
_TRY = "try:\n    $$$BODY\nexcept $E:\n    $I"


def _tree(parse: Parse, text: str) -> Tree:
    """The tree of `text` over its own bytes."""
    return Tree(parse(text), text.encode())


def _node(tree: Tree, kind: str | None) -> Node:
    """The first node of type `kind`, else the file's first statement."""
    root = next(node for node in tree.nodes if node.parent is None)
    return next((n for n in tree.nodes if n.type == kind), tree.significant(root)[0])


def _unified(
    parse: Parse, pattern: str, *, target: str, kind: str | None = None
) -> tuple[bool, Unifier, Node]:
    """Unify a pattern (holes spelled `$`) with the target's statement or node of type `kind`.

    That answers whether it unified, the unifier holding what the holes took and the target node.
    """
    shape, found = _tree(parse, pattern.replace("$", Python.expando)), _tree(parse, target)
    unifier = Unifier(shape, target=found, expando=Python.expando)
    node = _node(found, kind)
    return unifier.unify(_node(shape, kind), target=node), unifier, node


def _rendered(
    parse: Parse, pattern: str, *, target: str, template: str, kind: str | None = None
) -> str | None:
    """The target node's replacement: the template filled with what the pattern's holes took."""
    unified, unifier, node = _unified(parse, pattern, target=target, kind=kind)
    parent = next((n for n in unifier.target.nodes if n.node == node.parent), None)
    assert unified
    return Template(template, dialect=Python()).rendered(
        unifier.captures, unifier.target, node, parent=parent
    )


@settings(max_examples=40)
@given(left=st.sampled_from(_EXPRESSIONS), right=st.sampled_from(_EXPRESSIONS))
def test_a_pattern_without_holes_unifies_with_exactly_its_own_text(
    parse: Parse, left: str, right: str
) -> None:
    unified, _, _ = _unified(parse, left, target=right)

    assert unified == (left == right)


@pytest.mark.parametrize(
    ("pattern", "target", "texts"),
    [
        pytest.param("f($X)", "f(1)", {"X": "1"}, id="a-hole-takes-one-argument"),
        pytest.param("f($X, $X)", "f(a, a)", {"X": "a"}, id="a-name-held-twice-takes-equal-text"),
        pytest.param("f($X, $X)", "f(a, b)", None, id="a-name-held-twice-refuses-unequal-text"),
        pytest.param("f($_, $Y)", "f(a, b)", {"Y": "b"}, id="a-wildcard-takes-nothing-kept"),
        pytest.param("f($$$A)", "f()", {"A": ""}, id="a-run-hole-takes-no-argument"),
        pytest.param(
            "f($$$A, $L)",
            "f(1, 2, 3)",
            {"A": "1, 2", "L": "3"},
            id="a-run-hole-leaves-the-last-argument-to-the-hole-after-it",
        ),
        pytest.param("f($X)", "f(1, 2)", None, id="a-pattern-lacking-an-argument-stops"),
        pytest.param("f($X)", "g(1)", None, id="a-different-callee-stops"),
        pytest.param(
            _TRY,
            "try:\n    a()\nexcept E:\n    pass\nelse:\n    b()\n",
            None,
            id="an-extra-clause-the-pattern-lacks-stops-it",
        ),
        pytest.param(
            "try:\n    $$$BODY\nexcept ($$$E,):\n    $I",
            "try:\n    a()\nexcept (A, B):\n    pass\n",
            {"BODY": "a()", "E": "A, B", "I": "pass"},
            id="a-tuple-of-handlers-binds-a-run",
        ),
        pytest.param(
            "try:\n    $$$BODY\nexcept ($$$E,):\n    $I",
            "try:\n    a()\nexcept A:\n    pass\n",
            None,
            id="a-single-handler-is-no-tuple",
        ),
        pytest.param(
            "if $C:\n    $S",
            "if x:\n    for y in z:\n        pass\n",
            {"C": "x", "S": "for y in z:\n        pass"},
            id="a-statement-hole-takes-any-one-statement",
        ),
    ],
)
def test_a_pattern_unifies_with_the_code_it_describes_and_each_hole_takes_its_text(
    parse: Parse, pattern: str, target: str, texts: dict[str, str] | None
) -> None:
    unified, unifier, _ = _unified(parse, pattern, target=target)

    assert (unified, unifier.texts()) == (texts is not None, texts or {})


@pytest.mark.parametrize(
    ("target", "texts"),
    [
        pytest.param(
            "try:\n    # note\n    a()\nexcept E:\n    pass\n",
            {"BODY": "# note\n    a()"},
            id="a-comment-between-the-keyword-and-the-body-goes-with-the-body",
        ),
        pytest.param(
            "try:\n    a()\n    # after\nexcept E:\n    pass\n",
            {"BODY": "a()\n    # after"},
            id="a-comment-after-the-last-statement-goes-with-the-body",
        ),
        pytest.param(
            "try:\n    a = 1  # one\n    # two\n    b = 2  # three\nexcept E:\n    pass\n",
            {"BODY": "a = 1  # one\n    # two\n    b = 2  # three"},
            id="trailing-and-own-line-comments-inside-the-body-stay-in-it",
        ),
        pytest.param(
            "try:\n    a()\nexcept E:  # gone\n    pass\n",
            {"BODY": "a()"},
            id="a-comment-ending-the-handler-line-is-in-no-capture",
        ),
    ],
)
def test_a_run_hole_captures_every_statement_and_the_comments_it_owns(
    parse: Parse, target: str, texts: dict[str, str]
) -> None:
    _, unifier, _ = _unified(parse, _TRY, target=target)

    assert {name: text for name, text in unifier.texts().items() if name == "BODY"} == texts


def test_a_comment_between_two_runs_belongs_to_the_run_below_it_and_is_held_once(
    parse: Parse,
) -> None:
    pattern = "if $C:\n    $$$A\n    mid()\n    $$$B"
    target = "if x:\n    a = 1  # one\n    # two\n    mid()\n    # three\n    b = 2\n"

    _, unifier, _ = _unified(parse, pattern, target=target)

    assert (unifier.texts()["A"], unifier.texts()["B"]) == ("a = 1  # one", "# three\n    b = 2")


@settings(max_examples=30)
@given(body=st.lists(st.sampled_from(_BODIES), min_size=1, max_size=3))
def test_a_template_naming_each_capture_where_it_stood_renders_the_original_text(
    parse: Parse, body: Sequence[tuple[str, ...]]
) -> None:
    lines = "\n        ".join(line for item in body for line in item)
    target = f"def run(x):\n    if x:\n        {lines}\n"

    text = _rendered(
        parse,
        "if $C:\n    $$$BODY",
        target=target,
        template="if $C:\n    $$$BODY",
        kind="if_statement",
    )

    assert text == f"if x:\n        {lines}"


@pytest.mark.parametrize(
    ("target", "template", "expected"),
    [
        pytest.param(
            "if x:\n    y = f(\n  1,\n)\n",
            "$$$BODY",
            None,
            id="a-capture-shifted-left-past-its-text-refuses",
        ),
        pytest.param(
            'if x:\n    y = """a\nb"""\n',
            "if $C:\n        $$$BODY",
            None,
            id="a-string-with-line-breaks-is-never-re-indented",
        ),
        pytest.param(
            "if x:\n    y = f(\n        1,\n    )\n",
            "while $C:\n        $$$BODY",
            "while x:\n        y = f(\n            1,\n        )",
            id="a-capture-moved-right-takes-its-later-lines-along",
        ),
    ],
)
def test_a_capture_is_moved_to_the_column_its_hole_stands_at_or_refuses(
    parse: Parse, target: str, template: str, expected: str | None
) -> None:
    text = _rendered(parse, "if $C:\n    $$$BODY", target=target, template=template)

    assert text == expected


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        pytest.param("g($$$A)", None, id="a-comment-would-swallow-the-closing-parenthesis"),
        pytest.param("g($$$A\n)", "g(a,  # first\n)", id="a-comment-may-end-its-line"),
    ],
)
def test_a_capture_ending_in_a_comment_leaves_no_template_text_on_its_line(
    parse: Parse, template: str, expected: str | None
) -> None:
    text = _rendered(parse, "f($$$A)", target="f(a,  # first\n)\n", template=template)

    assert text == expected


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        pytest.param("a * cast(int, b + c)", "(b + c)", id="under-a-binary-operator"),
        pytest.param("-cast(int, b + c)", "(b + c)", id="under-a-unary-operator"),
        pytest.param("cast(int, b + c)[0]", "(b + c)", id="under-a-subscript"),
        pytest.param("cast(int, b).attr", "b", id="an-atom-needs-none-under-an-attribute"),
        pytest.param("[cast(int, b + c)]", "b + c", id="as-an-element-of-a-list"),
        pytest.param("cast(int, b + c)", "b + c", id="as-a-statement"),
        pytest.param("x if cast(int, a) else y", "a", id="an-atom-in-a-conditional"),
        pytest.param("cast(int, a\n    + b)", "(a\n    + b)", id="an-operator-on-a-new-line"),
        pytest.param('cast(int, "a"\n    "b")', '("a"\n    "b")', id="strings-on-two-lines"),
        pytest.param(
            "cast(int, f(\n    a,\n))", "f(\n    a,\n)", id="a-call-covering-its-own-lines"
        ),
        pytest.param('cast(int, """a\nb""")', '"""a\nb"""', id="a-string-with-a-line-break"),
        pytest.param(
            "cast(int, {\n    1: 2,\n})", "{\n    1: 2,\n}", id="a-dictionary-over-lines"
        ),
        pytest.param('f"{cast(dict, {x: 1})}"', "({x: 1})", id="a-dictionary-in-an-interpolation"),
        pytest.param('f"{cast(set, {x})}"', "({x})", id="a-set-in-an-interpolation"),
        pytest.param('f"{cast(dict, {x: 1}):>5}"', "({x: 1})", id="a-dictionary-before-a-spec"),
        pytest.param('f"{cast(int, a + b)}"', "a + b", id="an-operator-in-an-interpolation"),
        pytest.param('f"{cast(dict, d)}"', "d", id="a-name-in-an-interpolation"),
    ],
)
def test_a_lone_hole_is_parenthesized_exactly_where_leaving_it_bare_changes_its_meaning(
    parse: Parse, target: str, expected: str
) -> None:
    text = _rendered(parse, "cast($T, $X)", target=target, template="$X", kind="call")

    assert text == expected


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        pytest.param({"match": "f($x)", "to": "$x"}, "upper case", id="a-lower-case-hole"),
        pytest.param(
            {"match": "f($X)", "to": "$Y"}, "are no holes", id="a-template-keeping-an-unknown-hole"
        ),
        pytest.param(
            {"match": "f($X)", "to": "$_"}, "are no holes", id="a-template-keeping-the-wildcard"
        ),
        pytest.param(
            {"match": "f($X)", "to": "$X", "where": {"Y": ("a",)}},
            "are no holes",
            id="a-where-naming-an-unknown-hole",
        ),
        pytest.param(
            {"match": "f($X)", "to": "$X", "language": "typescript", "imports": ("import a",)},
            "Python statements",
            id="imports-for-a-language-that-has-none",
        ),
    ],
)
def test_a_rewrite_names_only_the_holes_its_pattern_has(fields: Fields, reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        Rewrite.model_validate(fields)


def test_the_parse_a_rule_states_in_sql_is_the_parse_the_tests_declare(
    cursor: DuckDBPyConnection, parse: Parse
) -> None:
    rewrite = swallows()[0]
    finding = "WITH finding AS (SELECT 1 AS finding_key, 'a.py' AS path, 1 AS node) "

    row = cursor.execute(finding + Rewrite.fix([rewrite]), Rewrite.bindings([rewrite])).fetchone()

    assert row is not None
    assert [Node(**node) for node in row[-1][0]] == parse(rewrite.expando())
