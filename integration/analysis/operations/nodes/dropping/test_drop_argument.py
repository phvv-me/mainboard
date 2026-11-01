"""Dropping an argument takes the comma that parts it from a neighbor, leaves the parentheses of a
call that still has one, takes those of a decorator left without any, defers a second drop on the
same call, and refuses what it cannot take without losing a comment."""

import ast
from collections.abc import Callable, Mapping, Sequence

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from .readers import Dropped

type Drop = Callable[[Mapping[str, str], str, Sequence[tuple[int, str]]], Dropped]

_NAMES = ["k0", "k1", "k2", "k3"]
_SPREAD = "f(\n    a,\n    b,\n    c,\n)\n"


@pytest.mark.parametrize(
    ("path", "source", "target", "expected"),
    [
        pytest.param(
            "m.py",
            "@cuda.jit(debug=True)\ndef k(x): ...\n",
            (1, "debug=True"),
            "@cuda.jit\ndef k(x): ...\n",
            id="the-only-argument-of-a-decorator-takes-its-parentheses",
        ),
        pytest.param(
            "m.py",
            '@f("void(f4[:])")\ndef k(x): ...\n',
            (1, '"void(f4[:])"'),
            "@f\ndef k(x): ...\n",
            id="the-only-positional-argument-of-a-decorator",
        ),
        pytest.param(
            "m.py",
            "f(a)\n",
            (1, "a"),
            "f()\n",
            id="the-only-argument-of-a-call-leaves-its-parentheses",
        ),
        pytest.param(
            "m.py",
            "f(a, b)\n",
            (1, "a"),
            "f(b)\n",
            id="the-first-of-two-arguments-with-the-comma-after-it",
        ),
        pytest.param(
            "m.py",
            "f(a, b)\n",
            (1, "b"),
            "f(a)\n",
            id="the-last-of-two-arguments-with-the-comma-before-it",
        ),
        pytest.param(
            "m.py",
            "f(a, b, c)\n",
            (1, "b"),
            "f(a, c)\n",
            id="the-middle-of-three-arguments",
        ),
        pytest.param(
            "m.py",
            "f(a, b=1, c=2)\n",
            (1, "c=2"),
            "f(a, b=1)\n",
            id="a-keyword-argument-at-the-end",
        ),
        pytest.param(
            "m.py",
            "f(*a, **b)\n",
            (1, "**b"),
            "f(*a)\n",
            id="a-keyword-splat",
        ),
        pytest.param(
            "m.py",
            "f(a, g(b, c))\n",
            (1, "g(b, c)"),
            "f(a)\n",
            id="an-argument-that-is-a-call-with-arguments-of-its-own",
        ),
        pytest.param(
            "m.py",
            "f(a, b,)\n",
            (1, "b"),
            "f(a,)\n",
            id="the-last-argument-before-a-trailing-comma-leaves-the-comma-to-the-tidy",
        ),
        pytest.param(
            "m.py",
            "f( a ,  b )\n",
            (1, "a"),
            "f( b )\n",
            id="an-argument-in-spaced-out-parentheses",
        ),
        pytest.param(
            "m.py",
            'f("a, #b", c)\n',
            (1, "c"),
            'f("a, #b")\n',
            id="a-string-holding-a-hash-and-a-comma-is-no-comment",
        ),
        pytest.param(
            "m.py",
            _SPREAD,
            (2, "a"),
            "f(\n    b,\n    c,\n)\n",
            id="the-first-of-three-lines",
        ),
        pytest.param(
            "m.py",
            _SPREAD,
            (3, "b"),
            "f(\n    a,\n    c,\n)\n",
            id="the-middle-of-three-lines",
        ),
        pytest.param(
            "m.py",
            _SPREAD,
            (4, "c"),
            "f(\n    a,\n    b,\n)\n",
            id="the-last-of-three-lines",
        ),
        pytest.param(
            "m.py",
            "f(\r\n    a,\r\n    b,\r\n)\r\n",
            (3, "b"),
            "f(\r\n    a,\r\n)\r\n",
            id="a-file-with-windows-line-endings",
        ),
        pytest.param(
            "m.py",
            'f("é", "日本", c=1)\n',
            (1, '"日本"'),
            'f("é", c=1)\n',
            id="an-argument-after-multibyte-text",
        ),
        pytest.param(
            "m.py",
            "class C(B, A, metaclass=M):\n    pass\n",
            (1, "A"),
            "class C(B, metaclass=M):\n    pass\n",
            id="a-base-of-a-class",
        ),
        pytest.param(
            "m.c",
            "void h() {\n    f(a, b);\n}\n",
            (2, "b"),
            "void h() {\n    f(a);\n}\n",
            id="an-argument-of-a-c-call",
        ),
    ],
)
def test_a_drop_takes_the_argument_and_the_comma_beside_it_and_nothing_else(
    drop: Drop, path: str, source: str, target: tuple[int, str], expected: str
) -> None:
    dropped = drop({path: source}, path, [target])

    assert (dropped.text, dropped.edits, dropped.taken) == (expected, [["replace"]], [True])


@pytest.mark.parametrize(
    ("path", "source", "target", "why"),
    [
        pytest.param(
            "m.py",
            "f(\n    a,\n    # keep\n    b,\n)\n",
            (4, "b"),
            "a comment stands in the argument or between it and its neighbors",
            id="a-comment-above-the-last-argument",
        ),
        pytest.param(
            "m.py",
            "f(\n    # keep\n    a,\n    b,\n)\n",
            (3, "a"),
            "a comment stands in the argument or between it and its neighbors",
            id="a-comment-above-the-first-argument",
        ),
        pytest.param(
            "m.py",
            "f(a, b,  # why\n  c)\n",
            (1, "b"),
            "a comment stands in the argument or between it and its neighbors",
            id="a-comment-after-the-comma-of-the-argument",
        ),
        pytest.param(
            "m.py",
            "f(a,  # about a\n  b)\n",
            (2, "b"),
            "a comment stands in the argument or between it and its neighbors",
            id="a-comment-after-the-previous-argument-which-the-drop-would-take",
        ),
        pytest.param(
            "m.py",
            "f(a, g(  # inner\n    b))\n",
            (1, "g(  # inner\n    b)"),
            "a comment stands in the argument or between it and its neighbors",
            id="a-comment-inside-the-argument",
        ),
        pytest.param(
            "m.py",
            "f(  # why\n    a)\n",
            (2, "a"),
            "a comment stands in the argument or between it and its neighbors",
            id="a-comment-in-the-parentheses-of-the-only-argument",
        ),
        pytest.param(
            "m.py",
            "f(a)\n",
            (1, "f"),
            "the node is no argument of a call",
            id="the-callee-of-a-call",
        ),
        pytest.param(
            "m.py",
            "def g(a, b): ...\n",
            (1, "b"),
            "the node is no argument of a call",
            id="a-parameter-of-a-definition",
        ),
        pytest.param(
            "m.py",
            "x = f(a)\n",
            (1, "x = f(a)"),
            "the node is no argument of a call",
            id="a-statement",
        ),
        pytest.param(
            "m.c",
            "int f(\n{\n    g(a, b);\n}\n",
            (3, "b"),
            "the node sits in a syntax error",
            id="an-argument-under-a-syntax-error",
        ),
    ],
)
def test_a_drop_is_refused_and_writes_nothing_when_it_would_lose_a_comment_or_no_call_holds_it(
    drop: Drop, path: str, source: str, target: tuple[int, str], why: str
) -> None:
    dropped = drop({path: source}, path, [target])

    assert (dropped.refusals, dropped.edits, dropped.taken, dropped.text) == (
        [why],
        [[]],
        [False],
        source,
    )


@pytest.mark.parametrize(
    ("source", "targets", "taken", "expected"),
    [
        pytest.param(
            "f(a, b)\n",
            ["a", "b"],
            [True, False],
            "f(b)\n",
            id="the-last-argument-overlaps-the-comma-of-the-one-before-it",
        ),
        pytest.param(
            "f(a, b, c)\n",
            ["b", "c"],
            [True, False],
            "f(a, c)\n",
            id="the-last-two-of-three-overlap-through-their-comma",
        ),
        pytest.param(
            "f(a, b, c)\n",
            ["a", "b"],
            [True, True],
            "f(c)\n",
            id="neighbors-before-the-last-only-touch",
        ),
        pytest.param(
            "f(a, b, c)\n", ["a", "c"], [True, True], "f(b)\n", id="arguments-apart-go-together"
        ),
    ],
)
def test_drops_on_one_call_in_one_plan_take_only_those_that_do_not_overlap(
    drop: Drop, source: str, targets: Sequence[str], taken: list[bool], expected: str
) -> None:
    dropped = drop({"m.py": source}, "m.py", [(1, target) for target in targets])

    assert (dropped.taken, dropped.text) == (taken, expected)


def test_the_drop_a_plan_held_back_is_made_by_the_next_pass(drop: Drop) -> None:
    first = drop({"m.py": "f(a, b)\n"}, "m.py", [(1, "a"), (1, "b")])

    second = drop({"m.py": first.text}, "m.py", [(1, "b")])

    assert (first.taken, first.text, second.text) == ([True, False], "f(b)\n", "f()\n")


def _call(names: Sequence[str], layout: str, *, decorated: bool) -> str:
    """A call (a decorator's, or an expression statement's) with the keywords `names`."""
    items = [f"{name}={_NAMES.index(name)}" for name in names]
    head, tail = ("@f", "\ndef g(): ...\n") if decorated else ("f", "\n")
    if not items:
        return head + ("" if decorated else "()") + tail
    body = {
        "inline": ", ".join(items),
        "trailing": ", ".join(items) + ",",
        "spread": "\n" + "".join(f"    {item},\n" for item in items),
    }[layout]
    return f"{head}({body}){tail}"


def _calls() -> st.SearchStrategy[tuple[list[str], int, str, bool]]:
    """Keyword names in order, the place of the one dropped, a layout and a decorator or not."""
    names = st.lists(st.sampled_from(_NAMES), min_size=1, max_size=4, unique=True)
    return names.flatmap(
        lambda each: st.tuples(
            st.just(each),
            st.integers(0, len(each) - 1),
            st.sampled_from(["inline", "trailing", "spread"]),
            st.booleans(),
        )
    )


@settings(max_examples=25)
@example(chosen=(["k0"], 0, "inline", True))
@example(chosen=(["k0", "k1"], 1, "trailing", False))
@example(chosen=(["k0", "k1", "k2"], 1, "spread", True))
@given(chosen=_calls())
def test_dropping_any_argument_leaves_a_call_that_parses_without_it(
    drop: Drop, chosen: tuple[list[str], int, str, bool]
) -> None:
    names, place, layout, decorated = chosen
    line = place + 2 if layout == "spread" else 1

    dropped = drop(
        {"m.py": _call(names, layout, decorated=decorated)},
        "m.py",
        [(line, f"{names[place]}={_NAMES.index(names[place])}")],
    )

    kept = [name for number, name in enumerate(names) if number != place]
    expected = _call(kept, layout, decorated=decorated)
    assert ast.dump(ast.parse(dropped.text)) == ast.dump(ast.parse(expected))
