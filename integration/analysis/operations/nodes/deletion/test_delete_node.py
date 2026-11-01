"""Deleting a node takes what belongs to it (decorators, the comment lines above, the blank lines
above those) and nothing else, puts the dialect's filler where a block would be left empty, claims
that block so a second deletion there waits a pass, and refuses what it cannot take whole."""

import ast
from collections.abc import Callable, Mapping, Sequence

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from .constants import BRACED_IF, DECORATED, FUNCTION, STATEMENT, UNBRACED_LOOP
from .readers import Deletion

type Delete = Callable[[Mapping[str, str], str, Sequence[tuple[int, str]]], Deletion]

_INERT = ["x.a", "x.b", "x == 1", "x != 2", "x.c.d"]
_DECORATED = (
    "import os\n\n\n{}@dec\n@dec2(1)\ndef f(x):\n    return x\n\n\ndef g():\n    return 1\n"
)
_CUDA_KERNEL = (
    "#include <x>\n\ntemplate <typename T>\n__global__ void t(T *p) { }\n\nvoid h() {}\n"
)
_EXTERN = 'extern "C" {\n// k\n__global__ void k(int *p) { p[0] = 1; }\n}\n'
_RUST = "fn first() {}\n\n// about dead\n/// doc\nfn dead() {}\n\nfn live() {}\n"
_CONSTRUCT = "a control construct holds the node with no block around it"
_ELSE = """int f(int x) {
    if (x) {
        x = 2;
    } else
        x == 1;
    return x;
}
"""
_CASE = """int f(int x) {
    switch (x) {
    case 1:
        x == 1;
        break;
    }
    return x;
}
"""


@pytest.mark.parametrize(
    ("path", "source", "target", "expected", "kinds"),
    [
        pytest.param(
            "m.py",
            "def f(x):\n    a = 1\n\n    # why\n    # also\n    x == 2\n\n    return x\n",
            (6, STATEMENT),
            "def f(x):\n    a = 1\n\n    return x\n",
            ("delete", "claim"),
            id="a-statement-with-the-comment-lines-and-blank-lines-above",
        ),
        pytest.param(
            "m.py",
            _DECORATED.format("# note\n"),
            (5, DECORATED),
            "import os\n\n\ndef g():\n    return 1\n",
            ("delete",),
            id="a-decorated-definition-with-decorators-comments-and-blank-lines",
        ),
        pytest.param(
            "m.py",
            _DECORATED.format(""),
            (6, FUNCTION),
            "import os\n\n\ndef g():\n    return 1\n",
            ("delete",),
            id="the-definition-inside-a-decorated-one-takes-its-decorators",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    x == 1  # why\n    return x\n",
            (2, STATEMENT),
            "def f(x):\n    return x\n",
            ("delete", "claim"),
            id="a-comment-ending-its-last-line",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    y = 1  # about y\n    x == 1\n    return x\n",
            (3, STATEMENT),
            "def f(x):\n    y = 1  # about y\n    return x\n",
            ("delete", "claim"),
            id="the-trailing-comment-of-the-statement-above-stays",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    a = 1\n    # alone\n\n    x == 1\n\n    return x\n",
            (5, STATEMENT),
            "def f(x):\n    a = 1\n    # alone\n\n    return x\n",
            ("delete", "claim"),
            id="a-comment-parted-by-a-blank-line-stays",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    a = 1\n\n    x == 1\n    return x\n",
            (4, STATEMENT),
            "def f(x):\n    a = 1\n\n    return x\n",
            ("delete", "claim"),
            id="the-blank-line-above-stays-when-code-follows-at-once",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    a = 1\n\n    x == 1\n",
            (4, STATEMENT),
            "def f(x):\n    a = 1\n",
            ("delete", "claim"),
            id="the-blank-line-above-goes-with-the-last-statement-of-a-body",
        ),
        pytest.param(
            "m.c",
            "void f(int x) {\n    int a;\n\n    x == 1;\n}\n",
            (4, STATEMENT),
            "void f(int x) {\n    int a;\n}\n",
            ("delete",),
            id="the-blank-line-above-goes-before-a-closing-brace",
        ),
        pytest.param(
            "m.py",
            "# header\nx == 1\ny = 2\n",
            (2, STATEMENT),
            "# header\ny = 2\n",
            ("delete",),
            id="a-comment-opening-the-file-stays",
        ),
        pytest.param(
            "m.py",
            "def f(x):\r\n    a = 1\r\n    # c\r\n    x == 1\r\n    return x\r\n",
            (4, STATEMENT),
            "def f(x):\r\n    a = 1\r\n    return x\r\n",
            ("delete", "claim"),
            id="a-file-with-windows-line-endings",
        ),
        pytest.param(
            "m.py",
            "def dead():\n    pass\n\n\ndef live():\n    pass\n",
            (1, FUNCTION),
            "def live():\n    pass\n",
            ("delete",),
            id="the-first-definition-of-a-file-with-the-blank-lines-below",
        ),
        pytest.param(
            "m.py",
            "x = 1\n\n\ndef f():\n    return 1",
            (4, FUNCTION),
            "x = 1\n",
            ("delete",),
            id="the-last-definition-of-a-file-without-a-final-newline",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    # keep\n    x.value\n",
            (3, STATEMENT),
            "def f(x):\n    # keep\n    pass\n",
            ("replace", "claim"),
            id="the-only-statement-of-a-block-becomes-pass",
        ),
        pytest.param(
            "m.py",
            "if True:\n    x == 1\nelse:\n    x == 2\n",
            (2, STATEMENT),
            "if True:\n    pass\nelse:\n    x == 2\n",
            ("replace", "claim"),
            id="one-branch-of-an-if",
        ),
        pytest.param(
            "m.py",
            "class A:\n    def f(self):\n        pass\n\n    def g(self):\n        pass\n",
            (2, FUNCTION),
            "class A:\n\n    def g(self):\n        pass\n",
            ("delete", "claim"),
            id="a-method-of-a-class-body",
        ),
        pytest.param(
            "m.c",
            "void f(int x) {\n    x == 1;\n}\n",
            (2, STATEMENT),
            "void f(int x) {\n}\n",
            ("delete",),
            id="a-c-statement-in-a-body-that-needs-none",
        ),
        pytest.param(
            "m.c",
            BRACED_IF,
            (3, STATEMENT),
            BRACED_IF.replace("        x == 1;\n", ""),
            ("delete",),
            id="a-c-statement-in-the-braces-of-an-if",
        ),
        pytest.param(
            "m.c",
            "#include <a.h>\n#include <b.h>\n\nvoid f(void) {}\n",
            (1, "preproc_include"),
            "#include <b.h>\n\nvoid f(void) {}\n",
            ("delete",),
            id="a-c-directive-ending-with-its-newline",
        ),
        pytest.param(
            "k.cu",
            _CUDA_KERNEL,
            (4, FUNCTION),
            "#include <x>\n\nvoid h() {}\n",
            ("delete",),
            id="a-cuda-kernel-with-its-template",
        ),
        pytest.param(
            "k.cu",
            _EXTERN,
            (3, FUNCTION),
            'extern "C" {\n}\n',
            ("delete",),
            id="a-cuda-kernel-inside-braces-of-a-linkage-specification",
        ),
        pytest.param(
            "m.ts",
            "export function dead() {}\n\nexport function live() {}\n",
            (1, "function_declaration"),
            "export function live() {}\n",
            ("delete",),
            id="a-typescript-function-with-its-export",
        ),
        pytest.param(
            "lib.rs",
            _RUST,
            (5, "function_item"),
            "fn first() {}\n\nfn live() {}\n",
            ("delete",),
            id="a-rust-function-with-its-line-comments",
        ),
    ],
)
def test_a_deletion_takes_what_belongs_to_the_node_and_nothing_else(
    delete: Delete,
    path: str,
    source: str,
    target: tuple[int, str],
    expected: str,
    kinds: tuple[str, ...],
) -> None:
    deleted = delete({path: source}, path, [target])

    assert (deleted.text, deleted.edits, deleted.taken) == (expected, [list(kinds)], [True])


@pytest.mark.parametrize(
    ("path", "source", "target", "why"),
    [
        pytest.param(
            "m.py",
            "def f(x):\n    x; return x\n",
            (2, STATEMENT),
            "the node shares a line with other code",
            id="a-statement-before-another-on-its-line",
        ),
        pytest.param(
            "m.py",
            "def f(x): x == 1\n",
            (1, STATEMENT),
            "the node shares a line with other code",
            id="a-statement-after-the-colon-of-its-block",
        ),
        pytest.param(
            "m.py",
            "def f(x):\n    x == 1;\n    return x\n",
            (2, STATEMENT),
            "the node shares a line with other code",
            id="a-statement-followed-by-a-semicolon",
        ),
        pytest.param(
            "m.c",
            "void f(int x) {\n    x == 1; /* a\n b */\n}\n",
            (2, STATEMENT),
            "the node shares a line with other code",
            id="a-statement-followed-by-a-comment-reaching-further-lines",
        ),
        pytest.param(
            "m.py",
            "x = 1\n",
            (1, "module"),
            "the node is the root",
            id="the-root-of-the-file",
        ),
        pytest.param(
            "m.go",
            "package main\n\nfunc f() {\n\tx == 1\n}\n",
            (4, STATEMENT),
            "no dialect describes go",
            id="a-language-no-dialect-describes",
        ),
        pytest.param(
            "lib.rs",
            "#[inline]\nfn dead() {}\n\nfn live() {}\n",
            (2, "function_item"),
            "an annotation above the node would pass to the next one",
            id="an-attribute-above-that-would-pass-to-the-next-item",
        ),
        pytest.param(
            "m.ts",
            "export declare function dead(): void;\n\nexport function live() {}\n",
            (1, "function_signature"),
            "wrappers nest",
            id="a-wrapper-inside-a-wrapper",
        ),
        pytest.param(
            "m.c",
            "int f(\n{\n    x == 1;\n    y == 2;\n}\n",
            (3, STATEMENT),
            "the node sits in a syntax error",
            id="a-statement-under-a-syntax-error",
        ),
        pytest.param(
            "m.c",
            "int f(int x) {\n    if (x)\n        x == 1;\n    return x;\n}\n",
            (3, STATEMENT),
            _CONSTRUCT,
            id="the-whole-body-of-a-c-if-with-no-braces",
        ),
        pytest.param(
            "m.c",
            _ELSE,
            (5, STATEMENT),
            _CONSTRUCT,
            id="the-whole-body-of-a-c-else-with-no-braces",
        ),
        pytest.param(
            "m.c",
            "int f(int x) {\n    while (x > 0)\n        x == 1;\n    return x;\n}\n",
            (3, STATEMENT),
            _CONSTRUCT,
            id="the-whole-body-of-a-c-while-with-no-braces",
        ),
        pytest.param(
            "m.c",
            "int f(int x) {\n    do\n        x == 1;\n    while (x);\n    return x;\n}\n",
            (3, STATEMENT),
            _CONSTRUCT,
            id="the-whole-body-of-a-c-do-loop-with-no-braces",
        ),
        pytest.param(
            "k.cu",
            UNBRACED_LOOP,
            (3, STATEMENT),
            _CONSTRUCT,
            id="the-whole-body-of-a-cuda-for-with-no-braces",
        ),
        pytest.param(
            "m.c",
            _CASE,
            (4, STATEMENT),
            _CONSTRUCT,
            id="a-statement-of-a-case",
        ),
        pytest.param(
            "m.c",
            "int f(int x) {\nout:\n    x == 1;\n    return x;\n}\n",
            (3, STATEMENT),
            _CONSTRUCT,
            id="the-statement-a-label-names",
        ),
        pytest.param(
            "m.ts",
            "function f(x: number): number {\n  if (x > 0)\n    x;\n  return x;\n}\n",
            (3, STATEMENT),
            _CONSTRUCT,
            id="the-whole-body-of-a-typescript-if-with-no-braces",
        ),
    ],
)
def test_a_deletion_is_refused_and_writes_nothing_when_it_cannot_take_the_node_whole(
    delete: Delete, path: str, source: str, target: tuple[int, str], why: str
) -> None:
    deleted = delete({path: source}, path, [target])

    assert (deleted.refusals, deleted.edits, deleted.taken, deleted.text) == (
        [why],
        [[]],
        [False],
        source,
    )


@pytest.mark.parametrize(
    ("source", "lines", "taken", "expected"),
    [
        pytest.param(
            "def f(x):\n    x.a\n    x.b\n",
            [2, 3],
            [True, False],
            "def f(x):\n    x.b\n",
            id="two-deletions-in-one-block-keep-the-second-for-the-next-pass",
        ),
        pytest.param(
            "def f(x):\n    x.a\n    x.b\n    x.c\n",
            [2, 4],
            [True, False],
            "def f(x):\n    x.b\n    x.c\n",
            id="two-of-three-statements-in-one-block",
        ),
        pytest.param(
            "if True:\n    x == 1\nelse:\n    x == 2\n",
            [2, 4],
            [True, True],
            "if True:\n    pass\nelse:\n    pass\n",
            id="one-deletion-in-each-of-two-blocks",
        ),
        pytest.param(
            "x == 1\ny == 2\nz == 3\n",
            [1, 2, 3],
            [True, True, True],
            "",
            id="every-statement-of-a-module-which-needs-none",
        ),
    ],
)
def test_deletions_in_one_plan_claim_the_block_they_judged(
    delete: Delete, source: str, lines: Sequence[int], taken: list[bool], expected: str
) -> None:
    deleted = delete({"m.py": source}, "m.py", [(line, STATEMENT) for line in lines])

    assert (deleted.taken, deleted.text) == (taken, expected)


def test_the_deletion_a_claim_held_back_is_made_by_the_next_pass_and_leaves_the_filler(
    delete: Delete,
) -> None:
    first = delete(
        {"m.py": "def f(x):\n    x.a\n    x.b\n"}, "m.py", [(2, STATEMENT), (3, STATEMENT)]
    )

    second = delete({"m.py": first.text}, "m.py", [(2, STATEMENT)])

    assert (first.taken, first.text, second.text) == (
        [True, False],
        "def f(x):\n    x.b\n",
        "def f(x):\n    pass\n",
    )


def _bodies() -> st.SearchStrategy[tuple[list[str], int]]:
    """A body of one to four inert statements and the place of the one deleted."""
    bodies = st.lists(st.sampled_from(_INERT), min_size=1, max_size=4)
    return bodies.flatmap(lambda body: st.tuples(st.just(body), st.integers(0, len(body) - 1)))


def _source(body: Sequence[str]) -> str:
    """A function whose body is the statements `body`."""
    return "def f(x):\n" + "".join(f"    {line}\n" for line in body)


@settings(max_examples=12)
@example(chosen=(["x.a"], 0))
@example(chosen=(["x.a", "x.b", "x.c", "x.a"], 3))
@given(chosen=_bodies())
def test_deleting_one_of_n_statements_leaves_n_minus_one_or_the_filler(
    delete: Delete, chosen: tuple[list[str], int]
) -> None:
    body, place = chosen

    deleted = delete({"m.py": _source(body)}, "m.py", [(place + 2, STATEMENT)])

    kept = [line for number, line in enumerate(body) if number != place] or ["pass"]
    assert ast.dump(ast.parse(deleted.text)) == ast.dump(ast.parse(_source(kept)))
