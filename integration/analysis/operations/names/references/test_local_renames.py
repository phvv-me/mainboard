"""A function's local is renamed in the scopes that read it and in none that binds its own, and a
rename that could change what a scope reads is refused with its reason."""

import pytest

from mainboard.analysis.operations import RewriteReferences

from ..builders import code, local_row, rewritten


@pytest.mark.parametrize(
    ("source", "after"),
    [
        pytest.param(
            code("def f(rows):", "    ab = len(rows)", "    return ab"),
            code("def f(rows):", "    n = len(rows)", "    return n"),
            id="a-plain-local",
        ),
        pytest.param(
            code("def f(rows):", "    ab = len(rows)", "    def g():", "        return ab + 1"),
            code("def f(rows):", "    n = len(rows)", "    def g():", "        return n + 1"),
            id="a-closure-reading-it",
        ),
        pytest.param(
            code(
                "def f():",
                "    ab = 1",
                "    def g(ab, x=ab):",
                "        return ab",
                "    return g(ab)",
            ),
            code(
                "def f():",
                "    n = 1",
                "    def g(ab, x=n):",
                "        return ab",
                "    return g(n)",
            ),
            id="a-nested-parameter-of-the-same-name-whose-default-reads-it",
        ),
        pytest.param(
            code(
                "def f(rows):",
                "    ab = 2",
                "    return [ab * r for r in rows], sum(ab for _ in rows)",
            ),
            code(
                "def f(rows):",
                "    n = 2",
                "    return [n * r for r in rows], sum(n for _ in rows)",
            ),
            id="a-comprehension-and-a-generator",
        ),
        pytest.param(
            code("def f(rows):", "    ab = len(rows)", "    return f'{ab!r:>{ab}}'"),
            code("def f(rows):", "    n = len(rows)", "    return f'{n!r:>{n}}'"),
            id="an-f-string",
        ),
        pytest.param(
            code("def f(self):", "    ab = self.ab", "    return g(ab=ab)"),
            code("def f(self):", "    n = self.ab", "    return g(ab=n)"),
            id="an-attribute-and-a-keyword-of-the-same-name",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    h = lambda ab: ab", "    return h(ab)"),
            code("def f():", "    n = 1", "    h = lambda ab: ab", "    return h(n)"),
            id="a-lambda-binding-its-own",
        ),
        pytest.param(
            code(
                "def f(rows):",
                "    ab = 0",
                "    for ab in rows:",
                "        ab += 1",
                "    del ab",
            ),
            code("def f(rows):", "    n = 0", "    for n in rows:", "        n += 1", "    del n"),
            id="a-loop-target-an-augmented-assignment-and-a-deletion",
        ),
        pytest.param(
            code("class C:", "    def m(self):", "        ab = 1", "        return ab  # ab"),
            code("class C:", "    def m(self):", "        n = 1", "        return n  # ab"),
            id="a-method-and-a-comment-that-stays",
        ),
        pytest.param(
            code("async def f(rows):", "    ab = await rows", "    return ab"),
            code("async def f(rows):", "    n = await rows", "    return n"),
            id="an-async-function",
        ),
        pytest.param(
            code("def f():", "    é = 'é'; ab = 1", "    return f'é{ab}é'"),
            code("def f():", "    é = 'é'; n = 1", "    return f'é{n}é'"),
            id="columns-after-non-ascii-text",
        ),
        pytest.param(
            code(
                "def f(w):", "    ab = 1", '    return f"x= {ab}" + f"{w=:{ab}}" + f"{w!r:{ab}}"'
            ),
            code("def f(w):", "    n = 1", '    return f"x= {n}" + f"{w=:{n}}" + f"{w!r:{n}}"'),
            id="f-string-fields-that-print-no-source-of-it",
        ),
    ],
)
def test_a_local_and_every_read_of_it_are_renamed(source: str, after: str) -> None:
    operation = RewriteReferences()

    edits = operation.edits(local_row(source, to="n"))

    assert edits is not None
    assert rewritten(source, edits) == after


@pytest.mark.parametrize(
    ("source", "to", "why"),
    [
        pytest.param(code("def f(ab):", "    return ab"), "n", "no plain local", id="a-parameter"),
        pytest.param(
            code("def f():", "    return 1"), "n", "no plain local", id="a-name-never-bound"
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    n = 2", "    return ab + n"),
            "n",
            "already a name",
            id="a-new-name-bound-in-the-function",
        ),
        pytest.param(
            code("n = 3", "", "", "def f():", "    ab = 1", "    return ab + n"),
            "n",
            "already a name",
            id="a-new-name-read-as-a-global",
        ),
        pytest.param(
            code("n = 3", "", "", "def f():", "    ab = 1", "    def g():", "        return n"),
            "n",
            "already a name",
            id="a-new-name-read-in-a-nested-function",
        ),
        pytest.param(
            code("def f(rows):", "    ab = rows", "    return len(ab)"),
            "len",
            "already a name",
            id="a-new-name-that-is-a-builtin-read",
        ),
        pytest.param(
            code("def f(rows):", "    ab = rows", "    return locals()['ab']"),
            "n",
            "by their spelling",
            id="locals",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    return eval('ab')"),
            "n",
            "by their spelling",
            id="eval",
        ),
        pytest.param(
            code("def f(rows):", "    ab = 1", "    return getattr(rows, 'ab')"),
            "n",
            "a token spells",
            id="getattr",
        ),
        pytest.param(
            code(
                "def f():", "    ab = 1", "    def g():", "        nonlocal ab", "        ab = 2"
            ),
            "n",
            "a token spells",
            id="a-nonlocal-statement",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    def g():", "        global ab", "    return ab"),
            "n",
            "a token spells",
            id="a-global-statement-in-a-nested-function",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    try:", "        pass", "    except OSError as ab:")
            + code("        pass"),
            "n",
            "a token spells",
            id="an-except-target",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    import ab"), "n", "a token spells", id="an-import"
        ),
        pytest.param(
            code("def f():", "    ab = 1", '    return f"{ab!r} {ab=}"'),
            "n",
            "a token spells",
            id="an-f-string-field-printing-its-own-source",
        ),
        pytest.param(
            code("def f(w):", "    ab = 1", '    return f"{ ab + max(ab, w) = :>{w}}"'),
            "n",
            "a token spells",
            id="an-f-string-field-printing-an-expression-of-it",
        ),
        pytest.param(
            code("def f():", "    ab = 1", '    return f"{f"{ab}"=}"'),
            "n",
            "a token spells",
            id="an-f-string-field-printing-a-nested-f-string",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    def ab():", "        pass"),
            "n",
            "a token spells",
            id="a-nested-definition",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    class C:", "        ab = 2", "    return ab"),
            "n",
            "no plain local",
            id="a-class-binding-it-too",
        ),
        pytest.param(
            code("def f():", "    ab = 1", "    return (lambda ab: ab, lambda: ab)"),
            "n",
            "no plain local",
            id="two-lambdas-the-tree-cannot-tell-apart",
        ),
        pytest.param("def f(:\n", "n", "does not parse", id="source-that-is-no-python"),
    ],
)
def test_a_local_rename_that_could_change_what_a_scope_reads_is_refused(
    source: str, to: str, why: str
) -> None:
    operation = RewriteReferences()

    edits = operation.edits(local_row(source, to=to))

    assert edits is None
    assert why in operation.why


@pytest.mark.parametrize(
    ("columns", "why"),
    [
        pytest.param({"to": "class"}, "no new name", id="a-keyword"),
        pytest.param({"to": "ab"}, "no new name", id="the-same-name"),
        pytest.param({"to": "a-b"}, "no new name", id="a-name-that-is-no-identifier"),
        pytest.param({"line": None}, "needs the line", id="a-function-scope-with-no-line"),
        pytest.param({"scope": "class"}, "malformed", id="a-scope-the-operation-does-not-know"),
    ],
)
def test_a_malformed_row_is_refused_before_anything_is_read(
    columns: dict[str, str | None], why: str
) -> None:
    operation = RewriteReferences()

    edits = operation.edits({**local_row(code("def f():", "    ab = 1")), **columns})

    assert edits is None
    assert why in operation.why
