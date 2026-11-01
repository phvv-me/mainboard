import ast
from collections.abc import Sequence

# A module under test: pytest, and a function the tests call that answers its arguments in a list.
_HEADER = "import pytest\n\n\ndef f(x, key=None):\n    return [x, key]\n\n\n"
_WORDS = ("one", "two", "three", "four", "five", "six")


def merged_test(
    argnames: Sequence[str], *, cases: Sequence[str], parameters: str, body: Sequence[str]
) -> str:
    """The test the tests of `trio` become, as `ast.unparse` writes it.

    argnames: the parameters the mark binds.
    cases: the values of each case, as `1, 'a'`.
    parameters: the parameter list of the test.
    body: the statements of the test.
    """
    names = repr(argnames[0]) if len(argnames) == 1 else repr(tuple(argnames))
    params = ", ".join(
        f"pytest.param({case}, id={word!r})" for case, word in zip(cases, _WORDS, strict=False)
    )
    lines = "".join(f"    {line}\n" for line in body)
    source = f"@pytest.mark.parametrize({names}, [{params}])\ndef test_f({parameters}):\n{lines}"
    return ast.unparse(ast.parse(source).body[0])


def named(names: Sequence[str], body: str = "    assert f({a}) == [{a}, None]") -> str:
    """A module of `f` and one test for each name, each `body` with its number `{a}` from 1.

    names: the names of the tests, in order.
    """
    return _HEADER + "\n\n".join(
        f"def {name}():\n{body.format(a=number)}\n" for number, name in enumerate(names, 1)
    )


def preamble() -> str:
    """What a module under test opens with: pytest, and the function its tests call."""
    return _HEADER


def suite(bodies: Sequence[str], *, params: str = "", deco: str = "") -> str:
    """A module of `f` and one test of each body, `test_f_one`, `test_f_two` and so on.

    bodies: the lines of each test, indented.
    params: the parameters of every test.
    deco: the decorator lines above every test, each ending in a line break.
    """
    return _HEADER + "\n\n".join(
        f"{deco}def test_f_{word}({params}):\n{body}\n"
        for word, body in zip(_WORDS, bodies, strict=False)
    )


def trio(body: str, *, params: str = "", deco: str = "") -> str:
    """A module of three tests, `body` giving each its number `{a}` (1 to 3) and `{b}` (2 to 4).

    body: the lines of each test, indented, and `{w}` for the word of its name.
    """
    return suite(
        [body.format(a=a, b=a + 1, w=_WORDS[a - 1]) for a in (1, 2, 3)], params=params, deco=deco
    )


def vectored(vectors: Sequence[tuple[str, str, str]]) -> str:
    """A module of one test for each vector of three literals, as source text.

    Each test asserts that `f(a, key=b)` answers `[c, b]`, so it passes when `c` equals `a`.
    """
    return suite([f"    assert f({a}, key={b}) == [{c}, {b}]" for a, b, c in vectors])
