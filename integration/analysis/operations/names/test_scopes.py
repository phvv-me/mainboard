"""`Scopes` states where a name binds and is read the way symtable does: a rename of a function's
local follows every scope reading it and none that binds its own, and it places a name in no scope
it cannot tell apart."""

import symtable
import textwrap

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.operations import RewriteReferences
from mainboard.analysis.operations.python.names import Scopes

from .builders import local_row, rewritten

_PATH = "m.py"
_FLAGS = (
    "is_local",
    "is_free",
    "is_global",
    "is_declared_global",
    "is_nonlocal",
    "is_parameter",
    "is_assigned",
    "is_referenced",
    "is_imported",
    "is_namespace",
)
_HEAD = "def f(rows, obj, call, deco):\n    ab = 0\n"
_TAIL = "    return ab\n"
_READS = (
    "x1 = ab + 1",
    "x2 = obj.ab",
    "x3 = call(ab=ab)",
    "x4 = f'{ab!r:>{ab}}'",
    "x5 = [ab * r for r in rows]",
    "x6 = [ab for ab in rows]",
    "x7 = sum(ab for _ in rows)",
    "x8 = sum(1 for ab in rows)",
    "x9 = lambda: ab",
    "x10 = lambda ab: ab",
    "x11 = lambda y=ab: y",
    "def g1():\n    return ab",
    "def g2(ab):\n    return ab",
    "def g3(y=ab, ab=2):\n    return y + ab",
    "def g4():\n    ab = 2\n    return ab",
    "def g5() -> ab:\n    return 1",
    "def g6():\n    def h():\n        return ab\n    return h",
    "def g7(y: ab = ab, *rest: ab, **more: ab):\n    return y",
    "@deco(ab)\ndef g8():\n    return 1",
    "ab += 1",
    "del ab",
    "if (y := ab):\n    pass",
    "for ab in rows:\n    pass",
    "with call(ab) as ab:\n    pass",
    "class C:\n    ab = 1",
    "try:\n    pass\nexcept OSError as ab:\n    pass",
    "def g9():\n    global ab",
    "def g10():\n    nonlocal ab",
    "import ab",
    "x12 = 'ab'",
)
_TAKEN = (
    "row_count = 1",
    "print(row_count)",
    "for row_count in rows:\n    pass",
    "import row_count",
    "def row_count():\n    pass",
    "global row_count",
    "z = [row_count for row_count in rows]",
    "z = lambda row_count: row_count",
    "z = lambda: row_count",
    "z = sum(row_count for row_count in rows)",
)
_WRAPS = (
    ("{}", 0),
    ("def k():\n{}", 1),
    ("def k():\n    def m():\n{}", 2),
    ("def k():\n    return ab\n{}", 1),
)


def _outline(table: symtable.SymbolTable, rename: tuple[str, str] | None, *, bound: bool) -> tuple:
    """A table, its symbols with their flags and the tables inside it, as symtable states them.

    With a `rename` of `(old, new)`, the symbols of the scopes reading the binding of function
    `f` are spelled `new`: `f` itself, and every table that reads or passes it on from a `bound`
    table around.
    """
    old, new = rename or ("", "")
    present = old in table.get_identifiers()
    here = bool(rename) and (
        table.get_name() == "f" or (bound and (not present or table.lookup(old).is_free()))
    )
    names = sorted(table.get_identifiers())
    flags = [tuple(getattr(table.lookup(name), flag)() for flag in _FLAGS) for name in names]
    symbols = [
        (new if here and name == old else name, flag)
        for name, flag in zip(names, flags, strict=True)
    ]
    inner = [_outline(child, rename, bound=here) for child in table.get_children()]
    return table.get_type().name, table.get_name(), table.get_lineno(), sorted(symbols), inner


def _renamed(source: str) -> str | None:
    """The function of `source` with local `ab` renamed `row_count`, None when it is refused."""
    edits = RewriteReferences().edits(local_row(source))
    return None if edits is None else rewritten(source, edits)


@st.composite
def _functions(draw: st.DrawFn) -> str:
    """A function binding `ab` and holding a few statements that read it in other scopes."""
    body = draw(st.lists(st.sampled_from(_READS), min_size=1, max_size=5))
    return _HEAD + textwrap.indent("\n".join(body), "    ") + "\n" + _TAIL


@st.composite
def _captures(draw: st.DrawFn) -> str:
    """A function binding `ab` with `row_count` taken in it, or in a scope inside it."""
    wrap, depth = draw(st.sampled_from(_WRAPS))
    taken = textwrap.indent(draw(st.sampled_from(_TAKEN)), "    " * depth)
    return _HEAD + textwrap.indent(wrap.format(taken), "    ") + "\n" + _TAIL


@settings(max_examples=250)
@example(source="def f(rows):\n    ab = len(rows)\n    return ab\n")
@example(source="def f(rows):\n    ab = 1\n    def g():\n        return ab + 1\n    return g()\n")
@example(source="def f(rows):\n    ab = 1\n    def g(ab):\n        return ab\n    return g(ab)\n")
@example(source="def f(rows):\n    ab = 2\n    return [ab * r for r in rows]\n")
@example(source="def f(rows):\n    ab = len(rows)\n    return f'{ab}'\n")
@given(source=_functions())
def test_renaming_a_local_keeps_every_scope_reading_the_same_binding(source: str) -> None:
    after = _renamed(source)
    assume(after is not None)

    shape = _outline(symtable.symtable(after or "", _PATH, "exec"), None, bound=False)
    expected = _outline(symtable.symtable(source, _PATH, "exec"), ("ab", "row_count"), bound=False)

    assert shape == expected


@settings(max_examples=120)
@example(source="def f(rows):\n    ab = 1\n    row_count = 1\n    return ab + row_count\n")
@given(source=_captures())
def test_a_rename_that_any_inner_scope_would_capture_is_refused(source: str) -> None:
    operation = RewriteReferences()

    edits = operation.edits(local_row(source))

    assert edits is None
    assert "already a name" in operation.why


@pytest.mark.parametrize(
    ("source", "name"),
    [
        pytest.param("def f(ab):\n    return ab\n", "ab", id="a-parameter"),
        pytest.param("def f():\n    global ab\n    ab = 1\n", "ab", id="a-name-declared-global"),
        pytest.param(
            "def g():\n    ab = 1\n    def f():\n        nonlocal ab\n        ab = 2\n",
            "ab",
            id="a-nonlocal",
        ),
        pytest.param(
            "def g():\n    ab = 1\n    def f():\n        return ab\n", "ab", id="a-free-read"
        ),
        pytest.param("def f():\n    return 1\n", "ab", id="a-name-the-function-never-mentions"),
        pytest.param("def f():\n    return len\n", "len", id="a-builtin-read"),
        pytest.param(
            "def f():\n    ab = 1\n    class C:\n        ab = 2\n",
            "ab",
            id="a-class-binding-it-too",
        ),
        pytest.param(
            "def f():\n    ab = 1\n    return (lambda ab: ab, lambda: ab)\n",
            "ab",
            id="two-lambdas-on-one-line-the-tree-cannot-tell-apart",
        ),
    ],
)
def test_a_name_that_is_no_plain_local_has_no_spans(source: str, name: str) -> None:
    scopes = Scopes(source, path=_PATH)
    line = source.splitlines().index(next(s for s in source.splitlines() if "def f" in s)) + 1

    assert scopes.local_spans(line, name) is None


def test_spans_count_their_columns_in_bytes_not_characters() -> None:
    source = "def f(rows):\n    é = 'é'; ab = len(rows)\n    return f'é{ab}é'\n"
    scopes = Scopes(source, path=_PATH)

    spans = scopes.local_spans(1, "ab") or []

    starts = [(span.start_line, span.start_column) for span in spans]
    lines = source.encode().split(b"\n")
    assert starts == [(2, lines[1].index(b"ab")), (3, lines[2].index(b"ab"))]


@pytest.mark.parametrize(
    ("source", "new", "taken"),
    [
        pytest.param("def f():\n    ab = 1\n    new = 2\n", "new", True, id="a-local"),
        pytest.param("def f():\n    ab = 1\n    return new\n", "new", True, id="a-global-read"),
        pytest.param("def f():\n    ab = 1\n    return len\n", "len", True, id="a-builtin-read"),
        pytest.param(
            "def f():\n    ab = 1\n    def g():\n        return new\n",
            "new",
            True,
            id="a-nested-read",
        ),
        pytest.param(
            "def f():\n    ab = 1\n    def new():\n        pass\n", "new", True, id="a-nested-name"
        ),
        pytest.param(
            "def f():\n    ab = 1\n    return [new for new in ()]\n",
            "new",
            True,
            id="a-comprehension-variable",
        ),
        pytest.param(
            "new = 1\n\n\ndef f():\n    ab = 1\n", "new", False, id="a-global-never-read"
        ),
        pytest.param(
            "def f():\n    ab = 1\n\n\ndef g():\n    new = 1\n",
            "new",
            False,
            id="a-sibling-function",
        ),
    ],
)
def test_a_scope_clashes_with_a_name_anywhere_inside_it_and_nowhere_beside_it(
    source: str, new: str, taken: bool
) -> None:
    line = next(n for n, text in enumerate(source.splitlines(), 1) if text.startswith("def f"))

    assert Scopes(source, path=_PATH).clash(line, new) is taken


@pytest.mark.parametrize(
    ("source", "reflective"),
    [
        pytest.param("def f():\n    return locals()\n", True, id="locals"),
        pytest.param("def f():\n    return vars()\n", True, id="vars"),
        pytest.param("def f():\n    return eval('1')\n", True, id="eval"),
        pytest.param("def f():\n    exec('x = 1')\n", True, id="exec"),
        pytest.param("def f():\n    return frame.f_locals\n", True, id="a-frame-s-locals"),
        pytest.param(
            "def f():\n    def g():\n        return locals()\n", True, id="inside-a-nested-scope"
        ),
        pytest.param("def f():\n    return dir\n", False, id="a-name-that-reads-nothing"),
    ],
)
def test_a_scope_reading_its_names_by_spelling_is_reflective(
    source: str, reflective: bool
) -> None:
    assert Scopes(source, path=_PATH).reflective(1) is reflective


@pytest.mark.parametrize(
    ("source", "line", "imported"),
    [
        pytest.param("from a import b\n", 1, True, id="a-statement-at-the-top"),
        pytest.param("if X:\n    from a import b\n", 2, True, id="a-statement-in-an-if"),
        pytest.param(
            "try:\n    pass\nexcept E:\n    from a import b\n", 4, True, id="in-a-handler"
        ),
        pytest.param("def f():\n    from a import b\n", 2, False, id="a-statement-in-a-function"),
        pytest.param("class C:\n    from a import b\n", 2, False, id="a-statement-in-a-class"),
        pytest.param("from a import b\n", 2, False, id="a-line-with-no-statement"),
    ],
)
def test_a_from_import_binds_in_the_module_only_outside_definitions(
    source: str, line: int, imported: bool
) -> None:
    assert Scopes(source, path=_PATH).imported(line) is imported


@pytest.mark.parametrize(
    ("source", "spelled"),
    [
        pytest.param("def f():\n    ab = 1\n    return ab\n", True, id="every-token-a-variable"),
        pytest.param(
            "def f(obj):\n    ab = obj.ab\n    return g(ab=ab)\n",
            True,
            id="an-attribute-and-a-keyword",
        ),
        pytest.param(
            "def f(obj):\n    ab = 1\n    return obj['ab']\n", False, id="a-string-naming-it"
        ),
        pytest.param(
            "def f(obj):\n    ab = 1\n    return getattr(obj, 'pkg.mod.ab')\n",
            False,
            id="a-path-ending-in-it",
        ),
        pytest.param(
            "def f(obj):\n    ab = 1\n    return 'about ab'\n",
            True,
            id="a-string-merely-mentioning-it",
        ),
        pytest.param(
            "def f():\n    ab = 1\n    def g():\n        global ab\n    return ab\n",
            False,
            id="a-global-statement-the-tree-places-nothing-for",
        ),
        pytest.param("def f():\n    ab = 1\n    import x as ab\n", False, id="an-import-alias"),
        pytest.param("def f():\n    ab = 1  # ab\n    return ab\n", True, id="a-comment"),
    ],
)
def test_every_token_spelling_the_name_is_a_variable_or_another_name_the_tree_places(
    source: str, spelled: bool
) -> None:
    scopes = Scopes(source, path=_PATH)

    spans = scopes.local_spans(1, "ab") or []

    assert scopes.spelled(spans, "ab", line=1) is spelled
