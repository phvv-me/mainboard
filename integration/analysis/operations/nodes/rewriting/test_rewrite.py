"""The rewrite-node operation: a declared rewrite or the captures a row binds become one Replace of
the node, a comment or text the template would lose refuses it, and a rule that declares rewrites
has them as its fix."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import pytest

from mainboard.analysis import Rule
from mainboard.analysis.engine.lint import Selection, fix, load_catalog, verbs
from mainboard.analysis.operations.nodes import Node, Rewrite, RewriteNode, Site

from ....engine.builders import written
from .builders import declared, sources, swallows
from .readers import Rewritten

type Parse = Callable[[str], list[Node]]
type Plan = Callable[[Mapping], Rewritten]
type Gather = Callable[..., Site]

# The statements of `bounded.py`'s try: what stays in the try, what follows it and the handler.
_STATEMENTS = (("expression_statement", 3), ("return_statement", 4), ("except_clause", 5))
_CASTS = (Rewrite(match="$_($T, $X)", to="$X"),)
_GLOB = (Rewrite(match="$P.glob($$$A)", to="$P.rglob($$$A)"),)
_HEAD = "import shutil\nfrom contextlib import suppress\n\n\ndef clean(path):\n"
_BOUND = "try:\n    $$$KEEP\n$$$HANDLERS\n$$$TAIL"
_LOADED = """\
def load(path):
    try:
        text = path.read_text()
    except OSError:
        return None
    return parse(text)
"""
_TALL_CAST = """\
cast(
        int,
        x.method(
            1,
            2,
        ),
    )"""
_TALL_ARGUMENT = """\
x.method(
        1,
        2,
    )"""
_TORN_CAST = """\
cast(
        int,
        a
        if b
        else 0,
    )"""
_TORN_ARGUMENT = """\
(a
    if b
    else 0)"""
_PICKED = """\
from typing import cast


def f(a, b, c, d):
    return cast(int, a if b else c) if d else 0
"""
_SWALLOWED = "no declared pattern unifies"
_LOST = "a comment outside the captures the template keeps would be lost"
_UNMOVABLE = "a capture cannot stand where its hole does without changing its meaning"


def _suppressed(handler: str, *body: str) -> str:
    """The corpus function `clean` once rewritten: `with suppress(handler)` over the body lines."""
    lines = "".join(f"        {line}\n" for line in body)
    return f"{_HEAD}    with suppress({handler}):\n{lines}"


def _ids(site: Site, *spots: tuple[str, int]) -> tuple[int, ...]:
    """The ids of the nodes of the site's subtree of each type and line."""
    return tuple(node.node for node in site.tree if (node.type, node.line) in spots)


def _bound(site: Site, **captures: tuple[int, ...] | None) -> dict:
    """The fix row binding the captures to the template that hoists a try's tail out of it."""
    return {"site": site, "captures": captures, "template": _BOUND}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param(
            "swallow.py", _suppressed("OSError", "shutil.rmtree(path)"), id="one-handler"
        ),
        pytest.param(
            "tuple.py",
            _suppressed(
                "FileNotFoundError, PermissionError", "shutil.rmtree(path)", "path.unlink()"
            ),
            id="a-tuple-of-handlers-and-an-ellipsis",
        ),
        pytest.param(
            "bare.py", _suppressed("BaseException", "shutil.rmtree(path)"), id="a-bare-handler"
        ),
        pytest.param(
            "leading.py",
            _suppressed("OSError", "# note", "shutil.rmtree(path)"),
            id="a-comment-between-the-keyword-and-the-body",
        ),
        pytest.param(
            "closing.py",
            _suppressed("OSError", "shutil.rmtree(path)", "# after"),
            id="a-comment-after-the-last-statement",
        ),
        pytest.param(
            "between.py",
            _suppressed("OSError", "a = 1  # one", "# two", "b = 2  # three"),
            id="trailing-and-own-line-comments-between-statements",
        ),
        pytest.param(
            "multibyte.py",
            _suppressed("OSError", 'print("é日本")  # 日本'),
            id="multibyte-text-and-comment",
        ),
        pytest.param(
            "string.py",
            _suppressed("OSError", 'print("""a\nb""")'),
            id="a-string-with-a-line-break",
        ),
    ],
)
def test_a_declared_rewrite_turns_the_node_into_its_template_and_adds_the_import(
    site: Gather, parse: Parse, plan: Plan, path: str, expected: str
) -> None:
    rewritten = plan(declared(site(path, kind="try_statement"), swallows(), parse))

    assert (rewritten.text, rewritten.kinds) == (expected, ["replace", "import"])


def test_a_rewrite_keeps_the_line_ending_of_a_file_with_carriage_returns(
    site: Gather, parse: Parse, plan: Plan
) -> None:
    body = ("shutil.rmtree(path)", "path.unlink()")

    rewritten = plan(declared(site("crlf.py", kind="try_statement"), swallows(), parse))

    assert rewritten.text == _suppressed("OSError", *body).replace("\n", "\r\n")


@pytest.mark.parametrize(
    ("path", "why"),
    [
        pytest.param("busy.py", _SWALLOWED, id="a-handler-that-does-something"),
        pytest.param("orelse.py", _SWALLOWED, id="an-else-clause"),
        pytest.param("handler.py", _LOST, id="a-comment-on-the-handler-line"),
    ],
)
def test_a_declared_rewrite_that_cannot_keep_everything_refuses_and_leaves_the_file(
    site: Gather, parse: Parse, plan: Plan, path: str, why: str
) -> None:
    rewritten = plan(declared(site(path, kind="try_statement"), swallows(), parse))

    assert (rewritten.text, rewritten.kinds, rewritten.why) == (sources()[path], [], why)


@pytest.mark.parametrize(
    ("line", "old", "new"),
    [
        pytest.param(6, 'cast(int, x["a"])', 'x["a"]', id="a-cast-among-statements"),
        pytest.param(10, "typing.cast(int, a + b)", "(a + b)", id="under-a-binary-operator"),
        pytest.param(14, "cast(int, a or b)", "(a or b)", id="under-an-attribute"),
        pytest.param(18, "-cast(int, a + b)", "-(a + b)", id="under-a-unary-operator"),
        pytest.param(33, _TALL_CAST, _TALL_ARGUMENT, id="an-argument-spanning-lines"),
        pytest.param(43, _TORN_CAST, _TORN_ARGUMENT, id="an-argument-the-call-kept-on-lines"),
    ],
)
def test_a_lone_hole_rewrite_puts_the_second_argument_where_the_call_stood(
    site: Gather, parse: Parse, plan: Plan, line: int, old: str, new: str
) -> None:
    rewritten = plan(declared(site("casts.py", kind="call", line=line), _CASTS, parse))

    assert (rewritten.text, rewritten.kinds) == (
        sources()["casts.py"].replace(old, new),
        ["replace"],
    )


def test_a_comment_inside_the_call_refuses_a_rewrite_that_would_drop_it(
    site: Gather, parse: Parse, plan: Plan
) -> None:
    rewritten = plan(declared(site("casts.py", kind="call", line=22), _CASTS, parse))

    assert (rewritten.text, rewritten.why) == (sources()["casts.py"], _LOST)


@pytest.mark.parametrize(
    ("line", "kinds", "why"),
    [
        pytest.param(3, ["replace"], "", id="a-call-that-fits-the-pattern"),
        pytest.param(4, [], _UNMOVABLE, id="a-closing-comment-would-swallow-the-parenthesis"),
    ],
)
def test_a_run_ending_in_a_comment_leaves_no_template_text_on_its_line(
    site: Gather, parse: Parse, plan: Plan, line: int, kinds: list[str], why: str
) -> None:
    rewritten = plan(declared(site("globs.py", kind="call", line=line), _GLOB, parse))

    assert (rewritten.kinds, rewritten.why) == (kinds, why)


def test_a_row_bound_template_renders_the_nodes_its_sql_named(site: Gather, plan: Plan) -> None:
    found = site("bounded.py", kind="try_statement")
    keep, tail, handlers = (_ids(found, spot) for spot in _STATEMENTS)

    rewritten = plan(_bound(found, KEEP=keep, TAIL=tail, HANDLERS=handlers))

    assert (rewritten.text, rewritten.kinds) == (_LOADED, ["replace"])


def test_a_row_bound_template_refuses_text_its_captures_leave_out(
    site: Gather, plan: Plan
) -> None:
    found = site("bounded.py", kind="try_statement")
    tail, handlers = (_ids(found, spot) for spot in _STATEMENTS[1:])

    rewritten = plan(_bound(found, KEEP=None, TAIL=tail, HANDLERS=handlers))

    assert (rewritten.text, rewritten.why) == (
        sources()["bounded.py"],
        "the captures leave out `text`",
    )


@pytest.mark.parametrize(
    ("path", "expected", "why"),
    [
        pytest.param(
            "bounded.py",
            _LOADED.replace("    return parse(text)\n", ""),
            "",
            id="a-leaf-in-a-dropped-capture-goes",
        ),
        pytest.param("noted_tail.py", None, _LOST, id="a-comment-in-a-dropped-capture-refuses"),
    ],
)
def test_a_capture_the_template_leaves_out_is_dropped_with_its_leaves_but_never_its_comments(
    site: Gather, plan: Plan, path: str, expected: str | None, why: str
) -> None:
    found = site(path, kind="try_statement")
    keep, tail, handlers = (_ids(found, spot) for spot in _STATEMENTS)
    row = _bound(found, KEEP=keep, TAIL=tail, HANDLERS=handlers)

    rewritten = plan({**row, "template": "try:\n    $$$KEEP\n$$$HANDLERS"})

    assert (rewritten.text, rewritten.why) == (expected or sources()[path], why)


@pytest.mark.parametrize(
    ("captures", "reason"),
    [
        pytest.param(("a", "a", "h"), "overlap", id="captures-sharing-a-node"),
        pytest.param(("a",), "which no capture names", id="a-template-hole-no-capture-names"),
    ],
)
def test_a_row_whose_sql_binds_the_captures_wrongly_fails_loudly(
    site: Gather, captures: Sequence[str], reason: str
) -> None:
    found = site("bounded.py", kind="try_statement")
    ids = {"a": _ids(found, _STATEMENTS[0]), "h": _ids(found, _STATEMENTS[2])}
    bound = dict(zip(("KEEP", "TAIL", "HANDLERS"), (ids[name] for name in captures), strict=False))

    with pytest.raises(ValueError, match=reason):
        RewriteNode().edits(_bound(found, **bound))


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        pytest.param(
            {"rewrites": "[{"}, "the row's rewrites is malformed", id="rewrites-not-json"
        ),
        pytest.param(
            {"patterns": [[{"node": 1}]]}, "the row's patterns is malformed", id="a-bad-parse"
        ),
    ],
)
def test_a_row_the_operation_cannot_read_is_refused_naming_what_is_wrong(
    site: Gather, fields: dict, why: str
) -> None:
    operation = RewriteNode()

    edits = operation.edits({"site": site("swallow.py", kind="try_statement"), **fields})

    assert (edits, operation.why) == (None, why)


def test_a_row_naming_a_node_the_facts_no_longer_hold_is_refused() -> None:
    operation = RewriteNode()

    edits = operation.edits({"rewrites": "[]", "patterns": []})

    assert (edits, operation.why) == (None, "the node is gone")


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param("PY-TYPE0009", id="a-cast-call"),
        pytest.param("PY-CUDA0011", id="a-cache-key-and-a-digest"),
        pytest.param("PY-EXCE0002", id="a-try-guarding-too-much"),
        pytest.param("ALL-CALL0002", id="a-blocking-call-without-a-timeout"),
        pytest.param("PY-CUDA0003", id="a-launch-ignoring-the-stream-its-function-holds"),
        pytest.param("PY-CUDA0004", id="work-on-the-default-stream"),
        pytest.param("CU-LAUN0002", id="a-kernel-launch-on-the-default-stream"),
    ],
)
def test_every_example_of_the_rules_fixed_by_rewrites_passes(
    capsys: pytest.CaptureFixture[str], rule: str
) -> None:
    load_catalog()

    with suppress(SystemExit):
        verbs.test(selection=verbs.Selection(rule=(rule,)))

    said = capsys.readouterr().out.splitlines()
    assert {line.partition(": ")[2] for line in said[:-1]} == {"pass"}
    assert len(said) == len(Rule.defined[rule].examples) + 1


def test_a_rule_declaring_rewrites_fixes_by_the_sql_they_generate() -> None:
    load_catalog()
    cast = Rule.defined["PY-TYPE0009"]

    assert (cast.fixes(), cast.fix) == ("sql", Rewrite.fix(cast.rewrites))
    assert Rewrite.bindings(cast.rewrites).items() <= cast.constants.items()


def test_a_rule_declaring_both_fix_and_rewrites_is_refused_at_definition() -> None:
    with pytest.raises(TypeError, match="both `fix` and `rewrites`"):

        class Both(Rule):
            query = "SELECT 1"
            fix = "SELECT 1"
            rewrites = _CASTS


@pytest.mark.parametrize(
    ("source", "fixed"),
    [
        pytest.param(
            "import typing as t\n\n\ndef f(x):\n    return t.cast(int, t.cast(str, x))\n",
            "def f(x):\n    return x\n",
            id="nested-casts-go-in-two-passes",
        ),
        pytest.param(_PICKED, _PICKED, id="a-cast-under-a-conditional-stays"),
        pytest.param(
            "from typing import cast\n\n\ndef f(x):\n    return cast(typ=int, val=x)\n",
            "from typing import cast\n\n\ndef f(x):\n    return cast(typ=int, val=x)\n",
            id="a-cast-written-with-keywords-stays",
        ),
    ],
)
def test_a_second_fixpoint_run_over_a_rewritten_file_changes_nothing(
    tmp_path: Path, source: str, fixed: str
) -> None:
    written(tmp_path, {"pyproject.toml": "[tool.mcmr]\n", "module.py": source})
    selection = Selection(rule=("PY-TYPE0009",))
    fix(tmp_path, selection=selection, write=True, unsafe=True)
    once = (tmp_path / "module.py").read_text(encoding="utf-8")

    fix(tmp_path, selection=selection, write=True, unsafe=True)

    assert (once, (tmp_path / "module.py").read_text(encoding="utf-8")) == (fixed, fixed)
