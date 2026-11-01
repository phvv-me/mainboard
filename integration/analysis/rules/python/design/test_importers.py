"""`move_row` is the whole `move-definitions` row for moving names between two modules: the rules
that move definitions (PY-EXCE0003, PY-ENUM0004, PY-MODE0001, PY-MODU0003, PY-TEST0001) and the
`refactor move` verb read it, and the verb installs it without asking any rule for it."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from mainboard.analysis.commands.tools import move
from mainboard.analysis.engine.lint import Lint, load_catalog, selected, verbs
from mainboard.analysis.rules.python.design.importers import MoveRow
from mainboard.analysis.rules.python.exceptions.core import SharedExceptionPlacement

from ....engine.builders import written

_PACKAGE = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}
_HELPER = "def helper() -> int:\n    return 1\n"
_ROW = "SELECT * FROM move_row($path, $names, $destination)"
_RULES_MOVING_DEFINITIONS = (
    "PY-ENUM0004",
    "PY-EXCE0003",
    "PY-MODE0001",
    "PY-MODU0003",
    "PY-TEST0001",
)


def _rows(files: dict[str, str], sql: str, **parameters: str | list[str]) -> list[dict]:
    """The rows of `sql` over the fact base of a scratch project holding `_PACKAGE` and `files`.

    The fact base carries `move_row` because a rule whose fix reads it is enabled.
    """
    assert MoveRow.name() == "move_row"
    load_catalog()
    with tempfile.TemporaryDirectory(prefix="mb-move-row-") as scratch:
        root = written(Path(scratch), {**_PACKAGE, **files})
        with Lint(root, (SharedExceptionPlacement,)) as lint:
            result = lint.facts(root).cursor(None).execute(sql, parameters)
            columns = [column for column, *_ in result.description]
            return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _moved(path: str, names: list[str], destination: str, files: dict[str, str]) -> list[dict]:
    return _rows(files, _ROW, path=path, names=names, destination=destination)


@settings(max_examples=12)
@example(source="src/pkg/a/__init__.py", folders=[], stem="helper", initializer=False)
@example(source="src/pkg/a.py", folders=["sub", "deep"], stem="helper", initializer=True)
@given(
    source=st.sampled_from(["src/pkg/a.py", "src/pkg/a/b.py", "src/pkg/a/__init__.py"]),
    folders=st.lists(st.sampled_from(["sub", "deep"]), max_size=2, unique=True),
    stem=st.sampled_from(["helper", "moved"]),
    initializer=st.booleans(),
)
def test_a_destination_that_does_not_exist_yet_is_named_by_its_path_for_any_source_layout(
    source: str, folders: list[str], stem: str, initializer: bool
) -> None:
    parts = ["src", "pkg", *folders, *([stem, "__init__.py"] if initializer else [f"{stem}.py"])]
    modules = [part.removesuffix(".py") for part in parts[1:] if part != "__init__.py"]

    rows = _moved(
        source, ["helper"], "/".join(parts), {source: _HELPER, **{p: "" for p in _stubs(source)}}
    )

    assert [row["destination_module"] for row in rows] == [".".join(modules)]


def _stubs(source: str) -> list[str]:
    """The package folders `source` sits in, each as an initializer a module needs to be found."""
    folders = Path(source).parent.parts[1:]
    return [str(Path("src", *folders[:n], "__init__.py")) for n in range(1, len(folders) + 1)]


@pytest.mark.parametrize(
    ("importer", "imports", "reads"),
    [
        pytest.param("from .a import helper\n\nVALUE = helper()\n", 1, 0, id="a-from-import"),
        pytest.param(
            "from . import a\n\nVALUE = a.helper()\n", 0, 1, id="a-read-through-the-module"
        ),
        pytest.param("from .a import other\n\nVALUE = other\n", 0, 0, id="a-name-that-stays"),
    ],
)
def test_the_row_lists_each_import_and_module_read_of_the_names_it_moves(
    importer: str, imports: int, reads: int
) -> None:
    files = {"src/pkg/a.py": _HELPER + "\n\nother = 2\n", "src/pkg/b.py": importer}

    rows = _moved("src/pkg/a.py", ["helper"], "src/pkg/c.py", files)

    assert [(len(row["importers"] or []), len(row["reached"] or [])) for row in rows] == [
        (imports, reads)
    ]


def test_a_path_that_is_no_project_module_gives_no_row() -> None:
    rows = _moved("src/pkg/missing.py", ["helper"], "src/pkg/c.py", {"src/pkg/a.py": _HELPER})

    assert rows == []


def test_a_macro_argument_qualified_by_an_outer_alias_keeps_its_correlation() -> None:
    files = {"src/pkg/a.py": _HELPER, "src/pkg/b.py": _HELPER}
    sql = (
        "SELECT d.path, r.module, r.destination_module FROM (VALUES "
        "('src/pkg/a.py', ['helper'], 'src/pkg/x.py'), "
        "('src/pkg/b.py', ['helper'], 'src/pkg/y.py')) AS d(path, names, destination), "
        "move_row(d.path, d.names, d.destination) r"
    )

    rows = _rows(files, sql)

    assert sorted((row["path"], row["module"], row["destination_module"]) for row in rows) == [
        ("src/pkg/a.py", "pkg.a", "pkg.x"),
        ("src/pkg/b.py", "pkg.b", "pkg.y"),
    ]


@pytest.mark.parametrize("rule", _RULES_MOVING_DEFINITIONS)
def test_every_example_of_a_rule_moving_definitions_passes(
    rule: str, capsys: pytest.CaptureFixture[str]
) -> None:
    verbs.test(selection=verbs.Selection(rule=(rule,)))

    assert "0 examples failed" in capsys.readouterr().out


def test_a_move_finds_its_row_when_no_enabled_rule_reads_it(tmp_path: Path) -> None:
    load_catalog()
    reading = [rule.id for rule in selected(()) if "move_row" in (rule.fix or "")]
    switches = "".join(f'\n[tool.mcmr.rules."{rule}"]\nenabled = false\n' for rule in reading)
    written(
        tmp_path,
        {
            **_PACKAGE,
            "pyproject.toml": _PACKAGE["pyproject.toml"] + switches,
            "src/pkg/errs.py": "class Boom(Exception):\n    pass\n",
            "src/pkg/use.py": "from .errs import Boom\n\n\ndef f():\n    raise Boom\n",
        },
    )

    code = move(
        ("src/pkg/errs.py::Boom",), to=Path("src/pkg/exceptions.py"), root=tmp_path, write=True
    )

    importer = (tmp_path / "src/pkg/use.py").read_text(encoding="utf-8")
    assert code == 0 and reading and importer.startswith("from .exceptions import Boom")
