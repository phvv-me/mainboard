"""A module's name is renamed where it is defined, in `__all__`, in each `from ... import` of it
with the reads that follow, and in each `module.name` read; a rename that could change what a
module reads is refused with its reason."""

from collections.abc import Iterable, Iterator

import pytest
from duckdb import DuckDBPyConnection

from mainboard.analysis.operations import RewriteReferences

from ...modules.readers import applied
from ..builders import code, facts_cursor, module_row
from .constants import REFUSALS, RENAMES, SOURCES


def _folder(case: str) -> str:
    """The package folder of `case`, under `src/pkg`."""
    return f"src/pkg/{case.replace('-', '_')}"


def _tree(case: str) -> dict[str, str]:
    """The files of `case` by path, each with the package files above it."""
    return {path: text for path, text in SOURCES.items() if path.startswith(_folder(case) + "/")}


def _after(case: str) -> dict[str, str]:
    """The files of `case` once the rename has changed the files it lists."""
    changed = {
        f"{_folder(case)}/{name}": text.replace("CASE", case.replace("-", "_"))
        for name, text in RENAMES[case].after.items()
    }
    return {**_tree(case), **changed}


@pytest.fixture(scope="module")
def facts() -> Iterator[DuckDBPyConnection]:
    """The facts of one scratch project holding every case in a package of its own."""
    with facts_cursor(SOURCES) as cursor:
        yield cursor


@pytest.mark.parametrize("case", list(RENAMES))
def test_a_module_name_rename_respells_importers_reads_and_all(
    case: str, facts: DuckDBPyConnection
) -> None:
    rename = RENAMES[case]
    row = module_row(facts, path=f"{_folder(case)}/a.py", name=rename.name, to=rename.to)

    edits = RewriteReferences().edits(row)

    assert edits is not None
    assert applied(_tree(case), edits) == _after(case)


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_module_name_rename_that_could_change_what_a_module_reads_is_refused(
    case: str, facts: DuckDBPyConnection
) -> None:
    refusal, operation = REFUSALS[case], RewriteReferences()
    path = f"{_folder(case)}/a.py"

    row = module_row(facts, path=path, name=refusal.name, to=refusal.to)

    assert operation.edits(row) is None
    assert refusal.why in operation.why


def _scanned(case: str, names: Iterable[str], **spelling: str) -> list[dict[str, str]]:
    """The `scanned` column holding the files `names` of `case` and the new ones of `spelling`."""
    files = {name: _tree(case)[f"{_folder(case)}/{name}"] for name in names} | spelling
    return [{"path": f"{_folder(case)}/{name}", "content": text} for name, text in files.items()]


@pytest.mark.parametrize(
    ("content", "why"),
    [
        pytest.param(
            code("from . import a", "", 'V = getattr(a, "_hidden")'),
            "spells `_hidden`",
            id="a-string-naming-it",
        ),
        pytest.param(
            code("from unittest import mock", "", 'mock.patch("pkg.a._hidden", 2)'),
            "spells `_hidden`",
            id="a-string-path-ending-in-it",
        ),
        pytest.param(
            code("from . import a", "", "def f(:"), "does not parse", id="a-file-that-is-no-python"
        ),
    ],
)
def test_a_python_file_the_rename_leaves_alone_must_spell_the_name_nowhere(
    content: str, why: str, facts: DuckDBPyConnection
) -> None:
    case, operation = "an-importer-and-its-reads", RewriteReferences()
    row = module_row(facts, path=f"{_folder(case)}/a.py", name="_hidden", to="hidden")

    edits = operation.edits({**row, "scanned": _scanned(case, [], **{"c.py": content})})

    assert edits is None
    assert why in operation.why


def test_files_the_rename_rewrites_and_other_variables_of_the_name_are_not_a_spelling_of_it(
    facts: DuckDBPyConnection,
) -> None:
    case, operation = "an-importer-and-its-reads", RewriteReferences()
    row = module_row(facts, path=f"{_folder(case)}/a.py", name="_hidden", to="hidden")
    unrelated = code("_hidden = 2", "V = obj._hidden", "W = 'hidden'")

    edits = operation.edits(
        {**row, "scanned": _scanned(case, ["a.py", "b.py"], **{"c.py": unrelated})}
    )

    assert edits is not None
    assert edits == operation.edits(row)


def test_a_read_through_a_name_and_spelled_in_full_is_renamed_once() -> None:
    sources = {
        "src/pkg/__init__.py": "_hidden = 1\n",
        "src/pkg/b.py": code("import pkg", "", "V = pkg._hidden"),
    }
    with facts_cursor(sources) as cursor:
        row = module_row(cursor, path="src/pkg/__init__.py", name="_hidden", to="hidden")

    edits = RewriteReferences().edits(row)

    assert edits is not None
    assert applied(sources, edits)["src/pkg/b.py"] == code("import pkg", "", "V = pkg.hidden")


@pytest.mark.parametrize(
    ("misstating", "why"),
    [
        pytest.param(
            {"statement": "from .a import other"},
            "does not import",
            id="a-statement-of-other-names",
        ),
        pytest.param(
            {"content": "def f(:\n"}, "does not parse", id="an-importer-that-is-no-python"
        ),
    ],
)
def test_a_row_that_misstates_an_importer_is_refused(
    misstating: dict[str, str], why: str, facts: DuckDBPyConnection
) -> None:
    case, operation = "an-importer-and-its-reads", RewriteReferences()
    path = f"{_folder(case)}/a.py"

    row = module_row(facts, path=path, name="_hidden", to="hidden", misstating=misstating)

    assert operation.edits(row) is None
    assert why in operation.why
