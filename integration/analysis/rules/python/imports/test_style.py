"""PY-IMPO0002: what a project imports from a nonpublic name or module is made public, a name by
`rewrite-references` and a module file by `move-modules`, and left alone when the change cannot be
made whole."""

from collections.abc import Mapping

import pytest

from mainboard.analysis.engine.lint import Selection, fix, load_catalog, verbs
from mainboard.analysis.rules.python.imports.style import ProjectPrivateImport

from ....engine.builders import written
from .builders import tree
from .constants import MOVES, REFUSALS, RENAMES

_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}
_CASES = {
    **{case: files.before for case, files in (RENAMES | MOVES).items()},
    **REFUSALS,
}


def _folder(case: str) -> str:
    """The package folder holding `case`, under `src/pkg`."""
    return f"src/pkg/{case.replace('-', '_')}"


def _spelled(case: str, files: Mapping[str, str]) -> dict[str, str]:
    """`files` of `case` with the folder's name where they say `CASE`, and an initializer."""
    named = {name: text.replace("CASE", case.replace("-", "_")) for name, text in files.items()}
    return {"__init__.py": "", **named}


@pytest.fixture(scope="module")
def fixed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, str]]:
    """Every case in a package of its own, fixed once; the files of each by path in its folder."""
    files = {
        f"{_folder(case)}/{name}": text
        for case, before in _CASES.items()
        for name, text in _spelled(case, before).items()
    }
    root = written(tmp_path_factory.mktemp("private"), {**_PROJECT, **files})
    load_catalog()
    fix(root, selection=Selection(rule=(ProjectPrivateImport.id,)), write=True, unsafe=True)
    return {case: tree(root / _folder(case)) for case in _CASES}


@pytest.mark.parametrize("case", list(RENAMES | MOVES))
def test_what_the_project_imports_from_a_nonpublic_name_or_module_is_made_public(
    fixed: Mapping[str, Mapping[str, str]], case: str
) -> None:
    assert fixed[case] == _spelled(case, (RENAMES | MOVES)[case].after)


@pytest.mark.parametrize("case", list(REFUSALS))
def test_an_import_whose_change_cannot_be_made_whole_is_left_as_it_is(
    fixed: Mapping[str, Mapping[str, str]], case: str
) -> None:
    assert fixed[case] == _spelled(case, REFUSALS[case])


def test_every_example_of_the_private_import_rule_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    verbs.test(selection=verbs.Selection(rule=(ProjectPrivateImport.id,)))

    said = capsys.readouterr().out.splitlines()
    examples = range(1, len(ProjectPrivateImport.examples) + 1)
    passes = [f"{ProjectPrivateImport.id} example {number}: pass" for number in examples]
    assert said == [*passes, "0 examples failed"]
