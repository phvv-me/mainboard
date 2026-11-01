"""ALL-NAMI0001: the model names a short Python local and the rename follows every read of it; it
is left alone when the rename cannot be made whole, and a model that gives no usable name is
trusted with nothing. Every test runs with the model double of `fakes.py` first on PATH, and
each asks about a function of its own, since the engine keeps a model's answers by question."""

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from mainboard.analysis import Hit
from mainboard.analysis.engine.lint import Selection, fix, load_catalog, verbs
from mainboard.analysis.rules.general.writing.naming import UninformativeLocalName

from ....engine.builders import written
from .constants import CLASHES, NAMED, PAIRED, RENAMES, TYPESCRIPT, UNFOLLOWED
from .fakes import Claude

_PROJECT = {
    "pyproject.toml": '[project]\nname = "pkg"\nversion = "0"\n',
    "src/pkg/__init__.py": "",
}
_PAIRED = "two-locals-of-one-function"
_TYPED = "src/pkg/typed.ts"


class Fixed(NamedTuple):
    """The files of every case once the rule's fix ran, and what the model was asked."""

    files: dict[str, str]
    questions: list[str]


def _path(case: str) -> str:
    """The file holding the function of `case`."""
    return f"src/pkg/{case.replace('-', '_')}.py"


def _asked(fixed: Fixed, source: str) -> list[str]:
    """The questions that quote the first `def` line of `source`."""
    marker = next(line for line in source.splitlines() if "def " in line)
    return [question for question in fixed.questions if marker in question]


@pytest.fixture(scope="module")
def fixed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fixed]:
    """Every case in a file of one project, fixed once by a model double naming all alike."""
    sources = {
        **{case: before for case, (before, _) in RENAMES.items()},
        **UNFOLLOWED,
        **CLASHES,
        _PAIRED: PAIRED[0],
    }
    files = {_path(case): source for case, source in sources.items()} | {_TYPED: TYPESCRIPT}
    folder = tmp_path_factory.mktemp("naming")
    with pytest.MonkeyPatch.context() as patch:
        claude = Claude(folder / "double", monkeypatch=patch, names=(NAMED,))
        root = written(folder / "project", {**_PROJECT, **files})
        load_catalog()
        fix(root, selection=Selection(rule=(UninformativeLocalName.id,)), write=True, unsafe=True)
        yield Fixed(
            {name: (root / name).read_text(encoding="utf-8") for name in files}, claude.questions
        )


@pytest.mark.parametrize("case", list(RENAMES))
def test_a_short_local_is_renamed_to_the_models_name_with_every_read_of_it(
    fixed: Fixed, case: str
) -> None:
    assert fixed.files[_path(case)] == RENAMES[case][1]


def test_two_locals_of_one_function_are_renamed_one_at_a_time_and_never_to_one_name(
    fixed: Fixed,
) -> None:
    assert fixed.files[_path(_PAIRED)] in PAIRED[1]


@pytest.mark.parametrize("case", list(UNFOLLOWED))
def test_a_local_the_rename_cannot_follow_is_left_and_the_model_is_not_asked(
    fixed: Fixed, case: str
) -> None:
    assert fixed.files[_path(case)] == UNFOLLOWED[case]
    assert not _asked(fixed, UNFOLLOWED[case])


@pytest.mark.parametrize("case", list(CLASHES))
def test_a_name_the_function_already_uses_is_refused_after_the_model_gave_it(
    fixed: Fixed, case: str
) -> None:
    assert fixed.files[_path(case)] == CLASHES[case]
    assert _asked(fixed, CLASHES[case])


def test_a_local_of_another_language_is_left_and_the_model_is_not_asked(fixed: Fixed) -> None:
    assert fixed.files[_TYPED] == TYPESCRIPT
    assert not [question for question in fixed.questions if "function typed" in question]


@pytest.mark.parametrize(
    ("function", "names", "status"),
    [
        pytest.param("silent", (), 1, id="a-model-that-gives-no-answer"),
        pytest.param("alike", ("ab",), 0, id="a-name-as-short-as-the-local"),
        pytest.param("brief", ("xy",), 0, id="a-name-too-short-to-say-anything"),
        pytest.param("spaced", ("row count",), 0, id="an-answer-that-is-no-identifier"),
        pytest.param("reserved", ("class",), 0, id="a-keyword-for-a-name"),
    ],
)
def test_a_model_that_gives_no_usable_name_changes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    function: str,
    names: tuple[str, ...],
    status: int,
) -> None:
    claude = Claude(tmp_path, monkeypatch=monkeypatch, names=names, status=status)
    source = f"def {function}(rows):\n    ab = len(rows)\n    return ab\n"
    row = {
        "path": _path(function),
        "content": source,
        "name": "ab",
        "qualname": function,
        "def_line": 1,
        "end_line": 3,
    }

    edits = UninformativeLocalName().repair(Hit(**row))

    assert edits is None
    assert len(claude.questions) == 1


def test_every_example_of_the_local_naming_rule_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    claude = Claude(tmp_path, monkeypatch=monkeypatch, names=(NAMED,))

    verbs.test(selection=verbs.Selection(rule=(UninformativeLocalName.id,)))

    assert "0 examples failed" in capsys.readouterr().out
    assert not claude.questions
