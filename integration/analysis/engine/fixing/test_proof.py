"""Whether the tests a kept component leaves collected are the ones its fixes account for. pytest
lists the ids of the test files in the project and in the mirror holding the change: a merge keeps
each test as a case, a deletion removes the tests its change line names, and every other fix
leaves each test function, whatever cases it has. A component that differs is refused with how."""

from pathlib import Path

import pytest

from mainboard.analysis.engine.fixing.proof import Collection

from ..builders import written
from .builders import applied, module

_AT = "test_t.py::"
_CASES = ("one", "two", "three")
_MARK = '@pytest.mark.parametrize("case_0", [{cases}])\ndef test_f(case_0):\n    assert case_0\n'


def _merged(*names: str) -> str:
    """A module whose one test has a case for each of `names`."""
    cases = ", ".join(f'pytest.param({number}, id="{name}")' for number, name in enumerate(names))
    return "import pytest\n\n\n" + _MARK.format(cases=cases)


_THREE = module("test_f_one", "test_f_two", "test_f_three")
_MERGE = ", ".join(f"{_AT}test_f_{name} -> {_AT}test_f[{name}]" for name in _CASES)
_BARE = ", ".join(f"test_f_{name} -> test_f[{name}]" for name in _CASES)


@pytest.mark.parametrize(
    ("before", "after", "change", "moves", "why"),
    [
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": _merged(*_CASES)},
            f"parametrized {_MERGE}",
            (),
            [],
            id="a-merge-that-keeps-each-test-as-a-case",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": _merged(*_CASES)},
            f"parametrized {_BARE}",
            (),
            [],
            id="a-merge-that-names-its-tests-by-function",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": _merged("one", "two")},
            f"parametrized {_BARE}",
            (),
            [f"lose 1 test ids ({_AT}test_f[three])"],
            id="a-merge-by-function-that-drops-a-case",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": module("test_f_one", "test_f_three")},
            f"removed the duplicates {_AT}test_f_two -> gone, kept {_AT}test_f_one",
            (),
            [],
            id="a-deletion-that-names-the-test-it-removes",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": module("test_f_one", "test_f_three")},
            "deleted 1 span (duplicate test intent cluster count)",
            (),
            [f"lose 1 test ids ({_AT}test_f_two)"],
            id="a-deletion-that-names-nothing",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": _merged("one", "two")},
            f"parametrized {_MERGE}",
            (),
            [f"lose 1 test ids ({_AT}test_f[three])"],
            id="a-merge-that-drops-a-case",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": module("test_f_one", "test_f_two", "test_f_three", "test_f_four")},
            "rewrote 1 span (text around the logger)",
            (),
            [f"gain 1 test ids ({_AT}test_f_four)"],
            id="a-test-no-fix-said-it-added",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": "def test_f_one(:\n"},
            "rewrote 1 span (text around the logger)",
            (),
            ["no longer import: test_t.py"],
            id="a-file-the-fix-leaves-unimportable",
        ),
        pytest.param(
            {"test_t.py": "def test_f_one(:\n"},
            {"test_t.py": module("test_f_one")},
            "rewrote 1 span (text around the logger)",
            (),
            [],
            id="a-file-the-project-could-not-import-already",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": _THREE.replace("assert True", "assert not False")},
            "rewrote 3 spans (text around the logger)",
            (),
            [],
            id="a-rewrite-that-says-nothing-and-keeps-each-id",
        ),
        pytest.param(
            {"pkg/test_t.py": _THREE},
            {"lib/test_t.py": _THREE},
            "renamed 1 path (non ascii source path)",
            (("pkg", "lib"),),
            [],
            id="a-folder-the-fix-moved-with-its-tests",
        ),
        pytest.param(
            {"test_t.py": module("test_f")},
            {"test_t.py": _merged(*_CASES)},
            "rewrote 1 span (a loop over cases)",
            (),
            [],
            id="a-test-that-became-parametrized-keeps-its-function",
        ),
        pytest.param(
            {"test_t.py": _merged(*_CASES)},
            {"test_t.py": module("test_f")},
            "rewrote 1 span (parametrize that is a loop)",
            (),
            [],
            id="a-test-that-lost-its-cases-keeps-its-function",
        ),
        pytest.param(
            {"test_t.py": module("test_f")},
            {"test_t.py": _merged("asyncio")},
            "rewrote 1 span (anyio test)",
            (),
            [],
            id="a-test-run-per-backend-keeps-its-function",
        ),
        pytest.param(
            {"test_t.py": _merged("One_Value", "Two (2)")},
            {"test_t.py": _merged("one-value", "two-2")},
            "rewrote 2 spans (case ids in kebab case)",
            (),
            [],
            id="a-case-id-restyled-keeps-its-function",
        ),
        pytest.param(
            {"test_t.py": module("test_f")},
            {"test_t.py": _merged(*_CASES) + "\n\n" + module("test_g")},
            "rewrote 1 span (a loop over cases)",
            (),
            [f"gain 1 test ids ({_AT}test_g)"],
            id="a-function-added-while-parametrizing",
        ),
        pytest.param(
            {"test_t.py": _THREE},
            {"test_t.py": module("test_g")},
            f"parametrized {_MERGE}",
            (),
            [
                f"lose 3 test ids ({_AT}test_f[one], {_AT}test_f[three], {_AT}test_f[two]); "
                f"gain 1 test ids ({_AT}test_g)"
            ],
            id="a-merge-that-lost-its-function",
        ),
    ],
)
def test_the_test_ids_a_component_leaves_are_the_ones_its_fixes_account_for(
    tmp_path: Path,
    before: dict[str, str],
    after: dict[str, str],
    change: str,
    moves: tuple[tuple[str, str], ...],
    why: list[str],
) -> None:
    root, mirror = written(tmp_path / "project", before), written(tmp_path / "mirror", after)
    files = sorted(set(before) | set(after))
    fix = applied(change, files=files, moves=moves)

    broken = Collection(root).broken([set(files)], fixes=[fix], proven=mirror)

    assert [reason for _, reason in broken] == why


def test_a_component_is_judged_apart_from_the_others_and_the_listing_counts_what_it_found(
    tmp_path: Path,
) -> None:
    root = written(
        tmp_path / "project",
        {"test_a.py": module("test_x", "test_y"), "test_b.py": module("test_z")},
    )
    mirror = written(
        tmp_path / "mirror", {"test_a.py": module("test_x"), "test_b.py": module("test_z")}
    )
    fixes = [
        applied("deleted 1 span", files=["test_a.py"]),
        applied("rewrote", files=["test_b.py"]),
    ]
    collection = Collection(root)

    broken = collection.broken([{"test_a.py"}, {"test_b.py"}], fixes=fixes, proven=mirror)

    assert broken == [({"test_a.py"}, "lose 1 test ids (test_a.py::test_y)")]
    assert (collection.judged, collection.listed, collection.said) == (2, 2, "")


def test_a_project_pytest_cannot_list_judges_no_fix_and_says_why(tmp_path: Path) -> None:
    settings = {"pyproject.toml": "[tool.pytest.ini_options]\nstrict_config = true\nnope = 1\n"}
    root = written(tmp_path / "project", {**settings, "test_t.py": _THREE})
    mirror = written(tmp_path / "mirror", {**settings, "test_t.py": module("test_f_one")})
    collection = Collection(root)

    broken = collection.broken(
        [{"test_t.py"}], fixes=[applied("deleted 1 span", files=["test_t.py"])], proven=mirror
    )

    assert broken == []
    assert collection.said.startswith("pytest ended with 4")
    assert collection.judged == 0
