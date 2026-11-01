from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.analysis.catalog import Bundle
from mainboard.analysis.constants import DECLARED, FIXABLE
from mainboard.analysis.edits import Append
from mainboard.analysis.engine.lint import Lint, Selection, load_catalog
from mainboard.analysis.engine.lint import test as run_examples
from mainboard.analysis.rules.general.bundles import Ceiling
from mainboard.analysis.rules.general.dependencies.modules import (
    ModuleMemberCount,
    ModuleStatementCount,
)

from ..engine.builders import written
from ..project.fakes import Importers
from .builders import gitignored, staged
from .fakes import Paths

_NAMES = st.sampled_from(["src", "pkg", "mod.py", "test.py", "a", "b"])
_LINES = st.builds(
    lambda negation, lead, names, tail: negation + lead + "/".join(names) + tail,
    st.sampled_from(["", "!"]),
    st.sampled_from(["", "/", "**/"]),
    st.lists(_NAMES, min_size=1, max_size=3),
    st.sampled_from(["", "/", "/*", "/**"]),
)
_PATHS = st.lists(_NAMES, min_size=1, max_size=4).map("/".join)
_ROWS = st.lists(_PATHS, max_size=3)
_TWO_FUNCTIONS = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
_LINES_TABLE = '[tool.mcmr.rules."ALL-CEIL0001".module_lines]\nmaximum = 3\n'
_MEMBERS_TABLE = '[tool.mcmr.rules."ALL-CEIL0001".module_members]\nmaximum = 1\n'
_STATEMENTS_TABLE = '[tool.mcmr.rules."ALL-CEIL0001".module_statements]\nmaximum = 3\n'
_EXCLUDED = 'exclude = ["module.py"]\n'
_FIX = "SELECT finding_key, path, 'mended' AS text FROM finding"
_MEMBERS_FIX = (
    "SELECT finding_key, 'append' AS operation, path, "
    "'# members ' || CAST(value AS INTEGER) || chr(10) AS text FROM finding"
)
_STATEMENTS_FIX = (
    "SELECT finding_key, 'append' AS operation, path, "
    "'# statements ' || statement_count || chr(10) AS text FROM finding"
)


class _Trio(Bundle):
    """Three parts that report the paths they are given."""

    first: Paths = Paths()
    second: Paths = Paths()
    third: Paths = Paths()


@pytest.fixture
def mending_ceiling(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[Ceiling]]:
    """`Ceiling` with a fix on its module members and module statements parts."""
    monkeypatch.setattr(Ceiling, "applicability", "review")
    monkeypatch.setattr(ModuleMemberCount, "fix", _MEMBERS_FIX)
    monkeypatch.setattr(ModuleStatementCount, "fix", _STATEMENTS_FIX)
    yield Ceiling


@example(lines=["src", "!pkg", "src"], path="src/pkg")
@example(lines=["!src/**", "src"], path="src/x")
@given(lines=st.lists(_LINES, max_size=4), path=_PATHS)
def test_a_part_leaves_out_exactly_the_paths_its_exclude_lines_leave_out_in_gitignore(
    lines: list[str], path: str
) -> None:
    part = Paths(paths=(path,), exclude=tuple(lines))

    kept = duckdb.execute(
        part.branch("part"), {"part__paths": [path], "part__excluded": part.excluded()}
    ).fetchall()

    assert bool(kept) != gitignored(lines, path)


def test_a_bundle_binds_each_part_setting_under_the_part_name() -> None:
    bundle = Ceiling.model_validate(
        {"module_lines": {"maximum": 7}, "module_members": {"maximum": 3}}
    )

    bound = bundle.rows().parameters

    assert (bound["module_lines__maximum"], bound["module_members__maximum"]) == (7, 3)


@pytest.mark.parametrize(
    ("pyproject", "reported"),
    [
        pytest.param(_LINES_TABLE, 1, id="a-ceiling-below-the-module-reports-it"),
        pytest.param(
            _LINES_TABLE + _EXCLUDED, 0, id="an-exclude-line-silences-the-part-that-holds-it"
        ),
        pytest.param(
            _LINES_TABLE + 'exclude = ["*.py", "!module.py"]\n',
            1,
            id="a-negated-exclude-line-re-includes-its-path",
        ),
        pytest.param(
            _LINES_TABLE + _EXCLUDED + _MEMBERS_TABLE,
            1,
            id="an-exclude-line-leaves-the-other-parts-reporting",
        ),
    ],
)
def test_a_part_table_under_the_bundle_configures_only_that_part(
    tmp_path: Path, pyproject: str, reported: int
) -> None:
    (tmp_path / "module.py").write_text(_TWO_FUNCTIONS)
    (tmp_path / "pyproject.toml").write_text(pyproject)
    load_catalog()

    with Lint(tmp_path, (Ceiling,), fixing=False) as lint:
        found = lint.findings(lint.facts(tmp_path))["ALL-CEIL0001"]

    assert len(found) == reported


@pytest.mark.parametrize(
    "rule",
    [
        "ALL-CEIL0001",
        "ALL-CONF0002",
        "ALL-MANU0016",
        "ALL-MANU0017",
        "ALL-OVER0011",
        "PY-CEIL0001",
        "PY-RUFF0002",
    ],
)
def test_the_examples_of_a_pruned_rule_are_flagged_and_their_good_forms_are_not(rule: str) -> None:
    load_catalog()

    run_examples(selection=Selection(rule=(rule,)))


@given(first=_ROWS, second=_ROWS, third=_ROWS)
def test_two_parts_never_share_a_finding_key_and_finding_rows_numbers_every_row_they_report(
    first: list[str], second: list[str], third: list[str]
) -> None:
    bundle = _Trio(
        first=Paths(paths=tuple(first)),
        second=Paths(paths=tuple(second)),
        third=Paths(paths=tuple(third)),
    )

    with staged(bundle) as connection:
        numbered = connection.execute("SELECT * FROM finding_rows").fetchall()
        reported = connection.execute(bundle.sql(), bundle.rows().parameters).fetchall()

    keys = [key for key, *_ in numbered]
    assert len(set(keys)) == len(keys) == len(first) + len(second) + len(third)
    assert sorted(row[1:5] for row in numbered) == sorted(reported)


@pytest.mark.parametrize(
    ("fix", "fixes", "tables"),
    [
        pytest.param(None, "none", (), id="parts-that-only-report"),
        pytest.param(
            _FIX,
            "parts",
            ("finding_rows__first", "finding_rows__second", "finding_rows__third"),
            id="parts-that-fix",
        ),
    ],
)
def test_a_bundle_fixes_through_its_parts_when_any_part_declares_a_fix(
    monkeypatch: pytest.MonkeyPatch, fix: str | None, fixes: str, tables: Sequence[str]
) -> None:
    monkeypatch.setattr(Paths, "fix", fix)

    assert (_Trio.fixes(), tuple(_Trio().fixers)) == (fixes, tuple(tables))


@pytest.mark.parametrize(
    ("pyproject", "appended"),
    [
        pytest.param(
            _MEMBERS_TABLE + _STATEMENTS_TABLE,
            ("# members 2\n", "# statements 4\n"),
            id="each-part-fixes-from-its-own-rows",
        ),
        pytest.param(
            _MEMBERS_TABLE + _EXCLUDED + _STATEMENTS_TABLE,
            ("# statements 4\n",),
            id="an-exclude-line-leaves-its-part-out-of-the-fix",
        ),
    ],
)
def test_a_bundle_repairs_each_finding_through_the_part_that_reported_it(
    tmp_path: Path, mending_ceiling: type[Ceiling], pyproject: str, appended: Sequence[str]
) -> None:
    (tmp_path / "module.py").write_text(_TWO_FUNCTIONS)
    (tmp_path / "pyproject.toml").write_text(pyproject)
    load_catalog()

    with Lint(tmp_path, (mending_ceiling,)) as lint:
        repairs = lint.repairs(lint.facts(tmp_path), FIXABLE["review"])[mending_ceiling.id]

    assert [list(repair) for repair in repairs] == [
        [Append(path="module.py", text=text)] for text in appended
    ]


@pytest.mark.parametrize(
    ("imported", "repaired"),
    [
        pytest.param((), ("a.py", "b.py"), id="no-other-repository-imports-a-name"),
        pytest.param(("a.py",), ("b.py",), id="another-repository-imports-a-name-a-part-reported"),
    ],
)
def test_a_bundle_declaring_names_draws_up_no_fix_in_any_part_for_a_name_another_repository_uses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mending_ceiling: type[Ceiling],
    imported: Sequence[str],
    repaired: Sequence[str],
) -> None:
    monkeypatch.setitem(DECLARED, mending_ceiling.id, "path")
    pyproject = _MEMBERS_TABLE + _STATEMENTS_TABLE
    written(
        tmp_path, {"a.py": _TWO_FUNCTIONS, "b.py": _TWO_FUNCTIONS, "pyproject.toml": pyproject}
    )
    load_catalog()

    with Lint(tmp_path, (mending_ceiling,)) as lint:
        lint.consumers = Importers(imported)
        repairs = lint.repairs(lint.facts(tmp_path), FIXABLE["review"])[mending_ceiling.id]

    assert {edit for repair in repairs for edit in repair} == {
        Append(path=path, text=text)
        for path in repaired
        for text in ("# members 2\n", "# statements 4\n")
    }
