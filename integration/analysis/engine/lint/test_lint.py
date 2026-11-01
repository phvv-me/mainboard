"""A project in several languages is checked by the rules whose facts exist. A rule reading a view
no family derived (the Rust and TypeScript surfaces, which are not verified) is skipped before its
SQL runs, and the rules of the project's other languages go on."""

from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Lint, load_catalog
from mainboard.analysis.rules.python.design.modules import NonInitReexportModule
from mainboard.analysis.rules.python.testing.redundancy import (
    DuplicateTestIntentClusterCount,
    PytestReaching,
)
from mainboard.analysis.rules.rust.lifetimes import (
    ElidableLifetimeAnnotation,
    LifetimesAnnotations,
)
from mainboard.analysis.rules.rust.ownership import CloneInsideLoop
from mainboard.analysis.rules.typescript.modules import RelativeImportDepth, StarReexportSurface

from ..builders import written

_RELATIONS_SQL = (
    "SELECT function_name FROM duckdb_functions() "
    "WHERE function_type = 'table_macro' AND list_contains($names, function_name)"
)
_RULES = (NonInitReexportModule, StarReexportSurface, RelativeImportDepth, CloneInsideLoop)
_PYTHON = {"barrel.py": 'from .tool import tool\n\n__all__ = ["tool"]\n'}
_RUST = {"lib.rs": "pub fn one() -> i32 {\n    1\n}\n"}
_TYPESCRIPT = {"index.ts": "export const one = 1;\n"}
_CUDA = {"kernel.cu": "__global__ void zero(float* x) {\n  x[threadIdx.x] = 0.0f;\n}\n"}


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(_PYTHON | _TYPESCRIPT, id="python-and-typescript"),
        pytest.param(_PYTHON | _RUST, id="python-and-rust"),
        pytest.param(_PYTHON | _TYPESCRIPT | _RUST | _CUDA, id="python-typescript-rust-and-cuda"),
    ],
)
def test_a_project_in_several_languages_is_checked_by_the_rules_whose_facts_exist(
    tmp_path: Path, files: dict[str, str]
) -> None:
    load_catalog()
    written(tmp_path, files)

    with Lint(tmp_path, _RULES, fixing=False) as lint:
        found = lint.findings(lint.facts(tmp_path))

    assert {rule: [row[0] for row in rows] for rule, rows in found.items()} == {
        NonInitReexportModule.id: ["barrel.py"]
    }


def test_a_skipped_rule_leaves_the_relations_only_it_reads_uncreated(tmp_path: Path) -> None:
    load_catalog()
    written(tmp_path, _PYTHON | _RUST)
    rules = (DuplicateTestIntentClusterCount, ElidableLifetimeAnnotation)
    names = [PytestReaching.name(), LifetimesAnnotations.name()]

    with Lint(tmp_path, rules, fixing=False) as lint:
        base = lint.facts(tmp_path)
        created = base.connection.execute(_RELATIONS_SQL, {"names": names}).fetchall()

    assert created == [(PytestReaching.name(),)]
