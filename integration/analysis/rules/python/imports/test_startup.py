"""What an entry point runs at startup: PY-IMPO0009 lists in `__lazy_modules__` only the imports
whose names are read after import, never one a registry family needs, and PY-IMPO0010 names the
chain of eager imports that reaches a heavy library, on the first import of it that can wait."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from mainboard.analysis.engine.lint import Lint, load_catalog, verbs
from mainboard.analysis.rules.python.imports.startup import DeferrableImport, HeavyStartupImport

# One package whose `cli` module is the console script, and what its tests import.
_PACKAGE = {
    "pyproject.toml": (
        '[project]\nname = "pkg"\nversion = "0"\n\n[project.scripts]\npkg = "pkg.cli:main"\n'
    ),
    "src/pkg/__init__.py": "",
    "src/pkg/board.py": "class Board:\n    pass\n\n\ndef stamped(f):\n    return f\n",
}
# A registry family: its package exports the root and an implementation that registers on import.
_FAMILY = {
    "src/pkg/plugins/__init__.py": (
        'from .base import Backend\nfrom .vast import Vast\n\n__all__ = ["Backend", "Vast"]\n'
    ),
    "src/pkg/plugins/base.py": (
        "from patos import Registry\n\n\nclass Backend(Registry):\n    pass\n"
    ),
    "src/pkg/plugins/vast.py": "from .base import Backend\n\n\nclass Vast(Backend):\n    pass\n",
}
_IMPORT = "from .board import Board\n\n\n"
_BODY = _IMPORT + "def main():\n    return Board()\n"
# A list below the import it names, which Python reads too late to defer that import.
_LATE = """from .board import Board

__lazy_modules__ = ["pkg.board"]


def main():
    return Board()
"""
# An entry asking the family's root for an implementation by name.
_LOOKUP = 'from .plugins import Backend\n\n\ndef main():\n    return Backend.find("vast")\n'
# A chain from the entry to sqlalchemy through `mid` and `lake`, each module spelled twice: reading
# its import in a function, which can wait, and reading it at import, which cannot.
_MID = "from .lake import query\n\n\ndef run():\n    return query()\n"
_LAKE = "import sqlalchemy\n\n\ndef query():\n    sqlalchemy.true()\n"
_CALLS = "from .mid import run\n\n\ndef main():\n    return run()\n"
_RUNS = "from .mid import run\n\nRESULT = run()\n\n\ndef main():\n    return RESULT\n"
_MID_RUNS = "from .lake import query\n\nRESULT = query()\n\n\ndef run():\n    return RESULT\n"
_LAKE_RUNS = "import sqlalchemy\n\nTRUE = sqlalchemy.true()\n\n\ndef query():\n    return TRUE\n"


def _found(root: Path, files: Mapping[str, str]) -> dict[str, list[tuple]]:
    for name, content in {**_PACKAGE, **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    load_catalog()
    with Lint(root, (DeferrableImport, HeavyStartupImport), fixing=False) as lint:
        return lint.findings(lint.facts(root))


@pytest.mark.parametrize(
    ("files", "flagged"),
    [
        pytest.param({"src/pkg/cli.py": _BODY}, {"src/pkg/cli.py": "pkg.board"}, id="a-body-read"),
        pytest.param(
            {"src/pkg/cli.py": _IMPORT + "def main(board: Board) -> None:\n    pass\n"},
            {"src/pkg/cli.py": "pkg.board"},
            id="a-function-annotation",
        ),
        pytest.param(
            {"src/pkg/cli.py": "from . import board\n\n\ndef main():\n    return board.Board()\n"},
            {"src/pkg/cli.py": "pkg"},
            id="a-submodule-read-through-its-package",
        ),
        pytest.param(
            {
                "src/pkg/cli.py": _IMPORT
                + "DEFAULT = Board()\n\n\ndef main():\n    return DEFAULT\n"
            },
            {},
            id="a-module-level-read",
        ),
        pytest.param(
            {
                "src/pkg/cli.py": (
                    _IMPORT + "class Mine(Board):\n    pass\n\n\ndef main():\n    return Mine()\n"
                )
            },
            {},
            id="a-base-class",
        ),
        pytest.param(
            {"src/pkg/cli.py": "from .board import stamped\n\n\n@stamped\ndef main():\n    ...\n"},
            {},
            id="a-decorator",
        ),
        pytest.param(
            {"src/pkg/cli.py": _IMPORT + "def main(kind=Board):\n    return kind\n"},
            {},
            id="a-default-value",
        ),
        pytest.param(
            {
                "src/pkg/cli.py": (
                    _IMPORT + "class Model:\n    board: Board\n\n\ndef main():\n    return Model\n"
                )
            },
            {},
            id="a-class-body-annotation",
        ),
        pytest.param(
            {"src/pkg/cli.py": "import json\n\n\ndef main():\n    return json.dumps({})\n"},
            {},
            id="the-standard-library",
        ),
        pytest.param(
            {"src/pkg/cli.py": "from . import board\n\n\ndef main():\n    pass\n"},
            {},
            id="an-import-for-its-side-effect",
        ),
        pytest.param(
            {"src/pkg/cli.py": "def main():\n    pass\n", "src/pkg/other.py": _BODY},
            {},
            id="a-module-no-entry-reaches",
        ),
        pytest.param(
            {"src/pkg/cli.py": '__lazy_modules__ = ["pkg.board"]\n\n' + _BODY},
            {},
            id="a-list-above-the-import",
        ),
        pytest.param(
            {"src/pkg/cli.py": _LATE},
            {"src/pkg/cli.py": "pkg.board"},
            id="a-list-below-the-import-defers-nothing",
        ),
        pytest.param(
            {"src/pkg/cli.py": _LOOKUP, **_FAMILY},
            {"src/pkg/cli.py": "pkg.plugins", "src/pkg/plugins/__init__.py": "pkg.plugins.base"},
            id="a-family-keeps-its-implementation-eager",
        ),
    ],
)
def test_deferrable_import_lists_only_names_read_after_import(
    tmp_path: Path, files: Mapping[str, str], flagged: Mapping[str, str]
) -> None:
    found = _found(tmp_path, files)[DeferrableImport.id]
    assert {row[0]: row[3].split("`")[3] for row in found} == flagged


@pytest.mark.parametrize(
    ("lazy", "said"),
    [
        pytest.param(
            "",
            ["`pkg.cli` imports `sqlalchemy` at startup: pkg.cli -> pkg.lake -> sqlalchemy"],
            id="an-eager-chain",
        ),
        pytest.param('__lazy_modules__ = ["pkg.lake"]\n\n', [], id="a-lazy-link-in-the-chain"),
    ],
)
def test_heavy_startup_import_names_the_chain_that_reaches_the_library(
    tmp_path: Path, lazy: str, said: list[str]
) -> None:
    found = _found(
        tmp_path,
        {
            "src/pkg/cli.py": f"{lazy}from .lake import query\n\n\ndef main():\n    query()\n",
            "src/pkg/lake.py": "import sqlalchemy\n\n\ndef query():\n    sqlalchemy.true()\n",
        },
    )
    assert [row[3] for row in found[HeavyStartupImport.id]] == said


@pytest.mark.parametrize(
    "order",
    [
        pytest.param("abcdefgh", id="imported-in-order"),
        pytest.param("hgfedcba", id="imported-in-reverse"),
    ],
)
def test_a_library_that_equally_short_chains_reach_is_named_through_the_smallest_one(
    tmp_path: Path, order: str
) -> None:
    leaves = {f"src/pkg/{name}.py": _LAKE for name in "abcdefgh"}
    entry = "".join(f"from .{name} import query as {name}\n" for name in order)

    found = _found(tmp_path, {"src/pkg/cli.py": entry + "\n\ndef main():\n    a()\n", **leaves})

    assert [row[3] for row in found[HeavyStartupImport.id]] == [
        "`pkg.cli` imports `sqlalchemy` at startup: pkg.cli -> pkg.a -> sqlalchemy"
    ]


@pytest.mark.parametrize(
    ("chain", "stands"),
    [
        pytest.param((_CALLS, _MID, _LAKE), "src/pkg/cli.py", id="the-entry-import-can-wait"),
        pytest.param(
            (_RUNS, _MID, _LAKE), "src/pkg/mid.py", id="the-entry-reads-its-import-at-import"
        ),
        pytest.param(
            (_RUNS, _MID_RUNS, _LAKE), "src/pkg/lake.py", id="only-the-last-link-can-wait"
        ),
        pytest.param((_RUNS, _MID_RUNS, _LAKE_RUNS), "src/pkg/lake.py", id="no-link-can-wait"),
    ],
)
def test_a_heavy_chain_is_cut_at_its_first_deferrable_link(
    tmp_path: Path, chain: tuple[str, str, str], stands: str
) -> None:
    files = dict(zip(("src/pkg/cli.py", "src/pkg/mid.py", "src/pkg/lake.py"), chain, strict=True))

    found = _found(tmp_path, files)[HeavyStartupImport.id]

    assert [row[0] for row in found] == [stands]


def test_every_example_of_the_startup_rules_passes() -> None:
    """Each `bad` is flagged, the fix turns it into `good`, and `good` is not flagged."""
    verbs.test(selection=verbs.Selection(rule=(DeferrableImport.id, HeavyStartupImport.id)))
