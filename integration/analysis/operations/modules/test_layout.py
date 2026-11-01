"""`MoveModules` renames files and respells the imports of the modules among them; a folder a
module lands in gets the `__init__.py` its old folder had, a path the row removes goes with the
move, and a move that cannot be made whole is refused with its reason. The rules that move
modules (ALL-MODU0003, ALL-FILE0004, ALL-FILE0005) state their moves as rows of this operation."""

import pytest

from mainboard.analysis.engine.lint import verbs
from mainboard.analysis.operations import MoveModules, Operation
from mainboard.analysis.rules.general.dependencies.modules import ModuleInception
from mainboard.analysis.rules.general.project.filesystem import (
    DirectoryPathway,
    PrefixedModuleGroup,
)

from .builders import Row, moving
from .constants import CORE, RULES, TREE
from .readers import applied, created, removed

_APP = "src/pkg/app.py"
_PARTS = {
    "src/pkg/__init__.py": "",
    "src/pkg/parts/__init__.py": "",
    "src/pkg/parts/a_one.py": "ONE = 1\n",
}
_INNER = {
    "src/pkg/__init__.py": "",
    "src/pkg/outer/inner/__init__.py": "from . import a\n",
    "src/pkg/outer/inner/a.py": "A = 1\n",
    "src/pkg/outer/inner/data.json": "{}\n",
    _APP: "from .outer.inner import a\nfrom .outer.inner.a import A\n",
}


@pytest.mark.parametrize(
    ("sources", "moves", "after"),
    [
        pytest.param(
            {**TREE, _APP: "import pkg.rules.rules\n\nY = pkg.rules.rules.X\n"},
            [(RULES, CORE)],
            {
                **{path: text for path, text in TREE.items() if path != RULES},
                CORE: "X = 1\n",
                _APP: "import pkg.rules.core\n\nY = pkg.rules.core.X\n",
            },
            id="a-module-renamed-beside-its-package-initializer",
        ),
        pytest.param(
            _INNER,
            [
                ("src/pkg/outer/inner/__init__.py", "src/pkg/outer/__init__.py"),
                ("src/pkg/outer/inner/a.py", "src/pkg/outer/a.py"),
                ("src/pkg/outer/inner/data.json", "src/pkg/outer/data.json"),
            ],
            {
                "src/pkg/__init__.py": "",
                "src/pkg/outer/__init__.py": "from . import a\n",
                "src/pkg/outer/a.py": "A = 1\n",
                "src/pkg/outer/data.json": "{}\n",
                _APP: "from .outer import a\nfrom .outer.a import A\n",
            },
            id="a-folder-collapsed-into-the-one-that-held-only-it",
        ),
        pytest.param(
            _INNER,
            [
                ("src/pkg/outer/inner/__init__.py", "src/pkg/inner/__init__.py"),
                ("src/pkg/outer/inner/a.py", "src/pkg/inner/a.py"),
                ("src/pkg/outer/inner/data.json", "src/pkg/inner/data.json"),
            ],
            {
                "src/pkg/__init__.py": "",
                "src/pkg/inner/__init__.py": "from . import a\n",
                "src/pkg/inner/a.py": "A = 1\n",
                "src/pkg/inner/data.json": "{}\n",
                _APP: "from .inner import a\nfrom .inner.a import A\n",
            },
            id="a-folder-taking-the-place-of-the-one-that-held-only-it",
        ),
    ],
)
def test_a_move_renames_each_file_and_respells_the_imports_of_its_modules(
    sources: dict[str, str], moves: list[tuple[str, str]], after: dict[str, str]
) -> None:
    edits = Operation.rendered(moving(sources, *moves))

    assert edits is not None
    assert applied(sources, edits) == after


@pytest.mark.parametrize(
    ("columns", "made"),
    [
        pytest.param(
            {"inits": ["src/pkg/parts/__init__.py"]},
            ["src/pkg/parts/a/__init__.py"],
            id="the-old-folder-had-one",
        ),
        pytest.param({}, [], id="the-old-folder-had-none"),
        pytest.param(
            {
                "inits": ["src/pkg/parts/__init__.py"],
                "held": ["src/pkg/parts/a/__init__.py"],
            },
            [],
            id="the-new-folder-has-one-already",
        ),
    ],
)
def test_a_folder_a_module_lands_in_gets_the_init_its_old_folder_had(
    columns: dict[str, list[str]], made: list[str]
) -> None:
    move = ("src/pkg/parts/a_one.py", "src/pkg/parts/a/one.py")

    edits = Operation.rendered(moving(_PARTS, move, **columns))

    assert edits is not None
    assert created(edits) == made


def test_a_path_the_row_removes_goes_with_the_move() -> None:
    move = ("src/pkg/parts/a_one.py", "src/pkg/a_one.py")

    edits = Operation.rendered(moving(_PARTS, move, removed=["src/pkg/parts"]))

    assert edits is not None
    assert removed(edits) == ["src/pkg/parts"]


@pytest.mark.parametrize(
    ("row", "why"),
    [
        pytest.param(
            moving(TREE, (RULES, CORE), mentions=["docs/layout.md"]),
            "`docs/layout.md` names a moved module or path",
            id="a-mention-outside-python",
        ),
        pytest.param(
            moving(TREE, (RULES, CORE), ("src/pkg/rules/__init__.py", CORE)),
            "land on a path where another stands, starts or lands",
            id="two-moves-to-one-target",
        ),
        pytest.param(
            moving(TREE, (RULES, CORE), held=[CORE]),
            "land on a path where another stands, starts or lands",
            id="a-target-where-a-file-stands",
        ),
        pytest.param(
            moving(TREE, (RULES, CORE), (CORE, "src/pkg/rules/more.py")),
            "land on a path where another stands, starts or lands",
            id="a-target-where-another-move-starts",
        ),
        pytest.param(
            moving(TREE, ("src/pkg/rules/missing.py", CORE)),
            "`src/pkg/rules/missing.py` is no module of the project",
            id="a-module-the-facts-do-not-hold",
        ),
        pytest.param(
            moving(TREE, (RULES, "lib/core.py")),
            "`src/pkg/rules/rules.py` is no module of the project or leaves its source root",
            id="a-target-outside-the-source-root",
        ),
        pytest.param(
            moving({**TREE, _APP: "import pkg.rules.rules\n"}, (RULES, "src/other/core.py")),
            "an import of a moved module cannot be respelled",
            id="a-plain-import-that-would-bind-another-package",
        ),
        pytest.param(
            {"operation": "move-modules", "moves": [{"path": RULES}], "files": []},
            "the row's moves",
            id="a-move-without-a-target",
        ),
    ],
)
def test_a_move_that_cannot_be_made_whole_is_refused_with_its_reason(row: Row, why: str) -> None:
    operation = MoveModules()

    assert operation.edits(row) is None
    assert why in operation.why


def test_every_example_of_the_module_moving_rules_passes() -> None:
    """Each `bad` is flagged, the fix turns it into `good`, and a refusal leaves it as it was."""
    ids = (ModuleInception.id, DirectoryPathway.id, PrefixedModuleGroup.id)

    verbs.test(selection=verbs.Selection(rule=ids))
