"""`ModuleRelocation` respells the imports of a moved module: the from-imports of every module,
the plain `import a.b` of it with the `a.b.NAME` reads through it, and the relative imports of the
moved module itself, read from where it stood and written from where it lands."""

import pytest
from hypothesis import example, given, settings

from mainboard.analysis.edits import Rename
from mainboard.analysis.operations import ModuleRelocation

from .builders import path_of, records, trees
from .constants import MOVES, TREE
from .readers import applied, denoted_after, denoted_before

_APP = "src/pkg/app.py"
_INIT = "src/pkg/rules/__init__.py"


@pytest.mark.parametrize(
    ("path", "before", "after"),
    [
        pytest.param(
            _APP,
            "import pkg.rules.rules\n\nY = pkg.rules.rules.X\n",
            "import pkg.rules.core\n\nY = pkg.rules.core.X\n",
            id="a-plain-import-and-a-dotted-read",
        ),
        pytest.param(
            _APP,
            "import pkg.rules.rules as r\n\nY = r.X\n",
            "import pkg.rules.core as r\n\nY = r.X\n",
            id="a-plain-import-under-an-alias-keeps-the-alias",
        ),
        pytest.param(
            _APP,
            "import os, pkg.rules.rules\n\nY = (os, pkg.rules.rules.X.real)\n",
            "import os, pkg.rules.core\n\nY = (os, pkg.rules.core.X.real)\n",
            id="a-plain-import-among-others-and-a-read-past-the-module",
        ),
        pytest.param(
            _APP,
            "import pkg.rules\n\n\ndef f():\n    return pkg.rules.rules.X\n",
            "import pkg.rules\n\n\ndef f():\n    return pkg.rules.core.X\n",
            id="a-read-through-the-package-in-a-function",
        ),
        pytest.param(
            _APP,
            "from other import pkg\n\nY = pkg.rules.rules.X\n",
            "from other import pkg\n\nY = pkg.rules.rules.X\n",
            id="a-chain-whose-root-no-plain-import-binds-stays",
        ),
        pytest.param(
            _APP,
            "from pkg.rules import rules\nfrom pkg.rules.rules import *\n",
            "from pkg.rules import core as rules\nfrom pkg.rules.core import *\n",
            id="a-member-module-keeps-its-bound-name-and-a-star-import-follows",
        ),
        pytest.param(
            _INIT,
            "from .rules import X\nfrom . import rules\n",
            "from .core import X\nfrom . import core as rules\n",
            id="relative-imports-in-the-package-beside-the-module",
        ),
    ],
)
def test_the_imports_of_a_moved_module_are_respelled(path: str, before: str, after: str) -> None:
    sources = {**TREE, path: before}

    edits = ModuleRelocation.respelled(records(sources), MOVES)

    assert edits is not None
    assert applied(sources, edits)[path] == after


@pytest.mark.parametrize(
    ("before", "moves"),
    [
        pytest.param(
            "import pkg.rules.rules\n",
            {"pkg.rules.rules": "other.core"},
            id="a-plain-import-that-would-bind-another-top-package",
        ),
        pytest.param(
            "import pkg.rules\n\nY = pkg.rules.rules.X\n",
            {"pkg.rules.rules": "other.core"},
            id="a-read-that-would-bind-another-top-package",
        ),
        pytest.param("import pkg.rules.rules as\n", MOVES, id="a-file-that-does-not-parse"),
        pytest.param("import pkg . rules.rules\n", MOVES, id="a-module-name-spelled-with-spaces"),
        pytest.param(
            "import pkg.rules.rules\n\nY = (pkg.rules\n     .rules.X)\n",
            MOVES,
            id="a-dotted-read-split-over-lines",
        ),
    ],
)
def test_a_move_whose_imports_cannot_follow_is_refused(before: str, moves: dict[str, str]) -> None:
    sources = {**TREE, _APP: before}

    assert ModuleRelocation.respelled(records(sources), moves) is None


@settings(max_examples=60)
@example(drawn=({path_of("pkg.m"): "from . import m as n\n"}, "pkg.m", "pkg.a.moved"))
@example(
    drawn=(
        {path_of("pkg.a.m"): "from .. import m\nfrom pkg.b import n\n"},
        "pkg.a.m",
        "pkg.b.moved",
    )
)
@given(drawn=trees())
def test_every_relative_import_of_a_moved_module_resolves_to_the_same_module_after(
    drawn: tuple[dict[str, str], str, str],
) -> None:
    sources, old, new = drawn
    moves = {old: new}

    edits = ModuleRelocation.respelled(records(sources), moves)
    after = applied(sources, [*(edits or []), Rename(path=path_of(old), target=path_of(new))])

    assert edits is not None
    assert denoted_after(sources, moves=moves, after=after) == denoted_before(sources, moves=moves)
