import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest

from mainboard.analysis.engine.apply import Plan
from mainboard.analysis.engine.lint import Lint, load_catalog, selected
from mainboard.analysis.operations.nodes import Site
from mainboard.analysis.operations.python.suites import MergeTests

from ...engine.builders import written
from .readers import Merged

# Where the module under test lives, and the three tests merged unless a test names others.
_PATH = "test_m.py"
_TRIO = ("test_f_one", "test_f_two", "test_f_three")
_IDS = ("one", "two", "three")

# A test function by its name, with the node that holds it: the module or class body, past the
# decorated definition that wraps it.
_TESTS = """
    SELECT t.node_id, CASE WHEN w.type = 'decorated_definition' THEN w.parent_id
                           ELSE t.parent_id END
    FROM ast_nodes t JOIN ast_nodes w ON w.path = t.path AND w.node_id = t.parent_id
    WHERE t.path = ? AND t.type = 'function_definition' AND t.name = ?
"""


@contextmanager
def _rowed(
    source: str, *, names: Sequence[str] = _TRIO, into: str = "test_f", ids: Sequence[str] = _IDS
) -> Generator[tuple[Path, dict]]:
    """The project holding a module, and the row a rule's fix states to merge its tests.

    The row names the holder as its node, the tests in source order, the merged name and the ids
    of the cases, and carries the site of the holder.
    """
    load_catalog()
    with tempfile.TemporaryDirectory(prefix="mb-merge-") as scratch:
        root = written(Path(scratch), {_PATH: source})
        with Lint(root, selected(("PY-TEST0003",))) as lint:
            cursor = lint.facts(root).cursor(None)
            found = [
                row for name in names for row in cursor.execute(_TESTS, [_PATH, name]).fetchall()
            ]
            holder = found[0][1]
            site = Site.gathered(cursor, [(_PATH, holder)])[_PATH, holder]
        tests = [test for test, _ in found]
        yield root, {"site": site, "tests": tests, "name": into, "ids": list(ids)}


def _merged(
    source: str, *, names: Sequence[str] = _TRIO, into: str = "test_f", ids: Sequence[str] = _IDS
) -> Merged:
    """Merge the tests `names` of a module into the test `into`, with a case for each of `ids`."""
    with _rowed(source, names=names, into=into, ids=ids) as (root, row):
        operation = MergeTests()
        edits = operation.edits(row)
        plan = Plan(root)
        taken = edits is not None and plan.add(edits)
        data = plan.written().get(_PATH, (root / _PATH).read_bytes())
        return Merged(
            edits=[edit.kind for edit in edits or ()],
            refusal=operation.why,
            taken=taken,
            text=data.decode(),
        )


def _fixed(source: str, *, rule: str, context: Mapping[str, str] | None = None) -> Merged:
    """What the fix of `rule` makes of a module, every repair in one plan.

    context: the other files of the project, by path.
    """
    load_catalog()
    with tempfile.TemporaryDirectory(prefix="mb-merge-") as scratch:
        root = written(Path(scratch), {**(context or {}), _PATH: source})
        with Lint(root, selected((rule,))) as lint:
            repairs = lint.repairs(lint.facts(root), ["review"])[rule]
        plan = Plan(root)
        taken = [repair for repair in repairs if plan.add(repair)]
        data = plan.written().get(_PATH, (root / _PATH).read_bytes())
        return Merged(changes=[repair.change for repair in taken], text=data.decode())


@pytest.fixture(scope="module")
def merge() -> Callable[..., Merged]:
    """Merge tests of a scratch module as the engine does: from the facts of its tree."""
    return _merged


@pytest.fixture(scope="module")
def fix() -> Callable[..., Merged]:
    """Fix a scratch module through a rule: its query, its fix rows and its plan."""
    return _fixed


@pytest.fixture(scope="module")
def stated() -> Callable[..., AbstractContextManager[tuple[Path, dict]]]:
    """The row a rule's fix states to merge the tests of a scratch module, to alter before use."""
    return _rowed
