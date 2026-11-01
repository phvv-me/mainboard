import ast
import importlib.util
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from patos import FrozenModel


class Merged(FrozenModel):
    """What merging tests, or a rule's fix, made of one file.

    edits: the kinds of edit the merge made, none when it refused.
    refusal: why the merge refused, empty text when it did not.
    taken: whether the plan took the edits, which a refused merge has none to give.
    changes: the change line of each repair a rule's fix made.
    text: the file after the plan.
    """

    edits: list[str] = []
    refusal: str = ""
    taken: bool = True
    changes: list[str] = []
    text: str


def parametrized(text: str, *, name: str) -> str:
    """The test `name` of the module `text` with its decorators, as `ast.unparse` writes it."""
    function = next(
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    )
    return ast.unparse(function)


def _imported(text: str) -> ModuleType:
    """The module `text` defines, imported from a file as pytest imports a test module."""
    with tempfile.TemporaryDirectory(prefix="mb-merged-") as scratch:
        path = Path(scratch) / "merged_module.py"
        path.write_text(text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("merged_module", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _cases(name: str, function: Callable[..., None]) -> dict[str, dict[str, object]]:
    """The ids a test is collected under, each with the arguments its case passes."""
    marks = [mark for mark in getattr(function, "pytestmark", []) if mark.name == "parametrize"]
    if not marks:
        return {name: {}}
    names, table = marks[0].args
    argnames = [each.strip() for each in names.split(",")] if isinstance(names, str) else names
    return {f"{name}[{case.id}]": dict(zip(argnames, case.values, strict=True)) for case in table}


def _passes(function: Callable[..., None], arguments: dict[str, object]) -> bool:
    """Whether the test passes when called with `arguments`."""
    try:
        function(**arguments)
    except AssertionError:
        return False
    return True


def outcomes(text: str) -> dict[str, bool]:
    """Whether each case of the tests in the module `text` passes, by its collected id.

    The module is imported as pytest would, so a `parametrize` mark carries the cases it states.
    """
    tests = {
        name: each for name, each in vars(_imported(text)).items() if name.startswith("test_")
    }
    return {
        case: _passes(function, arguments)
        for name, function in tests.items()
        for case, arguments in _cases(name, function).items()
    }
