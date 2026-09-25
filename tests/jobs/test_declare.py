import ast

import pytest
from cyclopts import App

from mainboard import MissionError
from mainboard.jobs import Declaration, job
from mainboard.jobs.declare import MARK, declared


def test_the_decorator_keeps_its_declaration_on_the_target_and_changes_nothing_else() -> None:
    """A no-op at runtime: the function still runs, the application is still the application."""

    @job(needs=("data/corpus",), resources=("registry.toml",), fetch="evidence")
    def main() -> int:
        return 3

    app = job(needs=["data"])(App())
    assert main() == 3
    assert getattr(main, MARK) == Declaration(
        needs=("data/corpus",), resources=("registry.toml",), fetch="evidence"
    )
    assert isinstance(app, App)
    assert getattr(app, MARK, Declaration(needs=("data",))) == Declaration(needs=("data",))


@pytest.mark.parametrize(
    ("source", "name", "expected"),
    [
        (
            '@job(needs=("a", "b"), fetch="out")\ndef main():\n    pass\n',
            "main",
            Declaration(needs=("a", "b"), fetch="out"),
        ),
        (
            "import mainboard.jobs as mj\n\n@mj.job(resources=['r'])\ndef main():\n    ...\n",
            "main",
            Declaration(resources=("r",)),
        ),
        (
            'from cyclopts import App\napp = job(needs=("d",))(App(help="x"))\n',
            "app",
            Declaration(needs=("d",)),
        ),
        ("@job()\nasync def main():\n    pass\n", "main", Declaration()),
        ("@other()\ndef main():\n    pass\n", "main", Declaration()),
        ("def main():\n    pass\n", "main", Declaration()),
        ("app = App()\n", "app", Declaration()),
        ("app = wrap(App())\n", "app", Declaration()),
        ("x = 1\n", "main", Declaration()),
        (
            "class TestCases:\n    @job(needs=('d',))\n    def test_case(self):\n        pass\n",
            "TestOther::test_case",
            Declaration(),
        ),
    ],
    ids=[
        "a decorated function",
        "the decorator through its module",
        "a wrapped application",
        "an async function declaring nothing",
        "another decorator",
        "no decorator",
        "a bare application",
        "an application wrapped by something else",
        "no such target",
        "a method of a class the module does not define",
    ],
)
def test_a_declaration_is_read_off_the_syntax_without_importing_the_file(
    source: str, name: str, expected: Declaration
) -> None:
    assert declared(ast.parse(source), name) == expected


@pytest.mark.parametrize(
    "source",
    [
        "@job(needs=paths())\ndef main():\n    pass\n",
        "@job(**declared)\ndef main():\n    pass\n",
    ],
    ids=["a computed value", "a splatted mapping"],
)
def test_a_declaration_that_needs_running_is_refused_by_name(source: str) -> None:
    """The file is never imported, so a path only import time could compute is a path lost."""
    with pytest.raises(MissionError, match="must be literal"):
        declared(ast.parse(source), "main")
