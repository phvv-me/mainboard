import sys
from types import ModuleType

import pytest

from mainboard.trials import Absent, Owed, adaptive, driver


@pytest.fixture
def doubled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver stood in for, at the import call `driver` makes and nowhere wider.

    NOT `sys.modules`, and the reason is this suite itself: hypothesis is what most of these tests
    are written in, its pytest plugin imports the real module during every call phase, and a fake
    installed under that name takes the plugin down with it. Patching the one import call keeps
    the refusal path and the error handling of `driver` exactly as shipped.
    """
    monkeypatch.setattr(adaptive, "import_module", lambda name: ModuleType(name))


def owed() -> Owed:
    """The declared cell a candidate below owes its confirmation to."""
    return Owed(lane="alpha/test_law.py::test_confirms", cell={"shape": "w96"})


def test_a_missing_driver_refuses_by_naming_the_package_and_the_extra_that_ships_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ModuleNotFoundError three frames down does not tell a reader what to install."""
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(Absent, match=r"pip install mainboard\[search\]"):
        driver("optuna", "search")


def test_a_present_driver_comes_back_as_the_module_itself(doubled: None) -> None:
    """The import path under test is the real one, and only what comes back through it is fake."""
    assert driver("hypothesis", "adversarial").__name__ == "hypothesis"


def test_an_owed_confirmation_states_the_cell_it_names_and_says_so_when_it_names_none() -> None:
    """A candidate carries the debt in words, because the receipt is what a reader reaches for."""
    assert "alpha/test_law.py::test_confirms[shape=w96] on fresh seeds" in owed().stated
    bare = Owed(lane="alpha/test_law.py::test_confirms", seeds="two fresh")
    assert bare.stated.endswith("test_confirms on two fresh seeds")
