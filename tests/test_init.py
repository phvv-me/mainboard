import pytest

import mainboard


def test_every_exported_name_resolves_and_nothing_else_does() -> None:
    """The facade defers each name to its own module, so reading one has to actually reach it.

    A name that resolves to nothing would be an import error at the moment some caller finally
    touched it, which is exactly the failure a flat eager facade could never have, so the whole
    surface is read here. Reading it twice is the memoization, since a deferred name binds onto
    the module and stops costing an import lookup after the first ask.
    """
    for name in mainboard.__all__:
        assert getattr(mainboard, name) is getattr(mainboard, name)
    assert mainboard.Board.__name__ == "Board"
    assert mainboard.ExperimentStudy is not mainboard.ProfileStudy
    with pytest.raises(AttributeError, match="has no attribute 'Nothing'"):
        mainboard.Nothing  # noqa: B018
