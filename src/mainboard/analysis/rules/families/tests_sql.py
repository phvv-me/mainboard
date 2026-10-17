"""SQL the testing families share, as scalar macros."""

from ...catalog import Function


class TestsHome(Function):
    """The nearest function or class a node sits in (-1 for the module): `scope.current` is not,
    it also changes inside loops."""

    scope: str

    expression = "coalesce(greatest(scope['function'], scope['class']), -1)"
