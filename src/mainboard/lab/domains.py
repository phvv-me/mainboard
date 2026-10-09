import annotationlib
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

if TYPE_CHECKING:
    from collections.abc import Callable

type Scalar = str | int | float | bool
type Domain = Choices | IntRange | FloatRange | Fixed


class Marker(FrozenModel):
    """The base of every domain marker, `Annotated` metadata that leaves the field's own schema.

    A marker is written positionally, `IntRange(1, 8)`, so each subclass types its own `__init__`
    and hands its fields on by keyword.
    """

    def __init__(self, **fields: Scalar | tuple[Scalar, ...]) -> None:
        super().__init__(**fields)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return handler(source)


class Choices(Marker):
    """A domain of discrete named values in declaration order, declared as `Annotated` metadata."""

    values: tuple[Scalar, ...]

    def __init__(self, *values: Scalar) -> None:
        super().__init__(values=values)


class IntRange(Marker):
    """A domain of integers between two inclusive bounds, declared as `Annotated` metadata."""

    lo: int
    hi: int

    def __init__(self, lo: int, hi: int) -> None:
        super().__init__(lo=lo, hi=hi)


class FloatRange(Marker):
    """A domain of floats between two inclusive bounds, declared as `Annotated` metadata."""

    lo: float
    hi: float

    def __init__(self, lo: float, hi: float) -> None:
        super().__init__(lo=lo, hi=hi)


class Fixed(Marker):
    """A domain pinned to one value, declared as `Annotated` metadata."""

    value: Scalar

    def __init__(self, value: Scalar) -> None:
        super().__init__(value=value)


def space_of(cls_or_fn: type | Callable[..., object]) -> dict[str, Domain]:
    """The domain marker of every `Annotated` field (of a class) or parameter (of a function).

    Reads only `cls_or_fn`'s own annotations through PEP 649's `annotationlib`, not its MRO,
    since a generated `Experiment`'s config fields never live on a base class.
    """
    hints = annotationlib.get_annotations(cls_or_fn, format=annotationlib.Format.VALUE)
    return {
        name: item
        for name, hint in hints.items()
        for item in getattr(hint, "__metadata__", ())
        if isinstance(item, Choices | IntRange | FloatRange | Fixed)
    }
