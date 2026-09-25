import annotationlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

type Scalar = str | int | float | bool
type Domain = Choices | IntRange | FloatRange | Fixed


@dataclass(frozen=True, slots=True)
class Choices:
    """A domain of discrete named values in declaration order, declared as `Annotated` metadata."""

    values: tuple[Scalar, ...]

    def __init__(self, *values: Scalar) -> None:
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class IntRange:
    """A domain of integers between two inclusive bounds, declared as `Annotated` metadata."""

    lo: int
    hi: int


@dataclass(frozen=True, slots=True)
class FloatRange:
    """A domain of floats between two inclusive bounds, declared as `Annotated` metadata."""

    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class Fixed:
    """A domain pinned to one value, declared as `Annotated` metadata."""

    value: Scalar


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
