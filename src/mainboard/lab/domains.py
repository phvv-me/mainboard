import annotationlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

type Scalar = str | int | float | bool
type Domain = Choices | IntRange | FloatRange | Fixed


@dataclass(frozen=True, slots=True)
class Choices:
    """A domain of discrete named values, declared as `Annotated` metadata.

    values: the allowed values, in declaration order.
    """

    values: tuple[Scalar, ...]

    def __init__(self, *values: Scalar) -> None:
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class IntRange:
    """A domain of integers between two bounds, declared as `Annotated` metadata.

    lo: the smallest allowed value, inclusive.
    hi: the largest allowed value, inclusive.
    """

    lo: int
    hi: int


@dataclass(frozen=True, slots=True)
class FloatRange:
    """A domain of floats between two bounds, declared as `Annotated` metadata.

    lo: the smallest allowed value, inclusive.
    hi: the largest allowed value, inclusive.
    """

    lo: float
    hi: float


@dataclass(frozen=True, slots=True)
class Fixed:
    """A domain pinned to one value, declared as `Annotated` metadata.

    value: the only value the field may hold.
    """

    value: Scalar


def space_of(cls_or_fn: type | Callable[..., object]) -> dict[str, Domain]:
    """The declared config domain for every `Annotated` field or parameter on `cls_or_fn`.

    Reads `cls_or_fn`'s own resolved annotations (`annotationlib.get_annotations` in `VALUE`
    format, the PEP 649 resolver Python 3.14 evaluates deferred annotations through) and keeps
    the ones carrying exactly one `Choices`/`IntRange`/`FloatRange`/`Fixed` marker. Deliberately
    reads only `cls_or_fn`'s own annotations rather than its whole MRO, since a generated
    `Experiment` subclass's config fields never live on a base class its `space_of` caller has
    no reason to resolve. Works uniformly on a pydantic model class (its fields) or a plain
    function (its parameters), since both expose their annotations the same way.

    cls_or_fn: a class or callable whose own hints may carry domain metadata.
    """
    hints = annotationlib.get_annotations(cls_or_fn, format=annotationlib.Format.VALUE)
    space: dict[str, Domain] = {}
    for name, hint in hints.items():
        for item in getattr(hint, "__metadata__", ()):
            if isinstance(item, Choices | IntRange | FloatRange | Fixed):
                space[name] = item
    return space
