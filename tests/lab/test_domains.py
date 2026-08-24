from typing import TYPE_CHECKING, Annotated

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.lab import Choices, Fixed, FloatRange, IntRange, Run
from mainboard.lab.domains import space_of

from ..strategies import WORDS

if TYPE_CHECKING:
    from collections.abc import Callable

    from mainboard.lab.domains import Domain

# Exactly the scalar kinds a declared config field may hold, no NaN because a domain marker is
# compared for equality and a NaN bound would never equal the twin it was declared alongside.
_SCALARS = st.one_of(
    WORDS, st.integers(), st.booleans(), st.floats(allow_nan=False, allow_infinity=False)
)


class SampleFields:
    bits: Annotated[int, IntRange(1, 8)]
    label: Annotated[str, "not-a-domain-marker"]
    plain: int


class NoMarkers:
    plain: int


def sample_function(
    run: Run,
    *,
    variant: Annotated[str, Choices("a", "b")] = "a",
    label: Annotated[str, "not-a-domain-marker"] = "x",
    plain: int = 0,
) -> dict[str, float]:
    """A measuring function whose keyword parameters declare the config domain."""
    return {}


@given(
    values=st.lists(_SCALARS, max_size=4),
    bounds=st.tuples(st.integers(), st.integers()),
    span=st.tuples(
        st.floats(allow_nan=False, allow_infinity=False),
        st.floats(allow_nan=False, allow_infinity=False),
    ),
    pinned=_SCALARS,
)
def test_every_domain_marker_is_a_frozen_value_object_carrying_its_declaration(
    *,
    values: list[str | int | bool | float],
    bounds: tuple[int, int],
    span: tuple[float, float],
    pinned: str | int | bool | float,
) -> None:
    assert Choices(*values).values == tuple(values)
    assert Choices(*values) == Choices(*values)
    assert hash(Choices(*values)) == hash(Choices(*values))
    assert Choices(*values) != Choices(*values, "one-more")
    assert (IntRange(*bounds).lo, IntRange(*bounds).hi) == bounds
    assert IntRange(*bounds) == IntRange(*bounds)
    assert (FloatRange(*span).lo, FloatRange(*span).hi) == span
    assert FloatRange(*span) == FloatRange(*span)
    assert Fixed(pinned).value == pinned
    assert Fixed(pinned) == Fixed(pinned)


@pytest.mark.parametrize(
    ("declared", "space"),
    [
        pytest.param(SampleFields, {"bits": IntRange(1, 8)}, id="a-classs-annotated-fields"),
        pytest.param(
            sample_function, {"variant": Choices("a", "b")}, id="a-functions-keyword-parameters"
        ),
        pytest.param(NoMarkers, {}, id="nothing-carrying-a-marker"),
    ],
)
def test_space_of_keeps_only_what_carries_a_domain_marker(
    declared: type | Callable[..., dict[str, float]], space: dict[str, Domain]
) -> None:
    assert space_of(declared) == space
