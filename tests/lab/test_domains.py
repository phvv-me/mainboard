from typing import Annotated

import pytest

from mainboard.lab import (
    Choices,
    Fixed,
    FloatRange,
    IntRange,
    Run,
)
from mainboard.lab.domains import space_of


def test_choices_stores_every_positional_value() -> None:
    domain = Choices("a", "b", 3)
    assert domain.values == ("a", "b", 3)


def test_choices_equality_and_hash_follow_values() -> None:
    assert Choices("a", "b") == Choices("a", "b")
    assert hash(Choices("a", "b")) == hash(Choices("a", "b"))
    assert Choices("a") != Choices("b")


def test_int_range_holds_its_bounds() -> None:
    domain = IntRange(1, 8)
    assert domain.lo == 1
    assert domain.hi == 8


def test_float_range_holds_its_bounds() -> None:
    domain = FloatRange(0.0, 1.0)
    assert domain.lo == pytest.approx(0.0)
    assert domain.hi == pytest.approx(1.0)


def test_fixed_holds_its_value() -> None:
    assert Fixed("solo").value == "solo"


class SampleFields:
    bits: Annotated[int, IntRange(1, 8)]
    label: Annotated[str, "not-a-domain-marker"]
    plain: int


def test_space_of_reads_domain_metadata_off_a_class() -> None:
    assert space_of(SampleFields) == {"bits": IntRange(1, 8)}


def sample_function(
    run: Run,
    *,
    variant: Annotated[str, Choices("a", "b")] = "a",
    label: Annotated[str, "not-a-domain-marker"] = "x",
    plain: int = 0,
) -> dict[str, float]:
    return {}


def test_space_of_reads_domain_metadata_off_a_function() -> None:
    assert space_of(sample_function) == {"variant": Choices("a", "b")}


def test_space_of_returns_empty_for_a_class_with_no_domain_metadata() -> None:
    class NoMetadata:
        plain: int

    assert space_of(NoMetadata) == {}
