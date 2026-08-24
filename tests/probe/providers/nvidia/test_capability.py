import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from mainboard.probe import ComputeCapability

# Majors and minors well past anything shipped, so the rendering and ordering properties hold
# for a capability NVIDIA has not announced yet rather than only for the table below.
_CAPABILITIES = st.builds(ComputeCapability, major=st.integers(0, 20), minor=st.integers(0, 15))


@given(left=_CAPABILITIES, right=_CAPABILITIES)
@example(left=ComputeCapability(9, 0), right=ComputeCapability(8, 10))
def test_a_capability_renders_a_dotted_and_a_dot_free_form_and_orders_as_a_pair(
    left: ComputeCapability, right: ComputeCapability
) -> None:
    """Capabilities render as `sm_NN` and compare as pairs.

    The target is the dotted version with the dot removed, and comparison follows the pair
    rather than a decimal read, so 9.0 outranks 8.10 instead of losing to it.
    """
    assert str(left) == f"{left.major}.{left.minor}"
    assert left.sm == f"sm_{str(left).replace('.', '')}"
    assert repr(left) == f"ComputeCapability({left.major}, {left.minor})"
    assert (left > right) is ((left.major, left.minor) > (right.major, right.minor))


@pytest.mark.parametrize(
    ("major", "minor", "architecture"),
    [
        (8, 9, "Ada"),
        (7, 5, "Turing"),
        (9, 0, "Hopper"),
        (8, 0, "Ampere"),
        (3, 5, "Unknown"),
    ],
)
def test_the_architecture_table_checks_the_exact_pair_before_the_major(
    major: int, minor: int, architecture: str
) -> None:
    """The exact pair is looked up before the major decides.

    Ada and Turing share a major with Ampere and Volta, and everything else maps by major,
    down to `Unknown` for a generation with no entry.
    """
    assert ComputeCapability(major, minor).architecture == architecture
