from mainboard.probe import ComputeCapability


def test_str_and_repr() -> None:
    """`str`/`repr` render the dotted version and the constructor call respectively."""
    capability = ComputeCapability(8, 9)
    assert str(capability) == "8.9"
    assert repr(capability) == "ComputeCapability(8, 9)"


def test_sm_target_string() -> None:
    """`sm` concatenates major and minor without the dot."""
    assert ComputeCapability(9, 0).sm == "sm_90"


def test_architecture_exact_pair_wins_over_major() -> None:
    """Ada (8.9) and Turing (7.5) are looked up by exact pair before the major fallback."""
    assert ComputeCapability(8, 9).architecture == "Ada"
    assert ComputeCapability(7, 5).architecture == "Turing"


def test_architecture_falls_back_to_major() -> None:
    """A capability with no exact-pair entry maps by major alone."""
    assert ComputeCapability(9, 0).architecture == "Hopper"
    assert ComputeCapability(8, 0).architecture == "Ampere"


def test_architecture_unknown_for_unmapped_major() -> None:
    """A major with no mapping at all yields `Unknown`."""
    assert ComputeCapability(3, 5).architecture == "Unknown"


def test_ordering_across_two_digit_minors() -> None:
    """Comparison respects the (major, minor) tuple, not a naive decimal read."""
    assert ComputeCapability(9, 0) > ComputeCapability(8, 10)
