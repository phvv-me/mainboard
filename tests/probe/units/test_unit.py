from mainboard.probe import Unit, UnitKind, Vendor


def test_base_unit_neutral_defaults() -> None:
    """The base `Unit` reports unknown identity and empty memory."""
    unit = Unit()
    assert unit.kind == UnitKind.UNKNOWN
    assert unit.vendor == Vendor.UNKNOWN
    assert unit.label == "unknown"
    assert unit.architecture == "unknown"
    assert unit.memory.total_bytes == 0
    assert unit.memory.supported is True
