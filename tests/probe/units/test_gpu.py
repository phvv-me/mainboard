from mainboard.probe import GPU, UnitKind, Vendor


def test_base_gpu_defaults() -> None:
    """A bare `GPU` exposes zeroed sensors and an unsupported memory reading."""
    gpu = GPU()
    assert gpu.kind == UnitKind.GPU
    assert gpu.vendor == Vendor.UNKNOWN
    assert gpu.label == "unknown"
    assert gpu.uuid == ""
    assert gpu.architecture == "unknown"
    assert gpu.arch_key == "unknown"  # base key is the lowercased architecture name
    assert gpu.memory.total_bytes == 0
    assert gpu.memory.supported is False
    assert gpu.driver_version is None


def test_base_gpu_utilization_is_empty() -> None:
    reading = GPU(index=0).utilization
    assert (reading.gpu_pct, reading.memory_pct) == (0, 0)
