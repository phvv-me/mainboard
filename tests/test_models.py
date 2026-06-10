from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import (
    Clock,
    ClockInfo,
    ComputeCapability,
    CpuSnapshot,
    EnergyInterval,
    EnergyReading,
    Environment,
    GPUSnapshot,
    MachineSnapshot,
    Memory,
    ThermalState,
    ThermalTracker,
    ThrottleReason,
    UnitSnapshot,
    Utilization,
)
from mainboard.enums import Scheduler, UnitKind, Vendor
from mainboard.models.compiler_info import CompilerInfo
from mainboard.models.dimm_card import DimmCard

byte_counts = st.integers(min_value=0, max_value=10**15)
names = st.text(st.characters(blacklist_categories=("Cs", "Cc")), max_size=24)


def memory() -> st.SearchStrategy[Memory]:
    """Build `Memory` directly from its model with valid byte fields."""
    return st.builds(
        Memory,
        scope=names,
        total_bytes=byte_counts,
        used_bytes=byte_counts,
        free_bytes=byte_counts,
        unified=st.booleans(),
        source=names,
        supported=st.booleans(),
    )


def clock() -> st.SearchStrategy[Clock]:
    """Build a `Clock` reading from the model."""
    return st.builds(
        Clock,
        domain=names,
        current_mhz=st.none() | st.floats(0, 1e5, allow_nan=False),
        source=names,
        supported=st.booleans(),
    )


def unit_snapshot() -> st.SearchStrategy[UnitSnapshot]:
    """Build a neutral `UnitSnapshot` from the model."""
    return st.builds(
        UnitSnapshot,
        name=names,
        unit_name=names,
        kind=st.sampled_from(list(UnitKind)),
        vendor=st.sampled_from(list(Vendor)),
        clocks=st.lists(clock(), max_size=3).map(tuple),
        memory=memory(),
    )


def gpu_snapshot() -> st.SearchStrategy[GPUSnapshot]:
    """Build a `GPUSnapshot` from the model."""
    return st.builds(
        GPUSnapshot,
        name=names,
        unit_name=names,
        vendor=st.sampled_from(list(Vendor)),
        memory=memory(),
        fan_speed_pct=st.integers(0, 100),
    )


def environment() -> st.SearchStrategy[Environment]:
    """Build an `Environment` from the model."""
    return st.builds(
        Environment,
        user=names,
        group=names,
        groups=st.lists(names, max_size=3).map(tuple),
        scheduler=st.sampled_from(list(Scheduler)),
    )


def machine_snapshot() -> st.SearchStrategy[MachineSnapshot]:
    """Build a full `MachineSnapshot` from its component model strategies."""
    return st.builds(
        MachineSnapshot,
        hostname=names,
        cpu=st.builds(
            CpuSnapshot,
            name=names,
            architecture=names,
            vendor=st.sampled_from(list(Vendor)),
            logical_cores=st.integers(0, 256),
            physical_cores=st.integers(0, 256),
            total_memory_bytes=byte_counts,
        ),
        memory=memory(),
        environment=environment(),
        gpus=st.lists(gpu_snapshot(), max_size=3).map(tuple),
        npus=st.lists(unit_snapshot(), max_size=2).map(tuple),
    )


@given(snapshot=machine_snapshot())
def test_machine_snapshot_json_round_trips(snapshot: MachineSnapshot) -> None:
    """Any snapshot survives a JSON dump/parse cycle unchanged."""
    restored = MachineSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot


@given(snapshot=machine_snapshot())
def test_machine_snapshot_unit_count_and_kinds_track_accelerators(
    snapshot: MachineSnapshot,
) -> None:
    """`unit_count` is CPU plus accelerators and `kinds` reflects what is present."""
    assert snapshot.unit_count == 1 + len(snapshot.gpus) + len(snapshot.npus)
    assert UnitKind.CPU in snapshot.kinds
    assert (UnitKind.GPU in snapshot.kinds) == bool(snapshot.gpus)
    assert (UnitKind.NPU in snapshot.kinds) == bool(snapshot.npus)


@given(snap=unit_snapshot() | gpu_snapshot() | environment())
def test_frozen_models_round_trip(snap: UnitSnapshot | GPUSnapshot | Environment) -> None:
    """Each frozen sub-model round-trips through its own JSON form."""
    restored = type(snap).model_validate_json(snap.model_dump_json())
    assert restored == snap


@given(total=byte_counts, used=byte_counts)
def test_memory_ratios_and_unit_conversions(total: int, used: int) -> None:
    """`Memory` scales each field to gibibytes and guards a zero total."""
    used = min(used, total)
    free = total - used
    pool = Memory(scope="m", total_bytes=total, used_bytes=used, free_bytes=free)
    assert pool.percent_used == (0.0 if total == 0 else used / total * 100)
    assert pool.total_gb == total / 1024**3
    assert pool.used_gb == used / 1024**3
    assert pool.free_gb == free / 1024**3


@given(power=st.integers(0, 10**6), energy=st.integers(0, 10**9))
def test_energy_reading_and_interval_conversions(power: int, energy: int) -> None:
    """Energy conversions are linear and average power guards a zero interval."""
    assert EnergyReading(power_mw=power).power_w == power / 1000.0
    interval = EnergyInterval(energy_mj=energy, duration_s=0.0)
    assert interval.energy_j == energy / 1000.0
    assert interval.energy_wh == energy / 3_600_000.0
    assert interval.avg_power_w == 0.0


@given(energy=st.integers(1, 10**9), duration=st.floats(0.1, 3600, allow_nan=False))
def test_energy_interval_average_power(energy: int, duration: float) -> None:
    """A positive interval reports average power as joules over seconds."""
    interval = EnergyInterval(energy_mj=energy, duration_s=duration)
    assert interval.avg_power_w == pytest.approx(interval.energy_j / duration)


@given(major=st.integers(0, 99), minor=st.integers(0, 99))
def test_compute_capability_is_a_comparable_pair(major: int, minor: int) -> None:
    """Compute capability stringifies as `major.minor` and orders numerically."""
    cap = ComputeCapability(major, minor)
    assert str(cap) == f"{major}.{minor}"
    assert repr(cap) == f"ComputeCapability({major}, {minor})"
    assert cap >= ComputeCapability(major, minor)
    assert ComputeCapability(major, minor + 1) > cap
    assert ComputeCapability(9, 0) > ComputeCapability(8, 10)


@pytest.mark.parametrize("flag", list(ThrottleReason))
def test_throttle_reason_concern_is_total_over_the_enum(flag: ThrottleReason) -> None:
    """Every throttle flag classifies as benign only for idle/applications clocks."""
    benign = ThrottleReason.GPU_IDLE | ThrottleReason.APPLICATIONS_CLOCKS
    expected = bool(flag & ~benign)
    assert flag.is_concerning == expected


@given(
    temp=st.integers(0, 120),
    slowdown=st.integers(0, 120),
    reasons=st.integers(0, 0x1FF),
)
def test_thermal_state_margin_and_throttle_names(temp: int, slowdown: int, reasons: int) -> None:
    """Thermal margin is slowdown minus temperature and names match active bits."""
    state = ThermalState(
        temperature_c=temp, slowdown_threshold_c=slowdown, throttle_reasons=reasons
    )
    assert state.margin_c == slowdown - temp
    active = ThrottleReason(reasons)
    assert state.is_throttling == active.is_concerning
    assert set(state.throttle_names) == {
        flag.name for flag in ThrottleReason if flag in active and flag.name
    }


@given(
    states=st.lists(st.tuples(st.integers(0, 120), st.integers(-50, 80), st.integers(0, 0x1FF)))
)
def test_thermal_tracker_accumulates_peaks_and_throttles(
    states: list[tuple[int, int, int]],
) -> None:
    """The tracker keeps peak temperature, minimum margin, and any throttle event."""
    tracker = ThermalTracker()
    snapshots = [
        ThermalState(temperature_c=t, slowdown_threshold_c=t + margin, throttle_reasons=r)
        for t, margin, r in states
    ]
    for snap in snapshots:
        tracker.record(snap)
    assert tracker.sample_count == len(states)
    if snapshots:
        assert tracker.peak_temperature_c == max(s.temperature_c for s in snapshots)
        assert tracker.min_margin_c == min([999, *(s.margin_c for s in snapshots)])
    assert tracker.any_throttling == any(s.is_throttling for s in snapshots)


def test_clock_info_and_utilization_defaults_round_trip() -> None:
    """The tiny telemetry models default to zero and round-trip."""
    assert ClockInfo().model_dump() == {"sm_mhz": 0, "memory_mhz": 0}
    util = Utilization(gpu_pct=50, memory_pct=25)
    assert Utilization.model_validate_json(util.model_dump_json()) == util


@given(size=byte_counts, speed=st.none() | st.integers(0, 12000))
def test_dimm_card_capacity_and_population(size: int, speed: int | None) -> None:
    """A DIMM card is populated exactly when it carries bytes."""
    card = DimmCard(locator="DIMM_A1", size_bytes=size, speed_mhz=speed)
    assert card.is_populated == (size > 0)
    assert card.size_gb == size / 1024**3


@pytest.mark.parametrize(
    ("binary", "output", "expected_kind", "expected_version"),
    [
        ("nvcc", "nvcc: NVIDIA (R) Cuda compiler\nrelease 12.4, V12.4.131", "nvcc", "12.4"),
        ("clang", "Apple clang version 17.0.0 (clang-1700.0.13)", "clang", None),
        ("g++", "g++ (GCC) 13.2.0", "gcc", None),
        ("clang", "Ubuntu clang version 18 for grace", "clang-grace", None),
        ("weird", "some unrecognized tool v1", "unknown", None),
    ],
)
def test_compiler_info_parses_kind_and_version(
    binary: str,
    output: str,
    expected_kind: str,
    expected_version: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CompilerInfo` normalizes kind and pulls a release string from `--version`."""
    monkeypatch.setattr("mainboard.models.compiler_info.shell.run", lambda *cmd: output)
    info = CompilerInfo(path=Path(f"/usr/bin/{binary}"))
    assert info.kind == expected_kind
    if expected_version is None:
        assert info.version == output.lower().split("\n")[0].strip()
    else:
        assert info.version == expected_version
