from typing import NoReturn, Protocol

import pytest

from mainboard import Machine
from mainboard.probe import GPU, NvidiaGPU, Vendor
from mainboard.probe.providers.nvidia import apis as nvidia_apis_module

from ...support import (
    FakeNvidiaApis,
    FakeSensorlessDevice,
    InstallNvidiaStack,
    raise_unsupported,
)

_GIB = 1024**3


class Setup(Protocol):
    """Wire one CUDA/NVML stack shape into place before the assertion under test."""

    def __call__(self, install: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch) -> None: ...


def sensorless_cuda_core(install: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional layer loaded but this device has no memory sensor to read."""
    install()
    monkeypatch.setattr(NvidiaGPU, "system_device", FakeSensorlessDevice())


def unsupported_nvml_memory(install: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch) -> None:
    """No optional layer at all, and NVML refuses the memory query underneath it."""
    apis = install(has_cuda_core=False)
    monkeypatch.setattr(apis.nvml, "device_get_memory_info_v2", raise_unsupported)


def no_visible_device(install: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bindings load and answer with a device count of zero."""
    install(device_count=0)


def unimportable_bindings(install: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch) -> None:
    """A base install without the `[cuda]` extra, where the import itself fails."""

    def absent() -> NoReturn:
        raise ModuleNotFoundError("no cuda")

    nvidia_apis_module.nvidia_apis.cache_clear()
    monkeypatch.setattr(nvidia_apis_module, "nvidia_apis", absent)


@pytest.mark.parametrize(
    ("has_cuda_core", "source"),
    [
        pytest.param(True, "cuda-core-system", id="cuda-core"),
        pytest.param(False, "nvml", id="nvml"),
    ],
)
def test_a_visible_device_reports_the_same_identity_through_either_layer(
    has_cuda_core: bool, source: str, install_nvidia_stack: InstallNvidiaStack
) -> None:
    """Both API stacks answer the same identity, capability and capacity.

    Identity, capability and capacity read the same whether the optional `cuda.core` layer
    loaded or the provider fell back to `cuda.bindings` (runtime plus NVML) alone.
    """
    apis = install_nvidia_stack(has_cuda_core=has_cuda_core)
    assert apis.has_cuda_core is has_cuda_core
    assert NvidiaGPU.is_available() is True
    gpus = NvidiaGPU.all()
    assert len(gpus) == 2

    gpu = gpus[0]
    assert gpu.apis is nvidia_apis_module.nvidia_apis()  # a per-instance view of the cached stack
    assert gpu.vendor is Vendor.NVIDIA
    assert gpu.label == "NVIDIA GeForce RTX 4090"
    assert gpu.uuid == "GPU-deadbeef"
    assert str(gpu.cuda_architecture) == "8.9"
    assert gpu.architecture == "Ada"
    assert gpu.arch_key == "sm_89"
    assert gpu.driver == "580.65.06"  # the host driver, not the CUDA version it tops out at
    assert gpu.runtime_version == (13, 1)
    assert gpu.pci_bus_id == "0000:00:00.0"
    memory = gpu.memory
    assert (memory.total_bytes, memory.used_bytes, memory.free_bytes) == (
        24 * _GIB,
        6 * _GIB,
        18 * _GIB,
    )
    assert memory.source == source
    assert gpu.coherent is False  # a discrete card reports neither coherence attribute
    assert memory.unified is False


@pytest.mark.parametrize("has_cuda_core", [True, False], ids=["cuda-core", "nvml"])
def test_a_coherent_grace_hopper_pool_flags_its_memory_as_unified(
    has_cuda_core: bool, install_nvidia_stack: InstallNvidiaStack
) -> None:
    """Coherent-fabric devices carry the unified flag through every tier.

    A device reporting both pageable and concurrent-managed access sits on a coherent fabric
    where host RAM is a peer NUMA node of HBM, and the flag flows through either memory tier.
    """
    install_nvidia_stack(has_cuda_core=has_cuda_core, coherent=True)
    gpu = NvidiaGPU(index=0)
    assert gpu.coherent is True
    assert gpu.memory.unified is True


def test_a_driver_version_nvml_will_not_answer_reads_empty_rather_than_the_cuda_one(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Does a refused driver query leave the field empty instead of borrowing the CUDA version?

    The two are different facts and the receipt stamped one under the other's name for a whole
    generation, so an unanswerable driver is silence and never the number that reads like it.
    """

    def absent() -> NoReturn:
        raise AttributeError("module 'cuda.bindings.nvml' has no system_get_driver_version")

    monkeypatch.setattr(nvidia_host.nvml, "system_get_driver_version", absent)
    gpu = NvidiaGPU(index=0)
    assert gpu.driver == ""
    assert gpu.runtime_version == (13, 1)


def test_a_binding_without_the_attribute_query_degrades_to_not_coherent(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older binding with no `cudaDeviceGetAttribute` answers not-coherent, never a crash."""

    def absent(attr: int, index: int) -> NoReturn:
        raise AttributeError("module 'cuda.bindings.runtime' has no cudaDeviceGetAttribute")

    monkeypatch.setattr(nvidia_host.runtime, "cudaDeviceGetAttribute", absent)
    gpu = NvidiaGPU(index=0)
    assert gpu.coherent is False
    assert gpu.memory.unified is False


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(sensorless_cuda_core, id="cuda-core-sensorless"),
        pytest.param(unsupported_nvml_memory, id="nvml-unsupported"),
    ],
)
def test_the_memory_ladder_ends_at_the_cuda_runtime_reading(
    setup: Setup, install_nvidia_stack: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`cudaMemGetInfo` is the memory reading of last resort.

    Whichever tier refuses first, its free/total pair becomes the reading, with the current
    device restored around the query.
    """
    setup(install_nvidia_stack, monkeypatch)
    memory = NvidiaGPU(index=0).memory
    assert memory.source == "cuda-runtime"
    assert (memory.total_bytes, memory.used_bytes, memory.free_bytes) == (
        24 * _GIB,
        16 * _GIB,
        8 * _GIB,
    )


def test_a_failing_runtime_memory_query_raises_rather_than_reporting_zero_capacity(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last tier surfaces its failure instead of zeroing the reading.

    It has nothing to fall back on, and a host with no current device to restore still runs
    the query.
    """
    monkeypatch.setattr(nvidia_host.runtime, "cudaGetDevice", lambda: (99, 0))
    monkeypatch.setattr(nvidia_host.runtime, "cudaMemGetInfo", lambda: (99, 0, 0))
    monkeypatch.setattr(NvidiaGPU, "system_device", FakeSensorlessDevice())
    with pytest.raises(RuntimeError, match="cudaMemGetInfo"):
        _ = NvidiaGPU(index=0).memory


def test_a_failing_pci_bus_id_query_raises(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bus ID is how a visible index is resolved to a device, so a failure is fatal."""
    monkeypatch.setattr(
        nvidia_host.runtime, "cudaDeviceGetPCIBusId", lambda length, index: (99, b"")
    )
    with pytest.raises(RuntimeError, match="cudaDeviceGetPCIBusId"):
        _ = NvidiaGPU(index=0).pci_bus_id


@pytest.mark.parametrize(
    ("has_cuda_core", "refuse", "expected"),
    [
        pytest.param(True, False, (61, 37), id="cuda-core"),
        pytest.param(False, False, (48, 22), id="nvml"),
        pytest.param(False, True, (0, 0), id="both-refuse"),
    ],
)
def test_utilization_takes_the_first_layer_that_answers(
    has_cuda_core: bool,
    refuse: bool,
    expected: tuple[int, int],
    install_nvidia_stack: InstallNvidiaStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counters are optional hardware, so an unanswered read degrades to an empty pair."""
    apis = install_nvidia_stack(has_cuda_core=has_cuda_core)
    if refuse:
        monkeypatch.setattr(apis.nvml, "device_get_utilization_rates", raise_unsupported)
    reading = NvidiaGPU(index=0).utilization
    assert (reading.gpu_pct, reading.memory_pct) == expected


@pytest.mark.parametrize(
    ("refuse", "peak_gbs"),
    [(False, 1008.096), (True, 0.0)],
    ids=["bus_width_times_the_doubled_memory_clock", "nvml_refuses"],
)
def test_peak_bandwidth_is_the_bus_width_times_the_doubled_memory_clock(
    refuse: bool,
    peak_gbs: float,
    nvidia_host: FakeNvidiaApis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vendor headline figure, computed rather than looked up in a table of cards.

    A 384-bit bus at 10501 MHz is a 4090's 1008 GB/s, and a device whose clock query is
    unsupported scores nothing rather than reporting a fabricated peak.
    """
    if refuse:
        monkeypatch.setattr(nvidia_host.nvml, "device_get_max_clock_info", raise_unsupported)
    assert NvidiaGPU(index=0).peak_bandwidth_gbs == pytest.approx(peak_gbs)


def test_a_snapshot_gathers_every_sensor_the_device_answers(nvidia_host: FakeNvidiaApis) -> None:
    """One reading carries identity, region, power, temperature, utilization and processes."""
    reading = NvidiaGPU(index=0).snapshot(name="matmul")
    assert reading.unit_name == "NVIDIA GeForce RTX 4090"
    assert reading.region == "matmul"
    assert reading.energy.power_w == pytest.approx(17.647)
    assert reading.thermal.temperature_c == 42
    assert reading.utilization.gpu_pct == 61
    assert [(item.pid, item.used_bytes) for item in reading.processes] == [(4242, 2 * _GIB)]


def test_each_sensor_degrades_on_its_own_rather_than_sinking_the_reading(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device refusing every sensor still answers, with each field at its neutral value."""
    for query in (
        "device_get_power_usage",
        "device_get_temperature_v",
        "device_get_compute_running_processes_v3",
        "device_get_current_clocks_event_reasons",
    ):
        monkeypatch.setattr(nvidia_host.nvml, query, raise_unsupported)
    reading = NvidiaGPU(index=0).snapshot()
    assert reading.unit_name == "NVIDIA GeForce RTX 4090"
    assert (reading.energy.power_w, reading.thermal.temperature_c) == (0.0, 0)
    assert reading.processes == ()
    assert reading.thermal.is_throttling is False


def test_only_the_real_slowdowns_count_as_throttling(
    nvidia_host: FakeNvidiaApis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idle device is not a throttled one, so the benign bits never reach the reading.

    NVML answers with one mask covering both, and every idle GPU sets a bit in it, so
    reading the mask as a boolean would report a healthy device as held back.
    """
    reasons = nvidia_host.nvml.ClocksEventReasons
    assert NvidiaGPU(index=0).snapshot().thermal.is_throttling is False  # only the idle bit

    monkeypatch.setattr(
        nvidia_host.nvml,
        "clocks_event_reasons",
        reasons.EVENT_REASON_GPU_IDLE
        | reasons.EVENT_REASON_SW_POWER_CAP
        | reasons.THROTTLE_REASON_HW_THERMAL_SLOWDOWN,
    )
    thermal = NvidiaGPU(index=0).snapshot().thermal
    assert thermal.is_throttling is True
    assert thermal.throttle_names == ("power cap", "hardware thermal")


def test_system_api_refuses_when_the_optional_layer_never_loaded(
    install_nvidia_stack: InstallNvidiaStack,
) -> None:
    """`system_api` names the missing layer rather than handing back a `None` to call into."""
    install_nvidia_stack(has_cuda_core=False)
    with pytest.raises(RuntimeError, match="is unavailable"):
        _ = NvidiaGPU(index=0).system_api


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(no_visible_device, id="zero-devices"),
        pytest.param(unimportable_bindings, id="bindings-absent"),
    ],
)
def test_a_host_with_no_cuda_device_reports_nothing_instead_of_raising(
    setup: Setup, install_nvidia_stack: InstallNvidiaStack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the count is zero or the bindings are missing entirely, detection stays quiet."""
    setup(install_nvidia_stack, monkeypatch)
    assert NvidiaGPU.is_available() is False
    assert NvidiaGPU.all() == ()


def test_a_base_install_without_the_cuda_extra_degrades_at_the_import_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine without the CUDA extra still probes clean.

    The real `NvidiaApis` constructor runs here and fails to import `cuda.bindings.runtime`,
    so the whole fan-out through `GPU.all` and `Machine` has to survive the missing extra.
    """

    def absent(name: str) -> NoReturn:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(nvidia_apis_module, "import_module", absent)
    nvidia_apis_module.nvidia_apis.cache_clear()
    assert NvidiaGPU.is_available() is False
    assert all(gpu.vendor is not Vendor.NVIDIA for gpu in GPU.all())
    assert all(gpu.vendor is not Vendor.NVIDIA for gpu in Machine().gpus)
