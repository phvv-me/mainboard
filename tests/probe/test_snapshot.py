import json
from pathlib import Path
from typing import TYPE_CHECKING

from mainboard import HostFacts, Machine
from mainboard.probe import (
    GPU,
    CgroupMemory,
    Environment,
    Fabric,
    FabricPort,
    Memory,
    Scheduler,
    Scratch,
)

if TYPE_CHECKING:
    import pytest

_GIB = 1024**3


def test_collected_builds_facts_from_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """`collected` reads hostname, CPU, memory, cgroup, scratch, scheduler, and GPUs."""
    machine = Machine()
    monkeypatch.setattr(
        type(machine.host), "cgroup_memory", CgroupMemory(limit_bytes=100 * _GIB, capped=True)
    )
    monkeypatch.setattr(
        type(machine.host),
        "scratch",
        Scratch(path=Path("/local"), free_bytes=50 * _GIB, source="LOCALDIR"),
    )
    monkeypatch.setattr(type(machine.host), "memory", Memory(total_bytes=200 * _GIB))
    monkeypatch.setattr(type(machine), "environment", Environment(scheduler=Scheduler.PBS))
    monkeypatch.setattr(type(machine), "gpus", (GPU(index=0),))
    monkeypatch.setattr(Fabric, "probe", classmethod(lambda cls: ()))

    facts = HostFacts.collected()
    assert facts.schema_version == 1
    assert facts.cgroup.limit_bytes == 100 * _GIB
    assert facts.cgroup.capped is True
    assert facts.scratch.path == "/local"
    assert facts.scratch.free_bytes == 50 * _GIB
    assert facts.scratch.source == "LOCALDIR"
    assert facts.memory_total_bytes == 200 * _GIB
    assert facts.scheduler is Scheduler.PBS
    assert len(facts.gpus) == 1
    assert facts.gpus[0].arch_key is None  # the base GPU's arch is "unknown"


def test_collected_scratch_path_is_none_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable scratch tier serializes its path as `None`, not the string "None"."""
    machine = Machine()
    monkeypatch.setattr(type(machine.host), "scratch", Scratch())
    facts = HostFacts.collected()
    assert facts.scratch.path is None


def test_collected_reports_a_known_arch_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU with a real architecture surfaces its `arch_key` on the fact."""

    class NamedGPU(GPU):
        @property
        def arch_key(self) -> str:
            return "sm_90"

        @property
        def label(self) -> str:
            return "Test GPU"

    machine = Machine()
    monkeypatch.setattr(type(machine), "gpus", (NamedGPU(index=0),))
    facts = HostFacts.collected()
    assert facts.gpus[0].name == "Test GPU"
    assert facts.gpus[0].arch_key == "sm_90"


def test_round_trips_through_json() -> None:
    """`model_dump_json`/`model_validate_json` reproduce an equal `HostFacts`."""
    facts = HostFacts(
        hostname="gold",
        cpu_name="AMD EPYC",
        cpu_logical_cores=64,
        memory_total_bytes=512 * _GIB,
        scheduler=Scheduler.SLURM,
        fabric=(FabricPort(device="mlx5_0", port=1, link_layer="InfiniBand"),),
    )
    restored = HostFacts.model_validate_json(facts.model_dump_json())
    assert restored == facts


def test_open_model_tolerates_unknown_fields_from_a_newer_writer() -> None:
    """A payload with an extra field (a newer writer) still parses, dropping the unknown key."""
    payload = HostFacts(hostname="future-host").model_dump_json()

    data = json.loads(payload)
    data["a_field_this_reader_has_never_heard_of"] = "some-value"
    restored = HostFacts.model_validate_json(json.dumps(data))
    assert restored.hostname == "future-host"
    assert not hasattr(restored, "a_field_this_reader_has_never_heard_of")
