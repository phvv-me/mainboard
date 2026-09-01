import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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

_GIB = 1024**3


def test_collected_reads_every_probe_into_one_wire_portable_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot carries everything a remote dispatcher sizes jobs with.

    The cgroup ceiling, the scratch tier and each GPU's dispatch key all land on it. A GPU
    whose architecture is still unknown reports no key at all rather than the literal word.
    """

    class NamedGPU(GPU):
        """A GPU whose provider filled in a real architecture, unlike the unnamed base."""

        @property
        def arch_key(self) -> str:
            return "sm_90"

        @property
        def label(self) -> str:
            return "NVIDIA GH200"

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
    monkeypatch.setattr(type(machine), "gpus", (GPU(index=0), NamedGPU(index=1)))
    monkeypatch.setattr(Fabric, "probe", classmethod(lambda cls: ()))

    facts = HostFacts.collected()
    assert facts.schema_version == 1
    assert (facts.cgroup.limit_bytes, facts.cgroup.capped) == (100 * _GIB, True)
    assert (facts.scratch.path, facts.scratch.source) == (str(Path("/local")), "LOCALDIR")
    assert facts.scratch.free_bytes == 50 * _GIB
    assert facts.memory_total_bytes == 200 * _GIB
    assert facts.cpu_name == machine.cpu.label
    assert facts.cpu_logical_cores == machine.host.logical_cpus
    assert facts.scheduler is Scheduler.PBS
    assert [(gpu.name, gpu.arch_key) for gpu in facts.gpus] == [
        ("unknown", None),
        ("NVIDIA GH200", "sm_90"),
    ]


def test_an_unavailable_scratch_tier_serializes_its_path_as_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scratchless host serializes a JSON null, never the word `None`.

    A host with nowhere writable has no scratch path, and the reader on the far end has to
    see a null there rather than the string a plain conversion would have written.
    """
    monkeypatch.setattr(type(Machine().host), "scratch", Scratch())
    facts = HostFacts.collected()
    assert facts.scratch.path is None
    assert json.loads(facts.model_dump_json())["scratch"]["path"] is None


# Building a whole nested snapshot per example is the expensive part, and a round trip either
# holds for every shape or fails on the first, so a small budget buys the same confidence.
@settings(max_examples=15)
@given(facts=st.from_type(HostFacts))
def test_a_snapshot_round_trips_through_json_unchanged(facts: HostFacts) -> None:
    """A snapshot survives the wire byte for byte.

    The whole point of the model is crossing a wire, so dumping and validating it back has to
    reproduce an equal snapshot for any host it could describe, nested GPUs and ports
    included.
    """
    assert HostFacts.model_validate_json(facts.model_dump_json()) == facts


def test_a_payload_from_a_newer_writer_still_parses() -> None:
    """An older reader tolerates keys it has never heard of.

    Adding a field does not bump the schema version, so an unknown key must not fail the
    whole snapshot.
    """
    payload = json.loads(
        HostFacts(hostname="gold", fabric=(FabricPort(device="mlx5_0", port=1),)).model_dump_json()
    )
    payload["a_field_this_reader_has_never_heard_of"] = "some-value"

    restored = HostFacts.model_validate_json(json.dumps(payload))
    assert restored.hostname == "gold"
    assert restored.fabric == (FabricPort(device="mlx5_0", port=1),)
    assert not hasattr(restored, "a_field_this_reader_has_never_heard_of")
