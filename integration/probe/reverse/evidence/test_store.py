from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from sqlalchemy import func, select

from mainboard.probe.reverse import CuobjdumpProvider, Evidence, EvidenceStore, ReverseProvider
from mainboard.probe.reverse.models import (
    ArchCoverage,
    ArtifactKind,
    DeviceResult,
    InventoryDiff,
    KernelInventory,
    SassResult,
)
from mainboard.probe.reverse.service import DeviceProbe
from mainboard.probe.reverse.store import NotFound
from mainboard.state import Lake, schema
from mainboard.state.lake import maintenance as lake_module
from mainboard.state.lake import quoted

from ..builders import evidence
from ..fakes import FixtureRunner


@pytest.fixture
def lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lake]:
    # Like the integration lake scenarios, use real DuckLake in a throwaway workspace. Loading
    # preinstalled extensions is allowed; unlike the production loader this cannot install any.
    def load_only(self: Lake, connection: duckdb.DuckDBPyConnection, names: Sequence[str]) -> None:
        connection.execute(f"SET extension_directory = {quoted(self.extensions.as_posix())}")
        for name in names:
            connection.load_extension(name)

    monkeypatch.setattr(Lake, "_load", load_only)
    found = Lake(root=tmp_path, home=tmp_path / "lake-home", served="")
    found.home.mkdir()
    yield found
    found.session().close()


def test_store_roundtrip_and_duplicate_ingest_is_a_noop(lake: Lake) -> None:
    store = EvidenceStore(lake)
    original = evidence({"nested": {"z": 1, "a": 2}})
    key = store.ingest(original)
    assert key == original.key()
    assert store.read(key) == original
    assert store.ingest(original) == key
    assert store.session.rows(select(func.count()).select_from(schema.re_evidence)) == [(1,)]
    assert store.query(artifact_sha256=original.artifact.sha256, operation="inventory") == (
        original,
    )
    assert not store.query(provider="absent")
    assert store.read("absent") == NotFound(evidence_id="absent")
    stored_result = store.session.rows(select(schema.re_evidence.c.result_json))[0][0]
    assert original.result.model_validate_json(stored_result) == original.result
    store.session.close()


def test_version_seven_evolves_through_the_existing_schema_loop(
    lake: Lake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as old:
        old.setattr(lake_module, "VERSION", 7)
        old.setattr(
            schema,
            "tables",
            tuple(table for table in schema.tables if table is not schema.re_evidence),
        )
        old.setattr(
            schema,
            "views",
            tuple(view for view in schema.views if view.table is not schema.re_evidence_current),
        )
        lake.create()
    assert lake.query(select(func.max(schema.schema_log.c.version))) == [(7,)]
    lake.current()
    assert lake.query(select(func.max(schema.schema_log.c.version))) == [(8,)]
    store = EvidenceStore(lake)
    original = evidence()
    assert store.read(store.ingest(original)) == original
    store.session.close()


def test_current_view_follows_append_order_and_keeps_file_evidence_separate(lake: Lake) -> None:
    store = EvidenceStore(lake)
    original = evidence()
    key = store.ingest(original)
    replacement = original.model_copy(update={"raw": "latest"})
    store.session.append(
        schema.re_evidence,
        [
            {
                "ts": datetime(2000, 1, 1, tzinfo=UTC),
                "evidence_id": key,
                "record_json": replacement.model_dump(mode="json"),
                "dropped": False,
            }
        ],
    )
    restored = store.read(key)
    match restored:
        case Evidence():
            assert restored.raw == "latest"
        case NotFound():
            pytest.fail("current evidence is missing")
    assert store.session.rows(select(func.count()).select_from(schema.evidence)) == [(0,)]
    store.session.append(schema.re_evidence, [{"evidence_id": key, "dropped": True}])
    assert store.read(key) == NotFound(evidence_id=key)
    store.session.close()


def test_all_device_result_shapes_survive_json() -> None:
    original = evidence()
    inventory = KernelInventory(kernels=(), arches=())
    for result in (
        original.result,
        ArchCoverage(arches=()),
        SassResult(listings=()),
        inventory.diff(inventory),
    ):
        record = Evidence[DeviceResult].model_validate({**original.model_dump(), "result": result})
        restored = Evidence[DeviceResult].model_validate_json(record.model_dump_json())
        assert restored.result == result


def test_compare_collects_and_stores_its_source_inventories(
    lake: Lake,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def supporting(
        cls: type[ReverseProvider], kind: str, artifact: ArtifactKind, operation: str
    ) -> ReverseProvider:
        return CuobjdumpProvider(FixtureRunner())

    monkeypatch.setattr(ReverseProvider, "supporting", classmethod(supporting))
    baseline, newer = tmp_path / "13.4.so", tmp_path / "13.8.so"
    header = b"\x7fELF\x02\x01" + bytes(58)
    baseline.write_bytes(header + b"older")
    newer.write_bytes(header + b"newer")
    probe = DeviceProbe(lake)
    found = probe.run(baseline, "compare", against=str(newer))
    match found.result:
        case InventoryDiff():
            assert not found.result.added and not found.result.removed and not found.result.changed
        case _:
            pytest.fail("expected inventory comparison")
    assert len(probe.store.query(operation="inventory")) == 2
    assert probe.store.read(found.key()) == found
    assert found.params["before"] != found.params["after"]
    probe.store.session.close()
