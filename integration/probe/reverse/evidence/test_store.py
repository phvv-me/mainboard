import inspect
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from sqlalchemy import func, select

from mainboard.probe.reverse import CuobjdumpProvider, Evidence, EvidenceStore, ReverseProvider
from mainboard.probe.reverse.evidence import (
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


@pytest.fixture
def store(lake: Lake) -> Iterator[EvidenceStore]:
    """An evidence store over the lake, whose session closes at teardown."""
    found = EvidenceStore(lake)
    yield found
    found.session.close()


def test_store_roundtrip_and_duplicate_ingest_is_a_noop(store: EvidenceStore) -> None:
    original = evidence({"nested": {"z": 1, "a": 2}})
    key = store.ingest(original)
    assert (key, store.read(key), store.ingest(original)) == (original.key(), original, key)
    assert store.session.rows(select(func.count()).select_from(schema.re_evidence)) == [(1,)]
    assert store.query(artifact_sha256=original.artifact.sha256, operation="inventory") == (
        original,
    )
    assert not store.query(provider="absent")
    assert store.read("absent") == NotFound(evidence_id="absent")
    stored_result = store.session.rows(select(schema.re_evidence.c.result_json))[0][0]
    assert original.result.model_validate_json(stored_result) == original.result


@pytest.fixture
def version_seven(lake: Lake, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create the lake as version 7 did, with no evidence table or view."""
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


@pytest.mark.usefixtures("version_seven")
def test_version_seven_evolves_through_the_existing_schema_loop(lake: Lake) -> None:
    assert lake.query(select(func.max(schema.schema_log.c.version))) == [(7,)]
    lake.current()
    assert lake.query(select(func.max(schema.schema_log.c.version))) == [(lake_module.VERSION,)]
    store = EvidenceStore(lake)
    original = evidence()
    assert store.read(store.ingest(original)) == original
    store.session.close()


@pytest.fixture
def key(store: EvidenceStore) -> str:
    """The cache address of the default evidence, once ingested."""
    return store.ingest(evidence())


def test_current_view_follows_append_order_and_keeps_file_evidence_separate(
    store: EvidenceStore, key: str
) -> None:
    replacement = evidence().model_copy(update={"raw": "latest"})
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


_INVENTORY = KernelInventory(kernels=(), arches=())


@pytest.mark.parametrize(
    "result",
    [
        evidence().result,
        ArchCoverage(arches=()),
        SassResult(listings=()),
        _INVENTORY.diff(_INVENTORY),
    ],
    ids=["evidence", "arch-coverage", "sass", "inventory-diff"],
)
def test_all_device_result_shapes_survive_json(result: DeviceResult) -> None:
    original = evidence()
    record = Evidence[DeviceResult].model_validate({**original.model_dump(), "result": result})
    restored = Evidence[DeviceResult].model_validate_json(record.model_dump_json())
    assert restored.result == result


@pytest.fixture
def libraries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Two ELF libraries of different bytes, whose every operation the fixture runner answers."""

    class Supporter:
        @classmethod
        def supporting(
            cls: type[Supporter], kind: str, artifact: ArtifactKind, operation: str
        ) -> ReverseProvider:
            return CuobjdumpProvider(FixtureRunner())

    monkeypatch.setattr(
        ReverseProvider, "supporting", inspect.getattr_static(Supporter, "supporting")
    )
    baseline, newer = tmp_path / "13.4.so", tmp_path / "13.8.so"
    header = b"\x7fELF\x02\x01" + bytes(58)
    baseline.write_bytes(header + b"older")
    newer.write_bytes(header + b"newer")
    return baseline, newer


def test_compare_collects_and_stores_its_source_inventories(
    lake: Lake, libraries: tuple[Path, Path]
) -> None:
    baseline, newer = libraries
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
