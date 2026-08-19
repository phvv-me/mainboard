import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mainboard import MissionError
from mainboard.profile import MergeManifest, TraceSource

if TYPE_CHECKING:
    from mainboard.profile.protocols import Json


def _dict(value: Json) -> dict[str, Json]:
    """Narrow one `Json` value to a dict, for asserting into a nested render() result."""
    assert isinstance(value, dict)
    return value


def _list(value: Json) -> list[Json]:
    """Narrow one `Json` value to a list, for asserting into a nested render() result."""
    assert isinstance(value, list)
    return value


def test_trace_source_renders_path_and_machine() -> None:
    source = TraceSource(path="/traces/host.perfetto-trace", machine_name="gold")
    assert source.render() == {
        "path": "/traces/host.perfetto-trace",
        "machine": {"name": "gold"},
    }


def test_trace_source_renders_clock_sync_when_given() -> None:
    source = TraceSource(
        path="/traces/gpu.perfetto-trace",
        machine_name="gold-gpu",
        clock="BOOTTIME",
        sync_to_path="/traces/host.perfetto-trace",
    )
    rendered = source.render()
    assert rendered["clocks"] == {
        "sync_to": {"file": "/traces/host.perfetto-trace", "clock": "BOOTTIME"}
    }


def test_trace_source_renders_offset_ns_only_when_nonzero() -> None:
    source = TraceSource(
        path="/traces/a.perfetto-trace",
        machine_name="a",
        clock="BOOTTIME",
        sync_to_path="/traces/b.perfetto-trace",
        offset_ns=500,
    )
    assert _dict(source.render()["clocks"])["offset_ns"] == 500

    zero_offset = TraceSource(
        path="/traces/a.perfetto-trace",
        machine_name="a",
        clock="BOOTTIME",
        sync_to_path="/traces/b.perfetto-trace",
    )
    assert "offset_ns" not in _dict(zero_offset.render()["clocks"])


def test_merge_manifest_renders_version_one_shape() -> None:
    manifest = MergeManifest(
        sources=(TraceSource(path="/traces/a.perfetto-trace", machine_name="a"),)
    )
    rendered = _dict(manifest.render()["perfetto_manifest"])
    assert rendered["version"] == 1
    assert rendered["files"] == [{"path": "/traces/a.perfetto-trace", "machine": {"name": "a"}}]
    assert "attributes" not in rendered


def test_merge_manifest_includes_attributes_when_given() -> None:
    manifest = MergeManifest(
        sources=(TraceSource(path="/traces/a.perfetto-trace", machine_name="a"),),
        attributes={"job": "run-42"},
    )
    rendered = _dict(manifest.render()["perfetto_manifest"])
    assert rendered["attributes"] == {"job": "run-42"}


def test_merge_manifest_empty_sources_renders_an_empty_file_list() -> None:
    manifest = MergeManifest()
    rendered = _dict(manifest.render()["perfetto_manifest"])
    assert rendered["files"] == []


def test_chrome_json_source_without_sync_to_path_raises() -> None:
    manifest = MergeManifest(sources=(TraceSource(path="/traces/trace.json", machine_name="a"),))
    with pytest.raises(MissionError, match="cannot self-align"):
        manifest.render()


def test_chrome_json_source_with_sync_to_path_is_accepted() -> None:
    manifest = MergeManifest(
        sources=(
            TraceSource(path="/traces/host.perfetto-trace", machine_name="host"),
            TraceSource(
                path="/traces/trace.json",
                machine_name="chrome",
                clock="REALTIME",
                sync_to_path="/traces/host.perfetto-trace",
            ),
        )
    )
    rendered = _dict(manifest.render()["perfetto_manifest"])
    assert len(_list(rendered["files"])) == 2


def test_write_dumps_json_to_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = MergeManifest(
        sources=(TraceSource(path="/traces/a.perfetto-trace", machine_name="a"),)
    )
    manifest.write(path)
    data = json.loads(path.read_text())
    assert data["perfetto_manifest"]["version"] == 1
