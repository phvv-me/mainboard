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


def _body(manifest: MergeManifest) -> dict[str, Json]:
    """The `perfetto_manifest` body of a rendered document."""
    return _dict(manifest.render()["perfetto_manifest"])


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            TraceSource(path="/traces/host.perfetto-trace", machine_name="gold"),
            {"path": "/traces/host.perfetto-trace", "machine": {"name": "gold"}},
        ),
        (
            TraceSource(
                path="/traces/gpu.perfetto-trace",
                machine_name="gold-gpu",
                clock="BOOTTIME",
                sync_to_path="/traces/host.perfetto-trace",
            ),
            {
                "path": "/traces/gpu.perfetto-trace",
                "machine": {"name": "gold-gpu"},
                "clocks": {
                    "sync_to": {"file": "/traces/host.perfetto-trace", "clock": "BOOTTIME"}
                },
            },
        ),
        (
            TraceSource(
                path="/traces/a.perfetto-trace",
                machine_name="a",
                clock="BOOTTIME",
                sync_to_path="/traces/b.perfetto-trace",
                offset_ns=500,
            ),
            {
                "path": "/traces/a.perfetto-trace",
                "machine": {"name": "a"},
                "clocks": {
                    "sync_to": {"file": "/traces/b.perfetto-trace", "clock": "BOOTTIME"},
                    "offset_ns": 500,
                },
            },
        ),
    ],
    ids=["already_on_the_reference_clock", "aligned_to_another_trace", "aligned_with_a_skew"],
)
def test_a_source_renders_its_path_machine_and_clock_alignment(
    source: TraceSource, expected: dict[str, Json]
) -> None:
    """Clock alignment appears only when stated, and a zero skew is left out entirely."""
    assert source.render() == expected


@pytest.mark.parametrize(
    ("manifest", "files", "attributes"),
    [
        (MergeManifest(), [], None),
        (
            MergeManifest(
                sources=(TraceSource(path="/traces/a.perfetto-trace", machine_name="a"),)
            ),
            [{"path": "/traces/a.perfetto-trace", "machine": {"name": "a"}}],
            None,
        ),
        (
            MergeManifest(
                sources=(TraceSource(path="/traces/a.perfetto-trace", machine_name="a"),),
                attributes={"job": "run-42"},
            ),
            [{"path": "/traces/a.perfetto-trace", "machine": {"name": "a"}}],
            {"job": "run-42"},
        ),
    ],
    ids=["no_sources", "one_source", "labelled_merge"],
)
def test_the_manifest_renders_the_version_one_shape(
    manifest: MergeManifest, files: list[Json], attributes: dict[str, str] | None
) -> None:
    """Every document states version one and its files, and labels only when it has some."""
    body = _body(manifest)
    assert body["version"] == 1
    assert body["files"] == files
    assert body.get("attributes") == attributes


def test_a_chrome_json_source_must_name_what_it_synchronizes_against() -> None:
    """Chrome trace events carry no clock metadata, so a `.json` source cannot self-align.

    Failing fast here beats merging onto an arbitrary, unstated reference clock, and the
    same source is accepted the moment it says which trace it aligns to.
    """
    alone = MergeManifest(sources=(TraceSource(path="/traces/trace.json", machine_name="a"),))
    with pytest.raises(MissionError, match="cannot self-align"):
        alone.render()

    aligned = MergeManifest(
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
    files = _body(aligned)["files"]
    assert isinstance(files, list)
    assert len(files) == 2


def test_write_dumps_json_to_path(tmp_path: Path) -> None:
    """`write` puts the rendered document on disk as JSON."""
    path = tmp_path / "manifest.json"
    manifest = MergeManifest(
        sources=(TraceSource(path="/traces/a.perfetto-trace", machine_name="a"),)
    )
    manifest.write(path)
    assert json.loads(path.read_text())["perfetto_manifest"]["version"] == 1
