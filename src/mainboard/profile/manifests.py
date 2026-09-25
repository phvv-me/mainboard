# Perfetto merge-archive manifests: the `perfetto_manifest` version-1 JSON that Perfetto's trace
# processor merge tooling reads to lay several traces on one multi-host timeline.

import json
from os import PathLike
from pathlib import Path

from patos import FrozenModel

from ..core.errors import MissionError
from .protocols import Json


class TraceSource(FrozenModel):
    """One trace file entering the merge, and how it aligns to the merged clock.

    path: the trace file, native protobuf or Chrome JSON.
    machine_name: the host/process label, kept distinct per source in the merged timeline.
    clock: this trace's clock domain (e.g. `BOOTTIME`), empty when it already shares the
        merge's reference clock.
    sync_to_path: the trace this source's clock is synchronized against. Chrome JSON carries
        no clock-domain metadata and cannot self-align, so a `.json` path requires it.
    offset_ns: fixed offset applied after alignment, for a known clock skew.
    """

    path: str
    machine_name: str
    clock: str = ""
    sync_to_path: str = ""
    offset_ns: int = 0

    def render(self) -> dict[str, Json]:
        """This source as one entry of the manifest's `files` list."""
        entry: dict[str, Json] = {"path": self.path, "machine": {"name": self.machine_name}}
        if self.clock or self.sync_to_path:
            clocks: dict[str, Json] = {"sync_to": {"file": self.sync_to_path, "clock": self.clock}}
            if self.offset_ns:
                clocks["offset_ns"] = self.offset_ns
            entry["clocks"] = clocks
        return entry


class MergeManifest(FrozenModel):
    """A `perfetto_manifest` version-1 document: the sources to merge and shared metadata.

    attributes: free-form labels attached to the merged trace (a job id, a run label).
    """

    sources: tuple[TraceSource, ...] = ()
    attributes: dict[str, str] = {}

    def render(self) -> dict[str, Json]:
        """The `{"perfetto_manifest": {...}}` document; an unaligned `.json` source fails fast."""
        for source in self.sources:
            if source.path.endswith(".json") and not source.sync_to_path:
                raise MissionError(
                    f"trace source {source.path!r} is Chrome JSON, which carries no clock "
                    "metadata and cannot self-align; set sync_to_path (and clock) to the "
                    "trace it should synchronize against."
                )
        body: dict[str, Json] = {
            "version": 1,
            "files": [source.render() for source in self.sources],
        }
        if self.attributes:
            body["attributes"] = dict[str, Json](self.attributes)
        return {"perfetto_manifest": body}

    def write(self, path: str | PathLike[str]) -> None:
        """Write the rendered manifest as JSON to `path`."""
        Path(path).write_text(json.dumps(self.render(), indent=2), encoding="utf-8")
