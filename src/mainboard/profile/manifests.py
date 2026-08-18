# Perfetto merge-archive manifests: point several traces at one multi-host timeline, the
# `perfetto_manifest` version-1 JSON shape Perfetto's trace processor merge tooling reads.

import json
from os import PathLike
from pathlib import Path
from typing import cast

from patos import FrozenModel

from ..core.errors import MissionError
from .protocols import Json

_VERSION = 1


class TraceSource(FrozenModel):
    """One trace file entering the merge, and how it aligns to the merged clock.

    path: filesystem path to the trace file (native protobuf or Chrome JSON).
    machine_name: label for the host/process this trace came from, kept distinct per
        source so the merged timeline can tell them apart.
    clock: the clock domain this trace's timestamps are in (e.g. ``BOOTTIME``), empty
        when the trace already shares the merge's reference clock.
    sync_to_path: path of the trace this source's clock is synchronized against. Chrome
        JSON sources cannot self-align (their timestamps carry no clock-domain metadata),
        so :meth:`MergeManifest.render` requires this whenever `path` ends in ``.json``.
    offset_ns: fixed nanosecond offset applied after alignment, for a known clock skew.
    """

    path: str
    machine_name: str
    clock: str = ""
    sync_to_path: str = ""
    offset_ns: int = 0

    def render(self) -> dict[str, Json]:
        """This source as one entry of the manifest's ``files`` list."""
        entry: dict[str, Json] = {"path": self.path, "machine": {"name": self.machine_name}}
        if self.clock or self.sync_to_path:
            clocks: dict[str, Json] = {"sync_to": {"file": self.sync_to_path, "clock": self.clock}}
            if self.offset_ns:
                clocks["offset_ns"] = self.offset_ns
            entry["clocks"] = clocks
        return entry


class MergeManifest(FrozenModel):
    """A `perfetto_manifest` version-1 document: the sources to merge and shared metadata.

    sources: every trace file entering the merge.
    attributes: free-form labels attached to the merged trace (a job id, a run label).
    """

    sources: tuple[TraceSource, ...] = ()
    attributes: dict[str, str] = {}

    def render(self) -> dict[str, Json]:
        """The `{"perfetto_manifest": {...}}` document Perfetto's merge tool reads.

        Chrome-JSON sources cannot self-align (Chrome trace events carry no clock-domain
        metadata), so a `.json` source with no `sync_to_path` fails fast here rather than
        merging onto an arbitrary, unstated reference clock.
        """
        for source in self.sources:
            if source.path.endswith(".json") and not source.sync_to_path:
                raise MissionError(
                    f"trace source {source.path!r} is Chrome JSON, which carries no clock "
                    "metadata and cannot self-align; set sync_to_path (and clock) to the "
                    "trace it should synchronize against."
                )
        body: dict[str, Json] = {
            "version": _VERSION,
            "files": [source.render() for source in self.sources],
        }
        if self.attributes:
            # `attributes` is `dict[str, str]`, a narrower type than the `Json`-valued `body`;
            # every `str` value is itself a valid `Json`, so the cast only widens the type.
            body["attributes"] = cast("dict[str, Json]", dict(self.attributes))
        return {"perfetto_manifest": body}

    def write(self, path: str | PathLike[str]) -> None:
        """Write the rendered manifest as JSON to `path`."""
        Path(path).write_text(json.dumps(self.render(), indent=2), encoding="utf-8")
