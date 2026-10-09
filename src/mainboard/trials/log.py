"""The experiment-facing facade over trial receipts, the logger, artifacts, and profiling."""

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from copy import copy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from structlog.contextvars import bind_contextvars, reset_contextvars

from ..log import SINK, EventDict, logger, sinks
from ..observe.frames import Frame, Kind
from ..observe.spool import Spool
from ..profile.profiler import Collection, Profiler
from ..state.relations import ArrowStream, Relations, parquet_bytes
from .artifacts import Artifact, pinned
from .session import params_of
from .vocabulary import Outcome

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from pydantic import BaseModel, JsonValue

    from ..state.relations import Relation
    from .session import Trial

_LEVELS = frozenset(("debug", "info", "warning", "error", "critical", "exception"))
_IDENTITY = frozenset(("project", "node", "trial", "run", "params", SINK))
# What every event carries that is the logger's envelope rather than the caller's metadata.
_ENVELOPE = frozenset(("event", "level", "timestamp", "exc_info", "stack_info"))
# The media types an attached image or video is recorded under, by its file's suffix.
_IMAGES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".avif": "image/avif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}
_VIDEOS = {".mp4": "video/mp4", ".webm": "video/webm"}


class Log:
    """A trial-scoped logger; identity and artifact paths come from pytest, never retyped."""

    def __init__(self, trial: Trial) -> None:
        self.trial = trial
        session = trial.session
        universe = session.declared.universe
        path = Path(str(trial.item.path))
        manifest = session.manifest(path)
        if manifest is not None:
            trial.artifacts["run"] = manifest.model_dump(mode="json")
        node = universe.node_of(path)
        key = hashlib.sha256(trial.item.nodeid.encode()).hexdigest()
        directory = universe.dataset(node).root.parent / "artifacts" / session.run / key
        self.artifacts = session.store(node)
        self.spool = Spool(directory, "events")
        trial.artifacts["events"] = self.spool.dir.relative_to(self.artifacts.root).as_posix()
        self.counts: Counter[str] = Counter()
        self.metadata: dict[str, JsonValue] = {}
        self.identity = f"{session.run}/{key}"
        self.context: dict[str, JsonValue] = {
            **session.common,
            "project": universe.root.parent.name
            if universe.root.name == "experiments"
            else session.declared.tree.name,
            "node": node,
            "trial": trial.item.nodeid,
            "run": session.run,
            "params": params_of(trial.item, universe.axes),
            "manifest": manifest.model_dump(mode="json") if manifest is not None else None,
        }
        # Every line logged while the trial runs is kept with it, whichever library logged it;
        # the trial's own methods name the sink explicitly, so they are kept from any thread.
        sinks[self.identity] = self._message
        self.bound = bind_contextvars(**{SINK: self.identity})
        self.logger = logger.bind(**{SINK: self.identity})
        self._event("started", self.context, kind=Kind.started)

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Expose the logger's severity methods and the project's settlement vocabulary."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _LEVELS:
            return getattr(self.logger, name)
        if name in self.trial.session.declared.words:
            return partial(self.settle, name)
        raise AttributeError(name)

    def bind(self, **metadata: JsonValue) -> Log:
        """Bind diagnostic context without allowing it to replace provenance."""
        if _IDENTITY.intersection(metadata) or self.context.keys() & metadata.keys():
            raise ValueError("bound metadata cannot replace trial provenance")
        bound = copy(self)
        bound.metadata = {**self.metadata, **metadata}
        bound.logger = self.logger.bind(**metadata)
        return bound

    def metrics(self, **values: JsonValue) -> None:
        """Persist metrics immediately, independently of the eventual scientific verdict."""
        self._event("metrics", dict(values), kind=Kind.sample)

    def gate(self, registration: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Read a gate through the same registration digest used by existing trials."""
        return self.trial.gate(registration)

    def settle(self, word: str, reason: str = "", **measured: JsonValue) -> None:
        """Settle through the existing trial; logging is not a second outcome system."""
        if word not in self.trial.session.declared.words:
            raise ValueError(f"undeclared verdict: {word}")
        self._event("settled", {"verdict": word, "reason": reason, "measured": dict(measured)})
        self.trial.settle(word, reason=reason, **measured)

    def artifact(
        self,
        data: bytes | Path,
        *,
        name: str = "",
        media_type: str = "application/octet-stream",
        schema_name: str = "",
        source: str = "",
    ) -> Artifact:
        """Attach immutable bytes or a generated file; infer a name when none is meaningful.

        Bytes another trial already attached are referenced, not copied again.
        source: where pinned bytes were downloaded from (`hf://<repo>@<revision>/<file>`),
            read from the path itself for a Hugging Face hub cache file.
        """
        reference = self.artifacts.write(
            data.read_bytes() if isinstance(data, Path) else data,
            media_type=media_type,
            schema_name=schema_name,
            source=source or (pinned(data) if isinstance(data, Path) else ""),
        )
        label = name or self._name("artifact")
        self.trial.artifacts[label] = reference.model_dump()
        self._event("artifact", {"name": label, **reference.model_dump()})
        return reference

    def model(self, value: BaseModel, *, name: str = "", schema_name: str = "") -> Artifact:
        """Attach a model's JSON bytes, retaining the caller's name and schema."""
        return self.artifact(
            value.model_dump_json().encode(),
            name=name,
            media_type="application/json",
            schema_name=schema_name,
        )

    def table(
        self,
        rows: Relation | ArrowStream | Sequence[Mapping[str, JsonValue]],
        *,
        name: str = "",
        schema_name: str = "",
    ) -> Artifact:
        """Write a Parquet table: a DuckDB relation or an Arrow stream (a polars frame) as its
        own columns, or rows whose column types are read from all of them."""
        if isinstance(rows, duckdb.DuckDBPyRelation):
            relation = rows
        elif isinstance(rows, ArrowStream):
            relation = Relations().arrow(rows)
        else:
            relation = Relations().rows(rows)
        return self.artifact(
            parquet_bytes(relation),
            name=name or self._name("table"),
            media_type="application/vnd.apache.parquet",
            schema_name=schema_name,
        )

    def image(self, path: Path, *, name: str = "") -> Artifact:
        """Attach a rendered PNG, JPEG, AVIF, WebP or SVG without importing a plotting library."""
        return self.artifact(
            path, name=name or self._name("image"), media_type=_IMAGES[path.suffix.lower()]
        )

    def video(self, path: Path, *, name: str = "") -> Artifact:
        """Attach an encoded MP4 or WebM, AV1 or otherwise, as the bytes the encoder wrote."""
        return self.artifact(
            path, name=name or self._name("video"), media_type=_VIDEOS[path.suffix.lower()]
        )

    def read(self, alias: str) -> bytes:
        """Read an explicitly declared pinned input, recording its consumed identity."""
        reference = self.trial.session.declared.inputs[alias]
        data = reference.read(self.artifacts.root)
        self._event("input", {"alias": alias, **reference.model_dump()})
        return data

    def read_table(self, alias: str) -> Relation:
        """Read a pinned Parquet input as a DuckDB relation, no storage plumbing in sight."""
        return Relations().parquet(self.read(alias))

    @contextmanager
    def profile(
        self, *, name: str = "", collection: Collection | None = None
    ) -> Generator[Profiler]:
        """Capture with the existing profiler and attach evidence even when the body raises."""
        policy = collection or self.trial.session.declared.collection
        profiler = Profiler.under(policy)
        completed = False
        try:
            with profiler:
                yield profiler
            completed = True
        finally:
            self.model(
                profiler.result(),
                name=name or self._name("profile"),
                schema_name="mainboard.Profile",
            )
            self._event(
                "profile",
                {"completed": completed, "collection": json.loads(policy.model_dump_json())},
            )

    def close(self, *, passed: bool) -> None:
        """Flush the trial stream and remove only this logger's handler."""
        try:
            self._event(
                "ended", {"passed": passed, "verdict": self.trial.settled}, kind=Kind.ended
            )
            if self.trial.settled and self.trial.artifacts != self.trial.recorded_artifacts:
                reason = (
                    "Artifacts changed after the last receipt was settled; "
                    "settle only after output checks and cleanup evidence are attached"
                )
                self.trial.record("", reason=reason, measured={}, outcome=Outcome.FAILED)
                raise RuntimeError(reason)
        finally:
            sinks.pop(self.identity, None)
            reset_contextvars(**self.bound)
            self.spool.close()

    def _name(self, kind: str) -> str:
        self.counts[kind] += 1
        return f"{kind}-{self.counts[kind]}"

    def _message(self, event: EventDict) -> None:
        self._event(
            "message",
            {
                "text": str(event["event"]),
                "level": event["level"],
                "metadata": {key: value for key, value in event.items() if key not in _ENVELOPE},
            },
        )

    def _event(
        self, topic: str, payload: Mapping[str, JsonValue], *, kind: Kind = Kind.line
    ) -> None:
        self.spool.append(
            Frame.model_validate(
                {
                    "job": self.identity,
                    "kind": kind,
                    "at": datetime.now(UTC),
                    "payload": {
                        "topic": topic,
                        "trial": self.trial.item.nodeid,
                        "metadata": self.metadata,
                        "data": dict(payload),
                    },
                }
            )
        )
