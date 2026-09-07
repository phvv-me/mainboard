"""The experiment-facing facade over trial receipts, Loguru, artifacts, and profiling."""

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from copy import copy
from datetime import UTC, datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from ..observe.frames import Frame, Kind
from ..observe.spool import Spool
from ..profile.profiler import Collection, Profiler
from .artifacts import Artifact, Artifacts
from .session import params_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from loguru import Message, Record
    from pydantic import JsonValue

    from .session import Trial

_LEVELS = frozenset(
    ("trace", "debug", "info", "success", "warning", "error", "critical", "exception")
)
_IDENTITY = frozenset(("project", "node", "trial", "run", "params", "mainboard_trial"))


class Log:
    """A trial-scoped logger; identity and artifact paths come from pytest, never retyped."""

    def __init__(self, trial: Trial) -> None:
        self.trial = trial
        session = trial.session
        universe = session.declared.universe
        manifest = session.manifest(Path(str(trial.item.path)))
        if manifest is not None:
            trial.artifacts["run"] = manifest.model_dump(mode="json")
        node = universe.node_of(Path(str(trial.item.path)))
        key = hashlib.sha256(trial.item.nodeid.encode()).hexdigest()
        directory = universe.dataset(node).root.parent / "artifacts" / session.run / key
        self.artifacts = Artifacts(session.declared.tree, directory)
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
            "params": params_of(trial.item),
            "manifest": manifest.model_dump(mode="json") if manifest is not None else None,
        }
        self.logger = logger.bind(mainboard_trial=self.identity)
        self.sink = logger.add(
            self._message, filter=self._owned, level=0, catch=False, diagnose=False
        )
        self._event("started", self.context, kind=Kind.started)

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Expose Loguru severity methods and the project's existing settlement vocabulary."""
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
    ) -> Artifact:
        """Attach immutable bytes or a generated file; infer a name when none is meaningful."""
        reference = self.artifacts.write(
            data.read_bytes() if isinstance(data, Path) else data,
            media_type=media_type,
            schema_name=schema_name,
        )
        label = name or self._name("artifact")
        self.trial.artifacts[label] = reference.model_dump()
        self._event("artifact", {"name": label, **reference.model_dump()})
        return reference

    def table(
        self,
        rows: pl.DataFrame | Sequence[Mapping[str, JsonValue]],
        *,
        name: str = "",
        schema_name: str = "",
    ) -> Artifact:
        """Write a Parquet table readable directly by Polars or DuckDB."""
        frame = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
        buffer = BytesIO()
        frame.write_parquet(buffer, compression="zstd")
        return self.artifact(
            buffer.getvalue(),
            name=name or self._name("table"),
            media_type="application/vnd.apache.parquet",
            schema_name=schema_name,
        )

    def image(self, path: Path, *, name: str = "") -> Artifact:
        """Attach a rendered PNG, JPEG, or SVG without importing a plotting library."""
        types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        return self.artifact(
            path, name=name or self._name("image"), media_type=types[path.suffix.lower()]
        )

    def read(self, alias: str) -> bytes:
        """Read an explicitly declared pinned input, recording its consumed identity."""
        reference = self.trial.session.declared.inputs[alias]
        data = reference.read(self.artifacts.root)
        self._event("input", {"alias": alias, **reference.model_dump()})
        return data

    def read_table(self, alias: str) -> pl.DataFrame:
        """Read a pinned Parquet input without giving experiments storage plumbing."""
        return pl.read_parquet(BytesIO(self.read(alias)))

    @contextmanager
    def profile(
        self, *, name: str = "", collection: Collection | None = None
    ) -> Iterator[Profiler]:
        """Capture with the existing profiler and attach evidence even when the body raises."""
        policy = collection or self.trial.session.declared.collection
        profiler = Profiler.under(policy)
        completed = False
        try:
            with profiler:
                yield profiler
            completed = True
        finally:
            result = profiler.result()
            self.artifact(
                result.model_dump_json().encode(),
                name=name or self._name("profile"),
                media_type="application/json",
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
        finally:
            logger.remove(self.sink)
            self.spool.close()

    def _name(self, kind: str) -> str:
        self.counts[kind] += 1
        return f"{kind}-{self.counts[kind]}"

    def _owned(self, record: Record) -> bool:
        return record["extra"].get("mainboard_trial") == self.identity

    def _message(self, message: Message) -> None:
        record = message.record
        self._event(
            "message",
            {
                "text": record["message"],
                "level": record["level"].name,
                "metadata": {
                    key: value
                    for key, value in record["extra"].items()
                    if key != "mainboard_trial"
                },
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
