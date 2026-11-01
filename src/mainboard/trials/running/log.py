"""The experiment-facing facade over trial receipts, the logger, artifacts, and profiling."""

import hashlib
import json
from collections import Counter
from contextlib import contextmanager
from copy import copy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ...constants import SINK
from ...enums import Kind
from ...profile.profiler import Collection, Profiler
from ...state.relations import ArrowStream, Relations, parquet_bytes
from ..artifacts import Artifact, pinned
from ..session import params_of
from .stream import TrialStream

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    from pydantic import BaseModel, JsonValue

    from ...state.relations import Relation
    from ..session import Trial

_LEVELS = frozenset(("debug", "info", "warning", "error", "critical", "exception"))
_IDENTITY = frozenset(("project", "node", "trial", "run", "params", SINK))
# The media types an attached image is recorded under, by its file's suffix.
_IMAGES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".avif": "image/avif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


class Log:
    """A trial-scoped logger; identity and artifact paths come from pytest, never retyped.

    Every `bind` copy shares the one `stream`, which the pytest fixture closes with the trial.
    """

    @contextmanager
    def _profile_evidence(
        self, profiler: Profiler, name: str, policy: Collection
    ) -> Generator[None]:
        """Attach the profile once its body has run, whether the body completed or raised."""
        try:
            yield
        except BaseException:
            self._record_profile(profiler, name, policy, completed=False)
            raise
        self._record_profile(profiler, name, policy, completed=True)

    @contextmanager
    def profile(
        self, *, name: str = "", collection: Collection | None = None
    ) -> Generator[Profiler]:
        """Capture with the existing profiler and attach evidence even when the body raises."""
        policy = collection or self.trial.session.declared.collection
        profiler = Profiler.under(policy)
        with self._profile_evidence(profiler, name, policy), profiler:
            yield profiler

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Expose the logger's severity methods and the project's settlement vocabulary."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _LEVELS:
            return getattr(self.logger, name)
        if name in self.trial.session.declared.words:
            return partial(self.settle, name)
        raise AttributeError(name)

    def __init__(self, trial: Trial) -> None:
        self.trial = trial
        self.counts: Counter[str] = Counter()
        self.metadata: dict[str, JsonValue] = {}
        node, key, manifest = self._locate()
        self.artifacts = trial.session.store(node)
        self.context = self._context(node, manifest)
        self.stream = TrialStream(trial, self.context, node=node, key=key)
        self.logger = self.stream.logger

    def _context(self, node: str, manifest: Artifact | None) -> dict[str, JsonValue]:
        """The provenance every event of this trial carries, read from its session and item."""
        session = self.trial.session
        universe = session.declared.universe
        return {
            **session.common,
            "project": universe.root.parent.name
            if universe.root.name == "experiments"
            else session.declared.tree.name,
            "node": node,
            "trial": self.trial.item.nodeid,
            "run": session.run,
            "params": params_of(self.trial.item, universe.axes),
            "manifest": manifest.model_dump(mode="json") if manifest is not None else None,
        }

    def _event(
        self, topic: str, payload: Mapping[str, JsonValue], *, kind: Kind = Kind.line
    ) -> None:
        self.stream.event(topic, payload, metadata=self.metadata, kind=kind)

    def _locate(self) -> tuple[str, str, Artifact | None]:
        """The node and key that place this trial, with its run manifest recorded on it if any."""
        session = self.trial.session
        path = Path(str(self.trial.item.path))
        manifest = session.manifest(path)
        if manifest is not None:
            self.trial.artifacts["run"] = manifest.model_dump(mode="json")
        node = session.declared.universe.node_of(path)
        key = hashlib.sha256(self.trial.item.nodeid.encode()).hexdigest()
        return node, key, manifest

    def _name(self, kind: str) -> str:
        self.counts[kind] += 1
        return f"{kind}-{self.counts[kind]}"

    def _record_profile(
        self, profiler: Profiler, name: str, policy: Collection, *, completed: bool
    ) -> None:
        """Attach a profiler's result under its name, noting whether the body completed."""
        self.model(
            profiler.result(),
            name=name or self._name("profile"),
            schema_name="mainboard.Profile",
        )
        self._event(
            "profile",
            {"completed": completed, "collection": json.loads(policy.model_dump_json())},
        )

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

    def bind(self, **metadata: JsonValue) -> Log:
        """Bind diagnostic context without allowing it to replace provenance."""
        if _IDENTITY.intersection(metadata) or self.context.keys() & metadata.keys():
            raise ValueError("bound metadata cannot replace trial provenance")
        bound = copy(self)
        bound.metadata = {**self.metadata, **metadata}
        bound.logger = self.logger.bind(**metadata)
        return bound

    def image(self, path: Path, *, name: str = "") -> Artifact:
        """Attach a rendered PNG, JPEG, AVIF, WebP or SVG without importing a plotting library."""
        return self.artifact(
            path, name=name or self._name("image"), media_type=_IMAGES[path.suffix.lower()]
        )

    def metrics(self, **values: JsonValue) -> None:
        """Persist metrics immediately, independently of the eventual scientific verdict."""
        self._event("metrics", dict(values), kind=Kind.sample)

    def model(self, value: BaseModel, *, name: str = "", schema_name: str = "") -> Artifact:
        """Attach a model's JSON bytes, retaining the caller's name and schema."""
        return self.artifact(
            value.model_dump_json().encode(),
            name=name,
            media_type="application/json",
            schema_name=schema_name,
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

    def settle(self, word: str, reason: str = "", **measured: JsonValue) -> None:
        """Settle through the existing trial; logging is not a second outcome system."""
        if word not in self.trial.session.declared.words:
            raise ValueError(f"undeclared verdict: {word}")
        self._event("settled", {"verdict": word, "reason": reason, "measured": dict(measured)})
        self.trial.settle(word, reason=reason, **measured)

    def table(
        self,
        rows: Relation | ArrowStream | Sequence[Mapping[str, JsonValue]],
        *,
        name: str = "",
        schema_name: str = "",
    ) -> Artifact:
        """Write a Parquet table from a relation, a stream or rows.

        A DuckDB relation or an Arrow stream (a polars frame) keeps its own columns, and rows have
        their column types read from all of them.
        """
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
