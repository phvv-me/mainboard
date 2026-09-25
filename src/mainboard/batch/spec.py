# One batch declared as data: a small TOML file, or the same jobs typed as repeated flags. It is
# the only input the verbs share, so it carries what to run and where (`run`), what data must ship
# (`prepare`) and how long the command should take (`estimate`).

import hashlib
import tomllib
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, TypedDict

from patos import FrozenModel
from pydantic import ValidationError, model_validator

from ..core.errors import MissionError
from ..manifest.render.interpolate import Interpolator, Json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# `--job gold:python -m foo`: the first colon splits, since a host alias never carries one.
_INLINE = ":"


class Submission(TypedDict, total=False):
    """The `Board.submit` keywords a job declares, every unset one left to the host profile."""

    queue: str
    walltime: str
    mem_gb: int
    gpus: int
    gpu_name: str
    max_usd: float
    nodes: int
    env: str
    container: str
    fetch: str | None
    node: str


class BatchJob(FrozenModel):
    """One job of a batch: what runs, where, and what it needs that the target lacks.

    name: the key every table row and receipt line carries.
    target: the host (or provider) alias the job is dispatched to.
    data: workspace paths needed on the target beyond the mirror (a dataset the manifest never
        syncs), so a transfer set counts them whether or not they changed.
    runtime_s: the command's expected wall seconds an estimate prices; zero prices setup alone.
    fetch: a results path pulled back when the job finishes.
    node: the ledger slug this job serves, carried into its record and receipts, or empty.
    """

    name: str = ""
    target: str
    command: str
    data: tuple[str, ...] = ()
    runtime_s: float = 0.0
    queue: str = ""
    walltime: str = ""
    mem_gb: int = 0
    gpus: int = 0
    gpu_name: str = ""
    max_usd: float = 0.0
    nodes: int = 1
    env: str = ""
    container: str = ""
    fetch: str = ""
    node: str = ""

    def submission(self) -> Submission:
        """This job's `Board.submit` keywords."""
        return Submission(
            queue=self.queue,
            walltime=self.walltime,
            mem_gb=self.mem_gb,
            gpus=self.gpus,
            gpu_name=self.gpu_name,
            max_usd=self.max_usd,
            nodes=self.nodes,
            env=self.env,
            container=self.container,
            fetch=self.fetch or None,
            node=self.node,
        )


class Selection(FrozenModel):
    """Which of a plan's jobs a verb acts on, by name or `fnmatch` glob, empty for all.

    A plan is worked through in waves (nine corpora ready, four not), and the pick is what a verb
    was asked rather than what the batch is, so it lives beside the declaration and the plan keeps
    its identity and receipts stream across every wave.
    """

    patterns: tuple[str, ...] = ()

    @classmethod
    def of(cls, typed: str) -> Selection:
        """The selection typed as one comma-separated option, blank entries dropped."""
        return cls(patterns=tuple(part.strip() for part in typed.split(",") if part.strip()))

    def chosen(self, jobs: Sequence[BatchJob]) -> tuple[BatchJob, ...]:
        """The declared jobs this selection names, in the plan's own order.

        A pattern naming nothing is refused with what the plan declares, since a mistyped name is
        a wave quietly going out short.
        """
        if not self.patterns:
            return tuple(jobs)
        unmatched = [
            pattern
            for pattern in self.patterns
            if not any(fnmatchcase(job.name, pattern) for job in jobs)
        ]
        if unmatched:
            declared = ", ".join(job.name for job in jobs)
            raise MissionError(
                f"no job named {unmatched[0]!r} in this batch; it declares {declared}"
            )
        return tuple(job for job in jobs if self.holds(job.name))

    def holds(self, name: str) -> bool:
        return not self.patterns or any(fnmatchcase(name, pattern) for pattern in self.patterns)


def _table(value: Json, *, at: str) -> dict[str, Json]:
    """`value` as the table a spec declares at `at`, refusing any other shape by name."""
    if not isinstance(value, dict):
        raise MissionError(f"[{at}] must be a table")
    return value


def _tables(value: Json, *, at: str) -> list[dict[str, Json]]:
    """`value` as the array of tables a spec declares at `at`, refusing any other shape."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise MissionError(f"[[{at}]] must be an array of tables")
    return [item for item in value if isinstance(item, dict)]


class BatchSpec(FrozenModel):
    """A whole batch declared as data: a name and the jobs, in the order they are prepared,
    priced and dispatched."""

    name: str
    jobs: tuple[BatchJob, ...]

    @property
    def batch_id(self) -> str:
        """This batch's name and a digest over every job it declares.

        Content-addressed so `prepare`, `estimate` and `run` write one stream `watch` later finds
        by id alone; changing what a job runs is a different batch.
        """
        digest = hashlib.blake2s(self.model_dump_json().encode(), digest_size=4).hexdigest()
        return f"{self.name}-{digest}"

    @classmethod
    def inline(cls, name: str, declared: Sequence[str]) -> BatchSpec:
        """The batch `name` from one `target:command` argument per job, the file-free way."""
        split = [job.partition(_INLINE) for job in declared]
        if bare := [job for job, separator, _ in split if not separator]:
            raise MissionError(f"jobs are written target:command, not {bare[0]!r}")
        return cls.of(
            name, [{"target": target, "command": command} for target, _, command in split]
        )

    @classmethod
    def load(cls, path: Path, overrides: Mapping[str, str] | None = None) -> BatchSpec:
        """The batch declared in the TOML file at `path`, named by its stem without a `name`.

        `[defaults]` fills every field a job leaves out. `[vars]` declares knobs every string may
        render as `mainboard.toml` does (`{{ vars.repetition }}`); `overrides` typed at the
        command line replace them, so one spec serves every repetition, and an undeclared name is
        refused rather than rendered blank.
        """
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise MissionError(f"no batch spec at {path}") from None
        except tomllib.TOMLDecodeError as error:
            raise MissionError(f"{path} is not valid TOML: {error}") from None
        declared_vars = document.get("vars", {})
        if unknown := sorted(set(overrides or ()) - set(declared_vars)):
            raise MissionError(
                f"{path} declares no [vars] named {', '.join(unknown)}; "
                f"it knows {', '.join(sorted(declared_vars)) or 'none'}"
            )
        document["vars"] = {**declared_vars, **(overrides or {})}
        rendered = Interpolator(path.parent).rendered(document)
        defaults = _table(rendered.get("defaults", {}), at="defaults")
        jobs = _tables(rendered.get("jobs", []), at="jobs")
        return cls.of(str(rendered.get("name", path.stem)), [{**defaults, **job} for job in jobs])

    @classmethod
    def of(cls, name: str, jobs: Sequence[dict[str, object]]) -> BatchSpec:
        """The batch `name` over parsed job tables, unnamed jobs named by target and position."""
        try:
            built = [BatchJob.model_validate(job) for job in jobs]
        except ValidationError as error:
            raise MissionError(f"batch {name!r} declares an unusable job:\n{error}") from None
        return cls(
            name=name,
            jobs=tuple(
                job if job.name else job.model_copy(update={"name": f"{job.target}-{at}"})
                for at, job in enumerate(built, start=1)
            ),
        )

    @model_validator(mode="after")
    def names_are_unique(self) -> BatchSpec:
        """Refuse two jobs under one name, since every receipt line is keyed by it."""
        seen = [job.name for job in self.jobs]
        if len(set(seen)) != len(seen):
            raise ValueError(f"job names repeat in batch {self.name!r}: {sorted(seen)}")
        if not seen:
            raise ValueError(f"batch {self.name!r} declares no jobs")
        return self
