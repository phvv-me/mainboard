# One batch declared as data: a small TOML file, or the same jobs typed as repeated flags. The
# declaration is the only input the three verbs share, so it carries everything they each need,
# what to run and where for `run`, what data must ship for `prepare`, and how long the command is
# expected to take for `estimate`.

import hashlib
import tomllib
from typing import TYPE_CHECKING, TypedDict

from patos import FrozenModel
from pydantic import ValidationError, model_validator

from ..core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# How a job typed at the command line names its target, `--job gold:python -m foo`. The first
# colon splits, since a host alias never carries one and a command routinely does.
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

    name: the job's label inside the batch, the key every table row and receipt line carries.
    target: the declared host alias (or provider alias) the job is dispatched to.
    command: the shell command the job runs.
    data: workspace paths this job needs on the target beyond the mirror, a dataset the manifest
        never syncs, so a transfer set counts them whether or not they changed.
    runtime_s: the command's expected wall seconds, which is what an estimate prices. Zero
        prices the setup alone, which is what a job whose runtime nobody has guessed costs
        before it starts.
    fetch: a results path pulled back when the job finishes.
    node: the ledger slug this job serves, carried into its record and receipts; absent stays
        a valid job.
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
        """This job's `Board.submit` keywords, the resource decisions it actually declares."""
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


class BatchSpec(FrozenModel):
    """A whole batch declared as data: a name and the jobs it fans across the fleet.

    name: the batch's human label, which its id is built from.
    jobs: the declared jobs, in the order they are prepared, priced and dispatched.
    """

    name: str
    jobs: tuple[BatchJob, ...]

    @property
    def batch_id(self) -> str:
        """This batch's identity: its name and a digest over every job it declares.

        Content-addressed so the same declaration always addresses the same receipts, which is
        what lets `prepare`, `estimate` and `run` write one stream and `watch` find it later by
        id alone. Changing what a job runs is a different batch and says so.
        """
        digest = hashlib.blake2s(self.model_dump_json().encode(), digest_size=4).hexdigest()
        return f"{self.name}-{digest}"

    @classmethod
    def inline(cls, name: str, declared: Sequence[str]) -> BatchSpec:
        """The batch `name` from `target:command` arguments, the file-free way to declare one.

        declared: one `target:command` per job, split at the first colon.
        """
        split = [job.partition(_INLINE) for job in declared]
        if bare := [job for job, separator, _ in split if not separator]:
            raise MissionError(f"jobs are written target:command, not {bare[0]!r}")
        return cls.of(
            name, [{"target": target, "command": command} for target, _, command in split]
        )

    @classmethod
    def load(cls, path: Path) -> BatchSpec:
        """The batch declared in the TOML file at `path`.

        A `[defaults]` table fills in every field a job leaves out, so a batch whose jobs share
        a walltime or an expected runtime says so once. The file's stem names the batch when its
        `name` key is absent.

        path: the spec file.
        """
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise MissionError(f"no batch spec at {path}") from None
        except tomllib.TOMLDecodeError as error:
            raise MissionError(f"{path} is not valid TOML: {error}") from None
        defaults = document.get("defaults", {})
        declared = [{**defaults, **job} for job in document.get("jobs", [])]
        return cls.of(document.get("name", path.stem), declared)

    @classmethod
    def of(cls, name: str, jobs: Sequence[dict[str, object]]) -> BatchSpec:
        """The batch `name` over already-parsed job tables, each validated where it is written.

        A job that named no label takes its target and position, so a spec stays as short as the
        decisions it actually makes while every row still has something to be called.
        """
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
