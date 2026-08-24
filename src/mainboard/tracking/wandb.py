# The Weights and Biases sink, and the only module in this package that names the service.
#
# THE MAPPING, receipts on the left, what a run is told on the right:
#   batch.opened    the stream's own context, carried into every run this sink later opens
#   job.prepared    history under `prepared/`
#   job.estimated   history under `estimated/`, and the same fields on the summary
#   job.submitted   history under `submitted/`, and the handle, target, kind and command as config
#   job.state       history under `state/`, which is the scheduler's word and our verdict
#   job.sample      history under `sample/`, the live machine series the node itself publishes
#   job.cost        history under `cost/`, and the same fields on the summary
#   job.refused     the summary's `refused`, then the run is closed with exit code 1
#   job.settled     the summary's verdict and detail, then the run is closed with its exit code
#   batch.closed    the stream's context again, since a batch-wide line belongs to no run
#
# THE IDENTITY. One run per job, and its wandb id is our own content-addressed digest over
# (stream, job), so the process that dispatches a job and the sweep that settles it hours later
# from another machine resume the same run instead of minting two. That is also why the step is
# seeded from the run's own position on resume rather than from a counter this process keeps.
#
# WHAT NEVER HAPPENS HERE. No key is read, printed or logged (`Credentials` merges the workspace
# `.env` and this module only asks whether the variable is now set), and nothing raises out of
# `publish`, because `Mirrored` treats this whole module as best effort and a batch must never
# die because a dashboard did.

import os
from importlib import import_module
from typing import TYPE_CHECKING, Protocol

from ..batch.receipts import Topic
from ..core.errors import MissionError
from ..dispatch.backends.base import Credentials
from ..dispatch.shared import logger
from ..dispatch.vocabulary import OK
from ..experiments.identity import run_id
from ..manifest.schema.tracking import TrackingMode
from .base import Tracker

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from types import ModuleType

    from pydantic import JsonValue

    from ..batch.receipts import Event
    from ..manifest.schema.tracking import Tracking

# The variable the service itself reads, and the only thing this module ever checks about it.
_KEY = "WANDB_API_KEY"

# What a topic's fields also belong on, the run's summary, so a finished run reads as a row
# rather than as a series somebody has to scrub to the end of.
_SUMMARIZED = frozenset({Topic.COST, Topic.ESTIMATED})

# The topics that end a run, and the exit code each one closes it with when the receipt carries
# none of its own. A refusal never ran at all, which is a failure of the dispatch rather than of
# the command, and both read as a non-zero close.
_CLOSING: dict[Topic, int] = {Topic.SETTLED: 1, Topic.REFUSED: 1}


class Fields(Protocol):
    """A run's config or summary, both of which take one mapping update."""

    def update(self, values: Mapping[str, JsonValue], **options: bool) -> None: ...


class Tracked(Protocol):
    """The slice of a run this sink drives, so nothing here is typed against the SDK."""

    @property
    def config(self) -> Fields: ...

    @property
    def step(self) -> int: ...

    @property
    def summary(self) -> Fields: ...

    @property
    def url(self) -> str: ...

    def finish(self, exit_code: int | None = None) -> None: ...

    def log(self, data: Mapping[str, JsonValue], step: int | None = None) -> None: ...


class WandbSink(Tracker):
    """One Weights and Biases run per job, resumed by our own content-addressed id.

    Every job of a stream shares that stream as its group, so a batch reads as one row of runs
    and a study reads as one row of trials. What a run carries is exactly what the receipts said,
    with the envelope's identity as config, each topic's scalars as history, and the terminal
    line closing the run with the exit code the verdict implies.

    The mode is resolved rather than obeyed: an `online` workspace with no key on this machine
    queues offline instead of blocking on a login prompt no dispatched job could ever answer, and
    says which directory `wandb sync` should drain. That is what makes a compute node with no
    egress work without configuring anything there.
    """

    name = "wandb"
    credential = _KEY

    def __init__(
        self, stream: str, *, declared: Tracking, directory: Path, workspace: str = ""
    ) -> None:
        """The stream this sink mirrors, with its runs, step cursors and context still empty."""
        super().__init__(stream, declared=declared, directory=directory, workspace=workspace)
        self.runs: dict[str, Tracked] = {}
        self.steps: dict[str, int] = {}
        self.context: dict[str, JsonValue] = {}

    @property
    def mode(self) -> str:
        """`online` only where a key actually is, since a keyless online run cannot open at all.

        The workspace `.env` is merged first, which is where the key lives on the machine that
        dispatches, and a node that never got one falls through to a queued offline run rather
        than to a failure. Only the variable's presence is read, never its value.
        """
        if self.declared.mode is TrackingMode.OFFLINE:
            return TrackingMode.OFFLINE
        Credentials().load()
        if os.environ.get(_KEY):
            return TrackingMode.ONLINE
        logger.info(
            "no %s here, so %s is tracked offline; drain it later with `wandb sync %s/wandb`",
            _KEY,
            self.stream,
            self.directory,
        )
        return TrackingMode.OFFLINE

    @property
    def project(self) -> str:
        """The project runs land in: what the manifest declared, else this workspace's name."""
        return self.declared.project or self.workspace or self.stream

    def close(self, job: str, *, exit_code: int) -> None:
        """End `job`'s run at `exit_code` and forget it, so a later line opens a fresh resume.

        Only ever reached with a run this sink just logged to, since `publish` opens the run
        before it reads the topic, so there is nothing here to guard against.
        """
        self.steps.pop(job)
        self.runs.pop(job).finish(exit_code=exit_code)

    @staticmethod
    def flattened(event: Event) -> dict[str, JsonValue]:
        """`event`'s scalar payload, keyed by its topic, the shape a history row is written in.

        A list-valued field (the paths a transfer names) is left to the receipts file, which is
        canonical and holds it whole. Only what can be a series becomes one.
        """
        prefix = event.topic.split(".")[-1]
        return {
            f"{prefix}/{field}": value
            for field, value in event.data.items()
            if isinstance(value, str | int | float | bool)
        }

    def publish(self, event: Event) -> None:
        """Tell this stream's runs what one receipt said, opening or closing a run as it says to.

        A batch-wide line names no job, so it is context every run this sink later opens carries
        rather than a row in any one of them.
        """
        if not event.job:
            self.context.update(dict(event.data))
            return
        run = self.run(event.job)
        measured = self.flattened(event)
        run.log(measured, step=self.step(event.job))
        if event.topic is Topic.SUBMITTED:
            run.config.update(dict(event.data), allow_val_change=True)
        if event.topic in _SUMMARIZED:
            run.summary.update(measured)
        if event.topic in _CLOSING:
            run.summary.update(measured)
            self.close(event.job, exit_code=exit_code(event))

    def run(self, job: str) -> Tracked:
        """`job`'s run, resumed by its content-addressed id or opened here for the first time.

        The id is ours rather than the service's, which is the whole reason a dispatch here and a
        sweep on another machine tomorrow write to one run. The step cursor is seeded from where
        the resumed run already stands, so a second process continues the series instead of
        rewriting its beginning.
        """
        if job in self.runs:
            return self.runs[job]
        identity = run_id({"stream": self.stream, "job": job})
        opened: Tracked = module().init(
            id=identity,
            name=job,
            group=self.stream,
            project=self.project,
            entity=self.declared.entity or None,
            mode=self.mode,
            dir=str(self.directory),
            config={"stream": self.stream, "job": job, "run_id": identity, **self.context},
            resume="allow",
            reinit="create_new",
        )
        self.runs[job] = opened
        self.steps[job] = opened.step
        return opened

    def step(self, job: str) -> int:
        """`job`'s next history position, advancing the cursor this sink keeps for it."""
        at = self.steps[job]
        self.steps[job] = at + 1
        return at


def exit_code(event: Event) -> int:
    """What a closing receipt says the run ended at: its own code, or what its verdict implies."""
    reported = event.data.get("exit_code")
    if isinstance(reported, int):
        return reported
    if event.data.get("verdict") == OK:
        return 0
    return _CLOSING[event.topic]


def module() -> ModuleType:
    """The imported `wandb` module, refusing with the one command that installs it.

    The import is here rather than at the top of the file because tracking is on by default and
    this package must stay installable without the service, so a workspace that never wanted the
    lane pays nothing for it and a workspace that did is told exactly what to run.
    """
    try:
        return import_module("wandb")
    except ImportError:
        raise MissionError(
            "tracking declares the wandb provider but the `wandb` package is not installed; "
            'run `mainboard add wandb -l python --dev`, or set [tracking] mode = "off"'
        ) from None
