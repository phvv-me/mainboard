# The sink half of the receipts contract. A batch, a plain submit and a study all publish the
# same `Event` stream, so mirroring any of them somewhere else is one implementation of `Bus`
# rather than three integrations, and this module is where such an implementation is found.
#
# Nothing here names a service. `Tracker` is the registry root a sink enrolls in by subclassing,
# exactly as a provider backend enrolls under `ProviderBackend`, and `mirrored` is the one
# composition every dispatch path goes through: the workspace's own file first, the declared
# sink beside it, best effort.

import abc
from typing import TYPE_CHECKING, ClassVar

from patos import Registry

from ..batch.receipts import Mirrored
from ..batch.runner import labelled_batch
from ..core.errors import MissionError
from ..experiments.identity import labelled_study, labelled_trial

if TYPE_CHECKING:
    from pathlib import Path

    from ..batch.receipts import Bus, Event
    from ..manifest.schema.tracking import Tracking


class Tracker(Registry, abc.ABC):
    """Registry root for the services a workspace mirrors its receipts into.

    A tracker is a write-only `Bus`: it takes every event the flow publishes and it never
    answers a replay, because a cursor a resumed pass reads must come from the workspace's own
    files. Enrolling is subclassing this and implementing `publish`, so a second service joins
    without an edit anywhere else.

    stream: the receipts stream being mirrored, a batch id, a study id or one run's own name.
    declared: the `[tracking]` table this workspace wrote.
    directory: where the stream's own files live, which is where a queued offline copy goes.
    workspace: the workspace's name, the project a sink falls back to when none was declared.
    """

    # The one environment variable a host needs before a job there can ship its own samples, so
    # a dispatch can stage exactly that and nothing else. A sink needing none declares none, and
    # this is the only way a service's variable name reaches the rest of the tool.
    credential: ClassVar[str] = ""

    def __init__(
        self, stream: str, *, declared: Tracking, directory: Path, workspace: str = ""
    ) -> None:
        self.stream = stream
        self.declared = declared
        self.directory = directory
        self.workspace = workspace

    @abc.abstractmethod
    def publish(self, event: Event) -> None:
        """Ship one event to the service, however that service spells it."""

    def replay(self) -> list[Event]:
        """Nothing, since a sink is where events go and never where they come from."""
        return []


def mirrored(
    canonical: Bus,
    declared: Tracking,
    *,
    stream: str,
    directory: Path,
    workspace: str = "",
) -> Bus:
    """`canonical` with the declared sink mirroring it, or `canonical` alone when none is.

    The one composition every dispatch path shares, so a batch, a plain submit and a study are
    tracked by the same code and a workspace turns all three off with one `mode = "off"`.

    canonical: the workspace's own transport, which always receives every event.
    declared: the `[tracking]` table.
    stream: the receipts stream being mirrored.
    directory: where the stream's files live.
    workspace: the workspace's name, a sink's fallback project.
    """
    if not declared.on:
        return canonical
    return Mirrored(
        canonical,
        sink(stream, declared=declared, directory=directory, workspace=workspace),
    )


def credential(declared: Tracking) -> str:
    """The environment variable the declared sink needs on a host, empty when it needs none.

    The one door a service's variable name comes through, so nothing outside a sink module ever
    spells it and a dispatch stages what the workspace declared rather than what it assumed.
    """
    if not declared.on:
        return ""
    return Tracker.find(declared.provider).credential


def streamed(name: str, handle: str) -> tuple[str, str]:
    """The `(stream, job)` one dispatched run belongs to.

    The one router from a dispatch label to a receipts stream, so a plain submit, a study trial
    and a batch job all reach the same sink and the same run without any caller re-reading a
    label shape. Every run resolves to something, since a machine asked to watch itself has to
    be told which job it is watching.

    name: the run's dispatch label, empty for a run nobody named.
    handle: the run's scheduler or provider handle, which names an unlabelled run.
    """
    if inside := labelled_batch(name):
        stream, _, job = inside.partition("/")
        return stream, job or stream
    if study := labelled_study(name):
        return study, labelled_trial(name) or study
    return (name, name) if name else (f"run-{handle}", handle)


def batched(name: str) -> bool:
    """Whether a batch already publishes every line about this run, so nothing else should.

    Its own flow writes every receipt about a batch job, from the dispatch through the sweep
    that settles it, and a second publisher would double every row. The job still samples
    itself, since that is the one thing only the machine running it can say.

    name: the run's dispatch label.
    """
    return bool(labelled_batch(name))


def sink(stream: str, *, declared: Tracking, directory: Path, workspace: str = "") -> Tracker:
    """The tracker `declared` names, refusing an unregistered provider with the roster."""
    try:
        found = Tracker.find(declared.provider)
    except KeyError:
        raise MissionError(
            f"no tracking provider {declared.provider!r}; registered providers are "
            f'{Tracker.names()}. Set [tracking] provider, or mode = "off".'
        ) from None
    return found(stream, declared=declared, directory=directory, workspace=workspace)
