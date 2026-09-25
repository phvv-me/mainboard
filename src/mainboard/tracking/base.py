# The sink half of the receipts contract. A batch, a plain submit and a study all publish the
# same `Event` stream, so mirroring any of them is one `Bus` implementation, enrolled by
# subclassing `Tracker` as a provider backend subclasses `ProviderBackend`. Nothing here names a
# service.

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

    A write-only `Bus`: it never answers a replay, because a resumed pass's cursor must come from
    the workspace's own files. A second service joins by subclassing and implementing `publish`.

    stream: the receipts stream being mirrored, a batch id, a study id or one run's own name.
    directory: where the stream's own files live, and a queued offline copy goes.
    workspace: the workspace's name, the project a sink falls back to when none was declared.
    """

    # The one environment variable a host needs before a job there can ship its own samples, so a
    # dispatch stages exactly that; the only way a service's variable name reaches the tool.
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
        return []


def mirrored(
    canonical: Bus,
    declared: Tracking,
    *,
    stream: str,
    directory: Path,
    workspace: str = "",
) -> Bus:
    """`canonical` (the workspace's own transport) mirrored best-effort into the declared sink.

    The one composition every dispatch path shares, so one `mode = "off"` turns all of them off.
    """
    if not declared.on:
        return canonical
    return Mirrored(
        canonical, sink(stream, declared=declared, directory=directory, workspace=workspace)
    )


def credential(declared: Tracking) -> str:
    """The environment variable the declared sink needs on a host, empty when it needs none."""
    if not declared.on:
        return ""
    return Tracker.find(declared.provider).credential


def streamed(name: str, *, handle: str) -> tuple[str, str]:
    """The `(stream, job)` a dispatched run belongs to, the one router from a dispatch label.

    Every run resolves to something, since a machine asked to watch itself must be told which job.

    name: the run's dispatch label, empty for a run nobody named.
    handle: the run's scheduler or provider handle, which names an unlabelled run.
    """
    if inside := labelled_batch(name):
        stream, _, job = inside.partition("/")
        return stream, job or stream
    if study := labelled_study(name):
        return study, labelled_trial(name) or study
    return (name, name) if name else (f"run-{handle}", handle)


def is_batched(name: str) -> bool:
    """Whether a batch's own flow already publishes every receipt about this run.

    A second publisher would double every row; the job still samples itself, the one thing only
    the machine running it can say.
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
