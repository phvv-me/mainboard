# Which stream and job a dispatched run's receipts belong to, the one router from a dispatch label.

from ..batch.runner import labelled_batch
from ..experiments.identity import labelled_study, labelled_trial


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
