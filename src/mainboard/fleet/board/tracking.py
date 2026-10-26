import json
from typing import TYPE_CHECKING

from ...batch.receipts import Topic, publish
from ...core.errors import MissionError
from ...core.section import Section
from ...dispatch.hosts.onboard import (
    facts_command,
    gpus_command,
    read_facts,
)
from ...dispatch.jobs.vocabulary import Resources
from ...dispatch.transport.shells import open_shell
from ...probe.readings.occupancy import Occupancy
from ...probe.readings.snapshot import HostFacts
from ...runtime.job import walltime_seconds
from ...tracking import Sampler, attesting, is_batched, sampling, streamed
from ..fitness import Fitness
from ..runs import Run
from .environments import Environments

if TYPE_CHECKING:
    from ...probe.system import System
    from ...runtime.job import ToolCall


class Tracking(Environments):
    """What a run records about itself and its machine: receipts, samples, attestations, facts."""

    def announce(self, label: str, run: Run, *, command: str, host: str, node: str = "") -> None:
        """Open this run's own receipts stream, so a dispatch outside a batch is tracked too.

        A batch publishes its own submissions and is skipped, so no line is written twice.

        label: the run's dispatch label, which says both where it belongs and who publishes it.
        command: what the job runs, recorded as this run's config.
        node: the ledger slug the run serves, carried on the line only when one was declared.
        """
        if is_batched(label):
            return
        stream, job = streamed(label, handle=run.handle.id)
        publish(
            self.receipts(stream),
            stream,
            Topic.SUBMITTED,
            job=job,
            data={
                "handle": run.handle.id,
                "target": host,
                "kind": run.handle.kind,
                "command": command,
                **({"node": node} if node else {}),
            },
        )

    def attest(self, stream: str, *, job: str) -> None:
        """Publish one attestation of this machine into `stream`'s receipts.

        The synchronous, once-only twin of `samples`. It reads the machine it runs on, so a
        dispatched job attests the node doing the work, the only reading that describes the
        measurement's conditions.
        """
        Sampler(self.receipts(stream), stream=stream, job=job, interval=0.0).attest()

    def attesting(self, tracked: tuple[str, str]) -> ToolCall | None:
        """The call this job makes to attest its own machine, None when sampling is off.

        Gated like `sampling`, but with no interval: an attestation happens exactly once and its
        whole value is that it happens before the work.

        tracked: the stream and job the attestation belongs to.
        """
        if not self.manifest.tracking.on:
            return None
        stream, job = tracked
        return attesting(stream=stream, job=job)

    def facts(self) -> HostFacts:
        """The host's probed hardware facts as the versioned wire snapshot.

        A remote host answers with the tool `install` put there, so the probe never depends on
        this workspace's mainboard importing under whatever interpreter the host ships.
        """
        if self.local:
            return HostFacts.collected(self.root)
        with open_shell(self.plan(container="none"), self.remote_root()) as shell:
            return read_facts(shell.run(facts_command(), activate=True))

    def findings(self, system: System) -> list[Section]:
        """What this host's software census means for this workspace, one judged row each.

        The judge `compute`, `setup` and `doctor --center` share, so a driver below the CUDA floor
        is the same row whichever verb found it.
        """
        return Fitness(self.root, self.manifest).judge(system, host=self.host)

    def occupancy(self) -> Occupancy:
        """Who holds each card of this host right now, local or through the host's own tool.

        A scheduler host answers for its card-less login node, not its allocations; ask `jobs`
        for what runs there.
        """
        if self.local:
            return Occupancy.collected()
        with open_shell(self.plan(container="none"), self.remote_root()) as shell:
            text = shell.run(gpus_command(), activate=True)
        line = next((line for line in reversed(text.splitlines()) if line.startswith("{")), "")
        if not line:
            raise MissionError(f"no occupancy in the probe output: {text.strip()[-240:]}")
        readings = json.loads(line)
        # This release keys the readings by host (`local` on the far side); an older one did not.
        return Occupancy.model_validate(readings.get("local", readings))

    def samples(
        self,
        stream: str,
        *,
        job: str,
        interval: float = 0.0,
        seconds: float = 0.0,
        parent: int = 0,
    ) -> Sampler:
        """This machine read into `stream`'s receipts for as long as the block runs.

        The in-process half of the live lane; a dispatched job gets the same through the CLI.

        interval: seconds between readings, the manifest's own when 0.
        seconds: a hard stop, 0 to sample until the caller stops it.
        parent: a process to end with, 0 for none.
        """
        return Sampler(
            self.receipts(stream),
            stream=stream,
            job=job,
            interval=interval or self.manifest.tracking.interval,
            seconds=seconds,
            parent=parent,
        )

    def sampling(self, tracked: tuple[str, str], *, resources: Resources) -> ToolCall | None:
        """The call this job makes so it samples itself, None when nothing samples it.

        tracked: the stream and job the samples belong to.
        resources: the resolved request, whose walltime bounds the sampler as it bounds the job.
        """
        declared = self.manifest.tracking
        if not declared.on:
            return None
        stream, job = tracked
        return sampling(
            stream=stream,
            job=job,
            interval=declared.interval,
            seconds=walltime_seconds(resources.walltime) if resources.walltime else 0.0,
        )
