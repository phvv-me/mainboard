import shlex

from plumbum import local as localhost

from ...batch.estimate.core import JobEstimate
from ...core.errors import MissionError
from ...dispatch.jobs.commandline import joined, vetted
from ...dispatch.jobs.vocabulary import Request
from ...jobs.testing import lanes as lanes_module
from ...jobs.testing.lanes import Group
from ...log import logger
from ...render import progress, totals
from ..base import Verbs
from ..options import Asked, Declared, Dispatching, Output
from .jobs import JobVerbs, Waiting

# The verbs' defaults: compact output, the profile's resources, a lone job, no batch.
_COMPACT = Output()
_ASKED = Asked()
_HOW = Dispatching()
_SPEC_ONLY = Declared()

# The columns a batch's price and its dispatch print, so an empty table still has its heading.
_ESTIMATE_COLUMNS = (
    "job",
    "target",
    "kind",
    "hardware",
    "wire_bytes",
    "runtime_s",
    "setup_p50_s",
    "setup_p90_s",
    "setup_samples",
    "rate_usd_hr",
    "rate_source",
    "expected_usd",
    "p90_usd",
)
_DISPATCH_COLUMNS = ("job", "target", "state", "handle", "kind", "reason")

# What every lane cell's pytest runs with: no reordering, and the short report a log reads.
_PYTEST = ("-p", "no:randomly", "-q", "--no-header")


class SubmitVerbs(Verbs):
    """The one verb that dispatches: a job, a batch, or a test lane split over hosts."""

    jobs: JobVerbs

    @staticmethod
    def _expected(priced: JobEstimate, *, results: str = "") -> str:
        """One line saying where a submit lands, what it brings home, and what the meter will read.

        A rate means a rented target and carries its tail cost. No rate names why there is none,
        so a machine this workspace owns reads as `owned` while a provider nobody could get a
        price out of says that instead, and neither prints a bare zero that looks like a
        promise. Where the rate came from rides beside it for the same reason: a live offer can
        be rented at that price and a stored one is last week's.

        A run that pulls nothing home says so out loud, since the only moment that is cheap to
        notice is before the job goes out rather than after it wrote a wave of receipts on a
        cluster.

        results: the path this dispatch pulls back, empty when it declared none.
        """
        hardware = f", {priced.hardware}" if priced.hardware else ""
        where = f"{priced.target} ({priced.kind}{hardware})"
        pulled = (
            f"results {results}" if results else "results NOT pulled back (no --fetch, no --node)"
        )
        if not priced.rate_usd_hr:
            meter = f"{priced.rate_source}, expected $0.00"
        else:
            meter = (
                f"${priced.rate_usd_hr:.2f}/hr ({priced.rate_source}), "
                f"expected ${priced.expected_usd:.2f} (p90 ${priced.p90_usd:.2f})"
            )
        return f"submit -> {where}: queue policy ok, {pulled}, {meter}"

    def _batch(self, declared: Declared, *, how: Dispatching, asked: Asked) -> int:
        """Price, or dispatch, every job a batch declares; print the batch id and each row."""
        batched = declared.planned(
            self.session.board("local"), name=how.name, resources=asked.given()
        )
        if how.estimate:
            with progress(f"pricing {batched.id}"):
                priced = [row.model_dump() for row in batched.estimate().jobs]
            # The total rides in the table, since every mode renders rows and a figure printed
            # beside them would be the one number `--json` dropped.
            summed = ("wire_bytes", "runtime_s", "expected_usd", "p90_usd")
            total = totals(priced, columns=_ESTIMATE_COLUMNS, summing=summed)
            title = f"estimate: {batched.id}"
            _COMPACT.print_rows([*priced, total], title=title, columns=_ESTIMATE_COLUMNS)
            return 0
        with progress(f"dispatching {batched.id}") as stage:
            dispatched = [entry.model_dump() for entry in batched.run(watch=stage)]
        logger.result(f"# batch {batched.id}")
        title = f"run: {batched.id}"
        _COMPACT.print_rows(dispatched, title=title, columns=_DISPATCH_COLUMNS)
        return self.jobs.show(batched.id, waiting=Waiting(wait=True)) if how.wait else 0

    def _cell(
        self, request: Request, group: Group, *, host: str, how: Dispatching
    ) -> tuple[str, str, str, int]:
        """Run one group of a lane's cells on `host`: its row and a local run's exit code.

        The target's own tokens are kept apart, since joined whole they reached the host as one
        (B43), and each cell runs as a fresh process under the lane's timeout.
        """
        fresh = ("--fresh", "--timeout", str(how.cell_timeout), *group.ids)
        line = [*shlex.split(request.command), "--", *fresh, "--", *_PYTEST]
        if host == "local":
            ran = self.session.board("local").run(line)
            return host, group.name, f"local exit {ran}", ran
        named = {"target": host, "command": joined(line), "name": f"lane-{host}-{group.name}"}
        with progress(f"submitting {group.name} on {host}") as stage:
            run = self.session.board(host).dispatch(request.model_copy(update=named), watch=stage)
        return host, group.name, run.handle.id, 0

    def _groups(self, target: str, how: Dispatching) -> tuple[Group, ...]:
        """A lane's cells, collected here and grouped by the split value or sliced."""
        with progress(f"collecting {target}"):
            lanes = "mainboard.jobs.testing.collector"
            probe = ["run", "--", "python", "-m", lanes, "collect", target]
            cells = lanes_module.parsed(localhost[self.session.project.package][probe]())
        if not cells:
            raise MissionError(f"{target} collected no cells")
        return lanes_module.grouped(cells, by=how.split, per_job=how.per_job)

    def _lane(self, request: Request, how: Dispatching) -> int:
        """Run a test lane's cells on each host, one job per group; `local` runs them in place.

        The lane's parametrization is the plan, and every handle is waited on, a failure among
        them or not (B39).
        """
        groups = self._groups(request.command, how)
        _COMPACT.print_rows(lanes_module.summary(how.hosts, groups), title="lane")
        if how.estimate:
            return 0
        if not how.proceeds():
            raise SystemExit(1)
        served = request.model_copy(
            update={"node": request.node or lanes_module.node_of(request.command)}
        )
        rows = [
            self._cell(served, group, host=host, how=how) for host in how.hosts for group in groups
        ]
        dispatched = [
            {"host": host, "group": group, "handle": shown} for host, group, shown, _ in rows
        ]
        _COMPACT.print_rows(dispatched, title="dispatched")
        waited = [
            self.jobs.show(shown, on=host, waiting=Waiting(wait=True, quiet=True))
            for host, _, shown, _ in (rows if how.wait else ())
            if host != "local"
        ]
        return next((code for *_, code in rows if code), 0) or next((s for s in waited if s), 0)

    def _one(self, request: Request, *, how: Dispatching, output: Output) -> int:
        """Price one job, ask before dispatching it, and print its handle."""
        board = self.session.board(request.target)
        pulled = board.results(request.fetch, node=request.node, command=request.command)
        logger.info(self._expected(board.expectation(request), results=pulled))
        if how.estimate:
            return 0
        if not how.proceeds():
            raise SystemExit(1)
        with progress(f"submitting on {request.target}") as stage:
            run = board.dispatch(request, watch=stage)
        if output.mode is None:
            logger.result(run.handle.id)
        else:
            output.print_record(run.handle.model_dump(), title="handle")
        return (
            self.jobs.show(run.handle.id, on=request.target, waiting=Waiting(wait=True))
            if how.wait
            else 0
        )

    def _request(self, command: tuple[str, ...], *, how: Dispatching, asked: Asked) -> Request:
        """The dispatch the tokens ask for, or the run `--resume` names asked for again.

        A resumed run keeps its command (unless one is given), its name, host, node and results
        path, and runs as its next attempt.
        """
        if not how.resume:
            if not how.on:
                raise MissionError("name a host: --on <alias>, or a batch with --batch")
            return asked.request(
                joined(command), target=how.on, name=how.name, attempt=how.attempt
            )
        cache = self.session.board("local").dispatcher.cache
        previous = cache.run(how.resume, how.on or None)
        target = how.on or previous.target
        kept = {
            "node": asked.node or previous.node,
            "fetch": asked.fetch or previous.fetch_path or "",
        }
        return asked.model_copy(update=kept).request(
            joined(command) if command else vetted(previous.script),
            target=target,
            name=previous.label,
            attempt=cache.count(name=previous.label, target=target) + 1,
        )

    def submit(
        self,
        *command: str,
        how: Dispatching = _HOW,
        asked: Asked = _ASKED,
        declared: Declared = _SPEC_ONLY,
        output: Output = _COMPACT,
    ) -> int:
        """Dispatch one job, a batch, or a test lane split over hosts, and print the handles.

        One job: a command, or `path/to/file.py::name` (which ships only the code it imports),
        on `--on HOST`. A batch: `--batch spec.toml`, or `--job target:command` repeated, each
        job to its own target; a target refusing is that job's row and the rest still go. A
        lane: `--on a,b --split model path/to/test.py::test` runs the test's cells as one job
        per `model` value on every named host (`--per-job N` slices instead).

        What it will cost prints first (target, queue admission, the meter); at a terminal it
        asks once, and `--yes` or a script proceeds. `--estimate` stops there, dispatching
        nothing. `--wait` blocks until everything submitted settles and exits its verdict.

        Every job gets `MB_CHECKPOINT`, a directory on its host that outlives the run, and
        `MB_ATTEMPT`. `--resume <handle|name>` submits that run again (same command, name, host,
        node and results path, today's code) as its next attempt, so the job finds in
        `MB_CHECKPOINT` whatever the failed attempt saved there; the lake keeps every attempt.

        Args:
            command: the command tokens, from the first token that is not an option of this verb,
                or `path/to/file.py::name` and, after `--`, the arguments its application takes.
        """
        if declared.batch or declared.job:
            return self._batch(declared, how=how, asked=asked)
        request = self._request(command, how=how, asked=asked)
        if how.is_lane:
            return self._lane(request, how)
        return self._one(request, how=how, output=output)
