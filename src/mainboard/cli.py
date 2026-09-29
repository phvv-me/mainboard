import os
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from importlib import metadata
from json import dumps, loads
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn

from cyclopts import App, Parameter
from plumbum import local as localhost
from pydantic import JsonValue
from rich.console import Console

from . import staleness, upkeep
from .batch.spec import BatchSpec, Selection
from .board import Board
from .center.migrate import Migration
from .center.standalone import Standalone
from .center.verify import Verification
from .ci import LocalLeg, Matrix, Package
from .completion import powershell
from .context.resolver import Resolver
from .core.errors import MissionError, NoWorkspace
from .core.project import Project
from .core.section import Section, Verdict, failed
from .core.shell import become
from .delimiter import Delimiter
from .dispatch import keys, vocabulary
from .dispatch.commandline import joined, vetted
from .dispatch.evidence import printed
from .dispatch.schedulers import HostUnreachable, standing
from .durable import schedule
from .engines.compile.provisioner import Provisioner
from .help import Help
from .holds import Holds
from .jobs import lanes as lanes_module
from .lint import Inventory, Linter
from .listing import Listing
from .log import configure
from .manifest.loading import composition, load, load_plot_config
from .manifest.schema.plot import PlotStyle
from .probe.occupancy import rows as occupancy_rows
from .probe.system import System
from .proc import Processes
from .render import diverted, mode_of, plain, progress, record, rows, totals
from .render.values import to_row
from .runtime.job import Job
from .runtime.runner import Runner
from .state import Importer, Lake
from .state.lake import PORT
from .vigil import STALL_SECONDS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from .batch.estimate import JobEstimate
    from .batch.runner import Batch
    from .batch.watch import BatchStatus
    from .ci import Result as CiResult
    from .deps import Change
    from .dispatch.onboard import HostSetup
    from .dispatch.state import MonitorReport
    from .engines.compile.backend.pixi import Pixi
    from .git import Step
    from .manifest.held import Held
    from .manuscript import Report as PaperReport
    from .render.values import Node
    from .verdicts import StreamVerdict


# The values a compact record leaves out: they say nothing an absent field does not.
_EMPTY: tuple[object, ...] = (None, "", [], {}, ())


@Parameter(name="*")
@dataclass(frozen=True, kw_only=True)
class Output:
    """How a verb prints its document: compact rows for agents by default, JSON, or rich tables.

    The default is what an agent reads cheapest: a header line of field names, then one
    tab-separated line per row, no colour, no box drawing.
    """

    json: bool = False
    """print canonical JSON instead of the compact rows."""
    human: bool = False
    """print rich tables for a person at a terminal instead of the compact rows."""
    fields: str = ""
    """a comma-separated projection over the printed fields."""

    def __post_init__(self) -> None:
        """Refuse both modes at once before the verb does any work."""
        mode_of(json_mode=self.json, human=self.human)

    @property
    def mode(self) -> str | None:
        """The render key, `None` for the compact default."""
        return mode_of(json_mode=self.json, human=self.human)

    def projection(self, default: Sequence[str] = ()) -> Sequence[str]:
        """The `--fields` names, trimmed and blanks dropped, `default` when none were given."""
        return tuple(part.strip() for part in self.fields.split(",") if part.strip()) or default

    def print_rows(
        self, payloads: Sequence[Mapping[str, Node]], *, title: str, columns: Sequence[str] = ()
    ) -> None:
        """Print many entities, `columns` keeping an empty table's heading unless projected.

        The compact default leaves out a column empty in every row: it tells an agent nothing
        and costs a tab per row. `--fields` keeps exactly the columns it names.
        """
        fields = self.projection(columns)
        if self.mode is None and not self.fields and payloads:
            flat = [to_row(payload) for payload in payloads]
            fields = [
                name
                for name in fields or flat[0]
                if any(row.get(name) not in (None, "") for row in flat)
            ]
        rows(payloads, mode=self.mode, fields=fields, title=title)

    def print_record(self, payload: Mapping[str, Node], *, title: str) -> None:
        """Print one entity; the compact default leaves out its empty fields."""
        if self.mode is None and not self.fields:
            payload = {key: value for key, value in payload.items() if value not in _EMPTY}
        record(payload, mode=self.mode, fields=self.projection(), title=title)


_COMPACT = Output()


@Parameter(name="*")
@dataclass(frozen=True, kw_only=True)
class Declared:
    """A batch's declaration beyond its spec file: inline jobs, a selection, `[vars]` values."""

    job: tuple[str, ...] = ()
    """a `target:command` job, repeatable, for a batch declared without a file."""
    only: str = ""
    """the plan's jobs to act on, names or `kind-*` globs, comma-separated; all when unset."""
    set_: Annotated[tuple[str, ...], Parameter(name="--set", negative="")] = ()
    """a `name=value` filling one of the spec file's `[vars]`, repeatable."""

    def batch(self, board: Board, spec: str, *, name: str = "") -> Batch:
        """The declared batch over `board`'s workspace, `spec` relative to its root."""
        return board.batch(self._spec(board.root, spec, name), selection=Selection.of(self.only))

    def _spec(self, root: Path, spec: str, name: str) -> BatchSpec:
        if spec:
            return BatchSpec.load(root / spec, _answers(self.set_))
        if self.set_:
            raise MissionError("--set fills a spec file's [vars]; a --job batch declares none")
        if not self.job:
            raise MissionError("declare a batch: --batch <spec file>, or --job target:command")
        return BatchSpec.inline(name or "batch", self.job)


_SPEC_ONLY = Declared()


def build(root: Path | None = None) -> App:
    """The CLI application, workspace discovery deferred until a verb runs.

    root: an explicit workspace root, discovered from the cwd when None.
    """
    project = Project()
    # Plain help: box drawing and colour cost an agent tokens and tell a person nothing more.
    app = App(
        name=project.name,
        help="One interface for environments, dispatch, and hardware.",
        help_formatter="plain",
        # No `--no-<flag>` twin for every boolean: every flag here defaults to off.
        default_parameter=Parameter(negative=()),
        console=Console(
            color_system=None, highlight=False, width=None if sys.stdout.isatty() else 160
        ),
    )
    host = App(name="host", help="Set up, list, inspect and reach the machines jobs run on.")
    job = App(name="job", help="Dispatch jobs to hosts and follow them until they settle.")
    self_ = App(name="self", help=f"Manage the {project.name} installation itself.")
    lake = App(name="lake", help="Keep the workspace's state lake: check, compact, serve.")
    app.command(host)
    app.command(job)
    paper = App(name="paper", help="Build a declared manuscript and plot figures from results.")
    app.command(lake)
    app.command(paper)
    app.command(self_)

    def workspace_root() -> Path:
        return root or project.find_root(Path.cwd())

    def board(on: str) -> Board:
        return Board(workspace_root(), host=on)

    @app.command(name="help")
    def help_(*query: str) -> None:
        """Show command help or search shipped docs and Python docstrings without a workspace.

        Args:
            query: an exact command path such as `batch run`, or words such as `Log.read_table`.
        """
        Help(app).show(" ".join(query))

    # A trailing-command verb hands its command on verbatim from the first token that is not one
    # of its own options, `Delimiter` placing the `--` (see `delimiter.py` for why the command is
    # not `allow_leading_hyphen`). cyclopts honours the delimiter for its help flags but not for
    # its version flag, so these verbs give up `--version` (the root app still answers it) rather
    # than answer `run python --version` with this tool's version.
    @app.command(version_flags=[])
    def run(*command: str, on: str = "local", env: str = "", container: str = "") -> int:
        """Run a command, or a job spelled `path/to/file.py::name`, through the host's plan.

        Native file targets run locally with the same runner and closure format as submitted
        jobs; use submit for remote file targets, and run collection and help locally. Plain
        remote diagnostic commands execute over SSH, on a cluster's login endpoint rather than in
        a batch allocation. The exit code is the command's own.

        Args:
            command: the command tokens, from the first token that is not an option of this verb,
                passed on verbatim with its own flags; a job's arguments follow `--`.
            on: the host alias, `local` for this machine.
            env: an environment name overriding the profile's choice.
            container: a container override, `none` forcing bare.
        """
        return board(on).run(command, env=env, container=container)

    @job.command(version_flags=[])
    def submit(
        *command: str,
        on: str = "",
        resume: str = "",
        batch: str = "",
        split: str = "",
        per_job: int = 0,
        cell_timeout: float = 900.0,
        name: str = "",
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        gpu_name: str = "",
        max_usd: float = 0.0,
        attempt: int = 1,
        fetch: str = "",
        node: str = "",
        needs: tuple[str, ...] = (),
        env: str = "",
        container: str = "",
        estimate: bool = False,
        wait: bool = False,
        yes: bool = False,
        declared: Declared = _SPEC_ONLY,
        output: Output = _COMPACT,
    ) -> int:
        """Dispatch one job, a batch, or a test lane split over hosts, and print the handles.

        One job: a command, or `path/to/file.py::name` (which ships only the code it imports),
        on `--on HOST`. A batch: `--batch spec.toml`, or `--job target:command` repeated, each
        job to its own target; a target refusing is that job's row and the rest still go. A
        lane: `path/to/test.py::test --on a,b --split model` runs the test's cells as one job
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
            on: the host alias; comma-separated for a lane.
            queue: a queued host's queue, the profile's default when empty.
            walltime: the wall-clock limit, `HH:MM:SS`, the profile's default when empty.
            mem_gb: system memory to ask the scheduler for, the profile's default when 0.
            gpus: cards per job, the profile's default when 0.
            env: the environment the job enters, the host profile's own when empty.
            container: a container override, `none` forcing bare.
            resume: a failed or cancelled run to continue, by handle or name, as its next attempt.
            batch: a batch spec file, relative to the workspace root.
            split: a lane's parametrize name; each of its values becomes one job.
            per_job: a lane's cells per job when no name groups them, 0 for all in one.
            cell_timeout: seconds one lane cell may take before its process is killed.
            name: the run's label, or the batch's name for a `--job` batch.
            gpu_name: the GPU type to rent, for a metered provider host.
            max_usd: the spend cap a provider host refuses to submit without.
            attempt: the 1-based try number feeding expression defaults.
            fetch: a results path recorded for later collection, the node's evidence directory
                when unset and `--node` names one.
            node: the ledger slug this run serves, carried into its record and receipts.
            needs: a workspace-relative data path the job reads on the host, repeatable.
            estimate: price the plan and print it; dispatch nothing.
            wait: block until every submitted job settles; exit 0 clean, 1 failed, 2 timeout.
            yes: dispatch without asking, what a script passes.
        """
        if batch or declared.job:
            return _submit_batch(batch, declared, name=name, estimate=estimate, wait=wait)
        if resume:
            cache = board("local").dispatcher.cache
            previous = cache.run(resume, on or None)
            on = on or previous.target
            name = previous.label
            node = node or previous.node
            fetch = fetch or previous.fetch_path or ""
            attempt = cache.attempts(name, on) + 1
            target = joined(command) if command else vetted(previous.script)
        else:
            target = joined(command)
        if not on:
            raise MissionError("name a host: --on <alias>, or a batch with --batch")
        if split or per_job or "," in on:
            return _submit_lane(
                target,
                hosts=[alias.strip() for alias in on.split(",") if alias.strip()],
                split=split,
                per_job=per_job,
                cell_timeout=cell_timeout,
                resources={
                    "queue": queue,
                    "walltime": walltime,
                    "mem_gb": mem_gb,
                    "gpus": gpus,
                    "gpu_name": gpu_name,
                    "max_usd": max_usd,
                },
                node=node,
                estimate=estimate,
                wait=wait,
                yes=yes,
            )
        workspace = board(on)
        priced = workspace.expectation(
            target,
            queue=queue,
            walltime=walltime,
            mem_gb=mem_gb,
            gpus=gpus,
            gpu_name=gpu_name,
            max_usd=max_usd,
            attempt=attempt,
        )
        pulled = workspace.results(fetch, node=node, command=target)
        print(_expected(priced, results=pulled), file=sys.stderr)
        if estimate:
            return 0
        if not yes and sys.stdin.isatty() and not _agreed():
            raise SystemExit(1)
        with progress(f"submitting on {on}") as stage:
            submitted = workspace.submit(
                target,
                watch=stage,
                name=name,
                queue=queue,
                walltime=walltime,
                mem_gb=mem_gb,
                gpus=gpus,
                gpu_name=gpu_name,
                max_usd=max_usd,
                attempt=attempt,
                fetch=fetch or None,
                node=node,
                needs=needs,
                env=env,
                container=container,
            )
        if output.mode is None:
            print(submitted.handle.id)
        else:
            output.print_record(submitted.handle.model_dump(), title="handle")
        if not wait:
            return 0
        return show(submitted.handle.id, on=on, wait=True)

    def _submit_batch(
        spec: str, declared: Declared, *, name: str, estimate: bool, wait: bool
    ) -> int:
        """Price, or dispatch, every job a batch declares; print the batch id and each row."""
        batched = declared.batch(board("local"), spec, name=name)
        if estimate:
            with progress(f"pricing {batched.id}"):
                priced = [row.model_dump() for row in batched.estimate().jobs]
            _tabled(
                priced,
                _ESTIMATE_COLUMNS,
                summing=("wire_bytes", "runtime_s", "expected_usd", "p90_usd"),
                output=_COMPACT,
                title=f"estimate: {batched.id}",
            )
            return 0
        with progress(f"dispatching {batched.id}") as stage:
            dispatched = batched.run(watch=stage)
        print(f"# batch {batched.id}")
        _COMPACT.print_rows(
            [entry.model_dump() for entry in dispatched],
            title=f"run: {batched.id}",
            columns=_DISPATCH_COLUMNS,
        )
        return show(batched.id, wait=True) if wait else 0

    def _submit_lane(
        target: str,
        *,
        hosts: list[str],
        split: str,
        per_job: int,
        cell_timeout: float,
        resources: dict[str, str | int | float],
        node: str,
        estimate: bool,
        wait: bool,
        yes: bool,
    ) -> int:
        """Run a test lane's cells on each host, one job per group; `local` runs them in place.

        The lane's parametrization is the plan: its cells are collected here, grouped by the
        `split` value or sliced, and each group runs its cells as fresh processes.
        """
        manifest = load(project.manifest(workspace_root()))
        if windows := [
            h for h in hosts if h in manifest.hosts and manifest.hosts[h].platform == "win-64"
        ]:
            raise MissionError(f"a lane cannot be queued on Windows hosts: {', '.join(windows)}")
        with progress(f"collecting {target}"):
            probe = ["run", "--", "python", "-m", "mainboard.jobs.lanes", "collect", target]
            cells = lanes_module.parsed(localhost[project.package][probe]())
        if not cells:
            raise MissionError(f"{target} collected no cells")
        groups = lanes_module.grouped(cells, by=split, per_job=per_job)
        served = node or lanes_module.node_of(target)
        fresh = ["--fresh", "--timeout", str(cell_timeout)]
        pytest_args = ["-p", "no:randomly", "-q", "--no-header"]
        _COMPACT.print_rows(lanes_module.summary(hosts, groups), title="lane")
        if estimate:
            return 0
        if not yes and sys.stdin.isatty() and not _agreed():
            raise SystemExit(1)
        dispatched: list[tuple[str, str, str]] = []
        code = 0
        for host in hosts:
            for chosen in groups:
                line = [target, "--", *fresh, *chosen.ids, "--", *pytest_args]
                if host == "local":
                    ran = board("local").run(line)
                    code = code or ran
                    dispatched.append((host, chosen.name, f"local exit {ran}"))
                    continue
                with progress(f"submitting {chosen.name} on {host}") as stage:
                    submitted = board(host).submit(
                        joined(line),
                        watch=stage,
                        name=f"lane-{host}-{chosen.name}",
                        node=served,
                        **resources,  # type: ignore[arg-type]
                    )
                dispatched.append((host, chosen.name, submitted.handle.id))
        _COMPACT.print_rows(
            [{"host": h, "group": g, "handle": i} for h, g, i in dispatched], title="dispatched"
        )
        for host, _, identity in dispatched if wait else ():
            if not identity.startswith("local exit"):
                code = code or show(identity, on=host, wait=True, quiet=True)
        return code

    @job.command
    def show(
        target: str,
        *,
        on: str = "",
        run: str = "",
        wait: bool = False,
        timeout: float = vocabulary.WAIT_SECONDS,
        stall: float = STALL_SECONDS,
        quiet: Annotated[bool, Parameter(show=False)] = False,
        output: Output = _COMPACT,
    ) -> int:
        """Print a job's or a batch's settled outcome, read only from its receipts; exit with it.

        One row per trial: its outcome, its gate sweep and the ledger node it serves, never a
        scheduler's or a session's memory. Exit 0 when every row settled clean, 1 on a failure,
        2 while anything is in flight (or `--wait` timed out), 3 when the receipts prove
        nothing, 4 when `--wait` saw a job stall.

        `--wait` blocks until it settles, sweeping exactly as `job list` does (results pulled
        back, rentals released), with each test cell and a heartbeat on stderr.

        Args:
            target: a handle or name as `submit`/`list` print it, a batch id, a receipts stream id,
                directory or file.
            on: the host alias narrowing a handle recorded on several hosts.
            run: which run of a receipts store to score, its newest when unset.
            wait: block until the target settles.
            timeout: with `--wait`, give up after this many seconds (exit 2); 0 waits forever.
            stall: with `--wait`, seconds a running job may print nothing on an idle card before
                the wait stops and exits 4; 0 never calls a job stalled.
        """
        verdicts = board("local").verdicts()
        if not wait:
            with progress(f"reading {target}"):
                settled = verdicts.of(target, host=on, run=run)
            return _settled(settled, output)
        if not quiet:
            print(f"waiting on {target}", file=sys.stderr, flush=True)
        with diverted():
            settled = verdicts.wait(
                target,
                host=on,
                timeout=timeout,
                interval=vocabulary.POLL_SECONDS,
                stall=stall,
                say=_said,
            )
        return _settled(settled, output)

    @job.command(name="list")
    def jobs(
        *,
        limit: int = 20,
        project: str = "",
        batch: str = "",
        watch: float = 0.0,
        every: str = "",
        output: Output = _COMPACT,
    ) -> None:
        """Settle every job that ended, then list the live ones and the newest settled ones.

        Each host is asked once about every run it still owes (one `qstat`, `squeue` or `pueue
        status`), so a wave of thirty says which run and which queue, since when, and when the
        scheduler expects a start. A run that ended is settled on the way: its results pulled
        back, its verdict recorded, its rental released. Live runs are never cut by `--limit`;
        what the listing left out is said on stderr, beside what settled this pass.

        Args:
            limit: how many settled runs to show behind the live ones, newest first.
            project: only the runs dispatched from this project (its directory under `research/`
                or `packages/`, or `MB_PROJECT`).
            batch: one batch's jobs instead, as `submit --batch` printed its id.
            watch: repeat every this many seconds until interrupted.
            every: install the settling pass into this machine's service manager at this period
                (`20m`), so jobs settle with no session open; `0` removes it.
        """
        if every:
            settling = schedule(workspace_root(), every)
            print(f"{Project().name}: {settling.detail}")
            if settling.fix:
                print(f"{Project().name}: run `{settling.fix}`")
            return
        if batch:
            watcher = board("local").watch(batch)
            show_status = partial(_status, output=output)
            if not watch:
                with progress(f"sweeping {batch}"):
                    status = watcher.once()
                show_status(status)
                return
            _followed(watcher.follow(watch), f"sweeping {batch}", show_status)
            return

        def listed() -> None:
            workspace = board("local")
            with progress("settling and asking every host about its live jobs"):
                report = workspace.monitor().once()
                taken = Listing(workspace, limit=limit, project=project).taken()
            if output.json and not watch:
                # The periodic pass reads this document: the rows and what the sweep moved.
                sweep = {**report.model_dump(), "changed": report.changed}
                print(
                    dumps(
                        {"jobs": [row.model_dump() for row in taken.rows], "sweep": sweep},
                        default=str,
                    )
                )
                return
            for change in _changes(report):
                print(
                    "settled " + " ".join(value for value in change.values() if value),
                    file=sys.stderr,
                )
            output.print_rows(
                [row.model_dump() for row in taken.rows], title="jobs", columns=_JOB_COLUMNS
            )
            if taken.note:
                print(taken.note, file=sys.stderr)

        listed()
        with suppress(KeyboardInterrupt):
            while watch:
                time.sleep(watch)
                listed()

    @job.command(show=False)
    def monitor(*, json: bool = False) -> None:
        """The settling pass periodic runners installed before `job list --every` call.

        Args:
            json: print the listing and the sweep as JSON, what those runners read.
        """
        jobs(output=Output(json=json))

    @app.command
    def add(
        spec: str,
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "conda",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
        output: Output = _COMPACT,
    ) -> None:
        """Declare a dependency in the manifest and re-solve, showing what the lock did.

        A bare name is pinned to whatever the ecosystem's index publishes right now, and a name
        carrying its own constraint is written exactly as given. The table it lands in is the
        one the flags name, and where the manifest already writes that kind of requirement in a
        particular table, the edit joins it there.

        Args:
            lang: the ecosystem whose resolver installs it.
            env: an environment name, the workspace-wide table when omitted.
            dev: declare it as a development-only requirement.
            resolve: `--no-resolve` stages several edits to solve once.
        """
        with progress(f"adding {spec}"):
            changes = (
                board("local").deps().add(spec, ecosystem=lang, env=env, dev=dev, resolve=resolve)
            )
        _changed(changes, output, title="add")

    @app.command
    def remove(
        name: str,
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
        output: Output = _COMPACT,
    ) -> None:
        """Drop a dependency from the manifest and re-solve, showing what the lock did.

        With no flags the whole manifest is searched, so dropping a requirement never asks
        which table it was written into. `--lang`, `--env` and `--dev` narrow that search to one
        ecosystem's, one environment's or the development-only tables, which is also how a name
        declared in more than one table is told apart.

        Args:
            name: the dependency to drop.
            lang: only this ecosystem's tables (`python`, `conda`, `nodejs`, ...).
            env: only this environment's tables.
            dev: only the development-only tables.
            resolve: `--no-resolve` stages several edits to solve once.
        """
        with progress(f"removing {name}"):
            changes = (
                board("local")
                .deps()
                .remove(name, ecosystem=lang, env=env, dev=dev, resolve=resolve)
            )
        _changed(changes, output, title="remove")

    @app.command
    def upgrade(
        name: str = "",
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "",
        env: str = "",
        dev: bool = False,
        output: Output = _COMPACT,
    ) -> None:
        """Move one dependency to its newest release, or the whole lock forward in its bounds.

        Named, the constraint itself is rewritten to what the ecosystem publishes now, which is
        the only way past a ceiling the manifest declares. Unnamed, the manifest is untouched
        and the lock is re-solved against the indexes, moving every pin as far as the declared
        constraints already allow. `--lang`, `--env` and `--dev` narrow the search for the name
        to one ecosystem's, one environment's or the development-only tables.

        Args:
            name: the dependency to move to its newest release; the whole lock when omitted.
            lang: only this ecosystem's tables.
            env: only this environment's tables.
            dev: only the development-only tables.
        """
        with progress(f"upgrading {name or 'the lock'}"):
            changes = board("local").deps().upgrade(name, ecosystem=lang, env=env, dev=dev)
        _changed(changes, output, title="upgrade")

    @app.command
    def new(
        name: str,
        *,
        template: str = "",
        description: str = "",
        dest: str = "",
        answer: tuple[str, ...] = (),
        output: Output = _COMPACT,
    ) -> None:
        """Scaffold a project from one of this workspace's declared templates.

        Which templates exist is the manifest's `[templates]` table, and the first declared one
        is what this renders when none is named. Every answer a template asks for comes from
        the name, from what the workspace already declared, or from the template's own default,
        so a project stays one argument while `--answer` covers the rest. The task rows a
        template generates are printed rather than pasted, since the root manifest's task table
        is hand-curated and the same project has to reach the type checker's search path beside
        it, and half of that edit landing on its own is worse than none of it.

        Args:
            name: the project name, which becomes its slug, its package and its task prefix.
            template: a declared template name or any location copier accepts.
            description: the one sentence the README and the task rows carry.
            dest: where to render it, under the template's own declared home when omitted.
            answer: a further `question=value` for the template, repeatable.
        """
        with progress(f"rendering {name}"):
            made = (
                board("local")
                .scaffold()
                .render(
                    name,
                    template=template,
                    description=description,
                    dest=dest,
                    answers=_answers(answer),
                )
            )
        payload = made.model_dump()
        # The rows print whole and pasteable at a terminal, since wrapped inside a table cell
        # they are harder to copy back out. A compact mode keeps the field, the only place a
        # caller reading the record gets them.
        if output.mode is None and made.snippet:
            print(made.snippet)
            payload.pop("snippet")
        output.print_record(payload, title="new")

    @app.command
    def doctor(
        env: str = "", *, center: bool = False, members: bool = False, output: Output = _COMPACT
    ) -> int:
        """Say whether this workspace is fit to work in, and exit nonzero when it is not.

        Asked at once and bounded: is the manifest coherent, is what is installed the
        environment it describes, what compute answers, do onboarded hosts match the manifest,
        does a periodic pass settle jobs, does every declared gate hold. Each row names the one
        command that repairs it; a sleeping host or a provider with no key is a word, not a
        failure.

        Args:
            env: the environment to examine, the local profile's own when omitted.
            center: also judge this machine as the workspace's center: git tooling (lfs, symlinks,
                long paths), agent configuration, the default environment on every shell's PATH, a
                torch and CUDA smoke run, scripts that behave differently here.
            members: also check every `[workspace] members` project works cloned alone: no import,
                path or task that only the monorepo satisfies, and a clean `uv` install.
        """
        with progress("examining the workspace"):
            local = board("local")
            sections = Verification(local).sections() if center else local.doctor(env).sections()
        code = _sectioned(sections, output, title="doctor")
        if members:
            standalone = Standalone(composition(project.manifest(workspace_root())))
            with progress("checking the members alone"):
                alone = standalone.sections(())
            code |= _sectioned(alone, output, title="members", heading=True)
        return code

    @app.command
    def install(env: str = "", *, profile: str = "") -> None:
        """Compile the manifest and install the environment on this machine.

        Another machine is onboarded with `setup`, which ends by running this verb there.

        Args:
            env: the environment name, this machine's declared profile choice when omitted.
            profile: the declared host profile describing this machine, so the environment's
                activation carries that host's modules; what `setup` passes when a host installs.
        """
        with progress(f"installing {env or 'the environment'}") as stage:
            board("local").install(env, profile=profile, watch=stage)

    @app.command
    def lock(env: str = "", *, profile: str = "") -> None:
        """Solve the manifest into the committed lock, then install what it pinned here.

        `install` never solves: it installs exactly the lock, so a host installs what this
        machine solved. This is the one verb that moves the lock. With no environment named,
        every declared one is solved, as `pixi lock` does: a shared input (a workspace
        package's pyproject.toml) moves them all at once, and a lock left behind fails on the
        host that installs it.

        Args:
            env: the environment to solve, every declared one when omitted.
            profile: the declared host profile describing this machine.
        """
        names = [env] if env else ["default", *load(project.manifest(workspace_root())).envs]
        for name in names:
            with progress(f"solving {name}") as stage:
                board("local").install(name, resolve=True, profile=profile, watch=stage)

    @self_.command
    def update() -> None:
        """Reinstall this tool from its source when the source moved, as `uv self update` does.

        Nothing checks or updates on its own: every other verb runs the code installed.
        """
        found = staleness.check()
        if found.source is None or not found.stale:
            print(f"{project.name}: {found.detail}")
            return
        print(f"{project.name}: {found.detail}; {staleness.update(found)}")

    @self_.command
    def version(*, output: Output = _COMPACT) -> None:
        """Show this installation: its version, where it runs from, and the engines it pins.

        DuckDB is named because a served lake needs the same release at both ends.
        """
        import duckdb  # noqa: PLC0415  (only this verb and the lake need it)

        output.print_record(
            {
                "version": metadata.version(project.package),
                "python": sys.executable,
                "source": str(staleness.check().source or ""),
                "duckdb": duckdb.__version__,
            },
            title="self version",
        )

    @app.command
    def completion(shell: Literal["bash", "zsh", "fish", "powershell"]) -> None:
        """Print the script that completes this tool's commands and options in a shell.

        Keep it in the shell's startup: `eval "$(mb completion zsh)"` (zsh, bash; `mb
        completion fish | source` in fish) or `mb completion powershell | Out-String |
        Invoke-Expression` in `$PROFILE`. The same four shells on Linux, macOS and Windows.

        Args:
            shell: the shell the script is for.
        """
        if shell == "powershell":
            print(powershell(app, (project.name, project.package)))
            return
        print(app.generate_completion(prog_name=project.name, shell=shell))

    @host.command(name="upgrade")
    def host_upgrade(on: str = "local", *, dry_run: bool = False) -> int:
        """Bring this machine or a host up to date in one pass, printing each step as it runs.

        The system's managers first (apt: update, full-upgrade, autoremove, autoclean; dnf, brew,
        snap, winget), then the pixi global toolbox, the dotfiles and this tool. Firmware is only
        audited. A host runs its own copy over a terminal, so `sudo` can ask for its password.

        Args:
            on: the host alias, `local` for this machine.
            dry_run: print the steps without running them.
        """
        if on != "local":
            flags = ("--dry-run",) if dry_run else ()
            board(on).interact(project.name, "host", "upgrade", *flags)
        return upkeep.upgrade(dry_run=dry_run)

    @host.command
    def unlock(*hosts: str) -> None:
        """Unlock each host's ssh key once, so every later connection this tool opens is silent.

        Keeps one ssh-agent on a fixed socket beside the ssh config, started on demand and
        outliving this process; every command this tool runs afterwards uses it. On Windows no ssh
        can share a connection, so this is what makes a passphrase-protected host (miyabi-g)
        reachable without a prompt per connection. Asks each key's passphrase once per boot.

        Args:
            hosts: the ssh aliases whose keys to add.
        """
        for host in hosts:
            if status := keys.unlock(host):
                raise MissionError(f"{host} still refuses a silent login (exit {status})")
            print(f"{host}: reachable without a prompt")

    @app.command(name="pack")
    def pack_(
        env: str = "default",
        *,
        on: str = "local",
        image: bool = False,
        sif: bool = False,
        push: str = "",
        output: Output = _COMPACT,
    ) -> None:
        """Build an environment into files a machine runs without installing anything.

        Always a self-extracting executable (pixi-pack: `./file` unpacks `env/` and
        `activate.sh`, no pixi or network needed); `--image` adds an OCI image (slim Debian plus
        the environment, CUDA from its own wheels, the driver from `--gpus all`), `--sif` an
        Apptainer file for HPC, `--push` the image to a registry. Each is named by the
        environment's digest, so an unchanged lock is never built twice. Built on the machine
        `--on` names, whose package mirrors are close, never uploaded from here. The workspace's
        own code is not inside: a dispatch ships it, as always.

        Args:
            env: the environment, installed on that machine already (`host setup --env`).
            on: the host to build on, `local` for this machine (Linux or macOS).
            image: also build an OCI image (docker).
            sif: also build an Apptainer SIF from the image.
            push: also push the image to this repository, `ghcr.io/<owner>/<name>`.
        """
        with progress(f"packing {env} on {on}"):
            packed = board(on).pack(env, image=image, sif=sif, push=push)
        output.print_record(packed.model_dump(), title="pack")

    @app.command(name="list", version_flags=[])
    def list_(*command: str, env: str = "default") -> NoReturn:
        """List the packages installed in an environment, through `pixi list`.

        Args:
            command: pixi's own arguments, a package regex first (`mb list torch --json`).
            env: the environment name.
        """
        pixi_verb("list", env, command)

    @app.command(version_flags=[])
    def tree(*command: str, env: str = "default") -> NoReturn:
        """Show an environment's dependency tree, through `pixi tree`.

        Args:
            command: pixi's own arguments, a package regex first (`mb tree numpy --invert`).
            env: the environment name.
        """
        pixi_verb("tree", env, command)

    def installed(env: str) -> Pixi:
        """The pixi holding `env` here, refused when `env` was never installed on this machine."""
        root = workspace_root()
        pixi = Provisioner(root, load(project.manifest(root))).pixi_for(env)
        if not pixi.ready(env):
            raise MissionError(
                f"environment {env!r} is not installed here; run `mb install {env}`"
            )
        return pixi

    def pixi_verb(verb: str, env: str, arguments: Sequence[str]) -> NoReturn:
        """Hand this process to pixi's `verb` over the installed `env`, frozen to its lock."""
        pixi = installed(env)
        argv = [str(pixi.executable), verb, "--frozen", "-e", env, *pixi.scope(), *arguments]
        become(argv[0], argv, env={**os.environ, **pixi.overrides})

    @app.command(version_flags=[])
    def shell(
        *command: str,
        on: str = "local",
        env: str = "",
        queue: str = "",
        walltime: str = "",
        keep: bool = False,
    ) -> NoReturn:
        """Open an interactive shell in this workspace's environment, here or on a host.

        The daily way in, and the one verb that works from a terminal where nothing is
        activated yet. This process becomes the shell, so quitting it returns to the terminal
        that asked. On a host the shell opens inside its mirrored workspace, and a queued host
        is asked for an interactive allocation first, so the terminal lands on a compute node
        rather than on the login node the request was made from.

        Args:
            command: on a host, a command to run instead of handing over the terminal, from the
                first token that is not an option of this verb.
            on: the host alias, `local` for this machine.
            env: the environment name, the profile's declared choice when omitted.
            queue: on a queued host, the queue the allocation targets, the profile's when omitted.
            walltime: on a queued host, the session's wall-clock limit, the profile's when omitted.
            keep: on a host, hold the session in tmux on the far side so a dropped terminal leaves
                the allocation up, and reattach to one already held.
        """
        if on != "local":
            board(on).interact(*command, env=env, queue=queue, walltime=walltime, keep=keep)
        elif command or queue or walltime or keep:
            raise MissionError(
                "a command, a queue, a walltime and --keep belong to a host's shell; run a "
                f"command here with `{project.name} run -- <command>`"
            )
        else:
            board("local").shell(env)

    @host.command
    def setup(
        host: str,
        *,
        env: str = "",
        resolve: bool = False,
        dotfiles: bool = False,
        center: bool = False,
        root: str = "",
        output: Output = _COMPACT,
    ) -> int:
        """Onboard a host until it can run jobs, then show what it became and what that means.

        The host installs the environment its declared profile names from the lock this
        workspace solved, shipped with the mirror. Then the findings `host list --facts` shows,
        judged from the census read back through the new activation.

        `--center` instead moves the center there (Windows, macOS or Linux): probes it, signs gh
        in, carries ssh config and keys, clones the monorepo at this HEAD with every owned
        submodule, carries what git does not hold (`.env`, the lake, agent configuration and
        memory), installs this tool and the default environment, and ends with `doctor --center`
        run there. Secrets ride ssh's stdin only. Running it again continues or re-verifies.

        Args:
            env: an environment name overriding the host profile's own.
            resolve: let the host solve for itself instead of installing the shipped lock.
            dotfiles: also apply the workspace's dotfiles (zsh, lvim, the pixi toolbox, herdr);
                every later sync keeps them. Without it a host gets only what jobs need: this
                tool, pixi, pueue and the environment.
            center: make this host the workspace's center instead of a job host.
            root: with `--center`, where the workspace goes there, `~/projects` when omitted.
        """
        if center:
            with progress(f"moving the center to {host}") as stage:
                sections = Migration(board("local"), host, root=root, watch=stage).run()
            return _sectioned(sections, output, title="center")
        workspace = board(host)
        with progress(f"setting up {host}") as stage:
            report = workspace.install(env, resolve=resolve, watch=stage, dotfiles=dotfiles)
        _onboarded(workspace, report, output, title="setup")
        return 0

    @host.command
    def sync(host: str, *, env: str = "", output: Output = _COMPACT) -> None:
        """Re-mirror a host already set up and re-provision it from the shipped lock.

        The fast path back after source moved: the tool is not reinstalled and the hardware is
        not probed again, so a Python edit reaches the host in the time the mirror takes. A
        host never set up needs `setup` first, which is where the probe and the tool come from.

        Args:
            env: an environment name overriding the host profile's own.
        """
        workspace = board(host)
        with progress(f"syncing {host}") as stage:
            report = workspace.install(env, resolve=False, watch=stage, sync_only=True)
        _onboarded(workspace, report, output, title="sync")

    @host.command
    def hold(
        provider: str,
        *,
        for_: Annotated[str, Parameter(name="--for")],
        as_: Annotated[str, Parameter(name="--as")] = "",
        gpu_name: str = "",
        gpus: int = 0,
        max_usd: float = 0.0,
        env: str = "",
        output: Output = _COMPACT,
    ) -> None:
        """Rent a machine and keep it as an ssh host until a deadline, set up and ready for jobs.

        A rental per job rebuilds the environment every time; a held machine is set up once and
        then takes `submit --on <alias>` in seconds. The machine gets an alias in the ssh config,
        onboards like `setup`, and is recorded with its deadline, which `monitor` and `compute`
        both enforce by releasing it, so a forgotten hold stops billing on time.

        Args:
            provider: the provider host to rent through, `vast` say.
            for_: how long to keep it once it is ready, `3h`, `90m` or `1h30m`.
            as_: the alias to reach it by, `<provider>-<card>` when omitted.
            gpu_name: the card to rent, in the provider's own spelling.
            gpus: cards per machine, the provider profile's default when 0.
            max_usd: the spend cap over the whole hold, landing included, the provider's default
                when 0.
            env: the environment to set up, the provider profile's own when omitted.
        """
        with progress(f"holding a {gpu_name or provider} machine") as stage:
            held = Holds(board("local")).hold(
                provider,
                duration=for_,
                alias=as_,
                gpu_name=gpu_name,
                gpus=gpus,
                max_usd=max_usd,
                env=env,
                watch=stage,
            )
        _held(held, output, title="hold")

    @host.command
    def release(alias: str, *, output: Output = _COMPACT) -> None:
        """End a held machine now: stop its billing, settle its record and drop its alias.

        Args:
            alias: the held machine's alias, as `hold` printed it.
        """
        with progress(f"releasing {alias}"):
            held = Holds(board("local")).release(alias)
        _held(held, output, title="release")

    @host.command(name="list")
    def compute(
        *hosts: str,
        facts: bool = False,
        gpus: bool = False,
        audit: bool = False,
        plan: bool = False,
        env: str = "",
        output: Output = _COMPACT,
    ) -> int:
        """List every compute path this workspace can reach; flags add detail per host.

        One row per path: this machine, each declared host (answering? set up?), each provider
        (credentials here? credit left? a live rate where cheap). Held machines past their
        deadline are released first. Probes run in parallel, so the fleet answers in the time
        the slowest host takes; a host that is down is a row, not a failure. No credential is
        ever printed. Each flag prints one more table keyed by host, for the hosts named or,
        when none is named, this machine (`--gpus` reads every ssh host).

        Args:
            hosts: narrow the listing to these aliases, `local` for this machine.
            facts: each host's probed hardware and software, then what they mean for this
                workspace (platform, lock, CUDA floor, card memory), each finding with its fix.
            gpus: who holds each card now: utilization, memory and the processes on it.
            audit: what each host could and should update, read-only; `host upgrade` applies it.
                Exits 1 when an audit row fails.
            plan: the execution plan each host resolves to: root, environment, container,
                scheduler, modules, exports.
            env: with `--plan`, an environment overriding the host profile's own.
        """
        workspace = board("local")
        for released in Holds(workspace).expire():
            gone = f"{released.provider} {released.handle}"
            print(f"released {released.alias}, {gone}", file=sys.stderr)
        if not (facts or gpus or audit or plan):
            with progress("probing every compute path"):
                paths = workspace.compute().paths()
            chosen = [path for path in paths if not hosts or path.name in hosts]
            output.print_rows([path.model_dump() for path in chosen], title="compute")
            return 0
        named = list(hosts) or ["local"]
        failed = False
        for alias in named if plan else ():
            resolved = Resolver(load(project.manifest(workspace_root()))).plan(alias, env=env)
            _heading(output, f"plan: {alias}")
            output.print_record(resolved.model_dump(), title=f"plan: {alias}")
        for alias in named if facts else ():
            _facts(board(alias), alias, output)
        if gpus:
            _gpus(board, named if hosts else _ssh_hosts(), output)
        for alias in named if audit else ():
            with progress(f"auditing {alias}"):
                sections = board(alias).audit()
            failed |= bool(_sectioned(sections, output, title=f"audit: {alias}", heading=True))
        return 1 if failed else 0

    def _ssh_hosts() -> list[str]:
        manifest = load(project.manifest(workspace_root()))
        return ["local", *(alias for alias, spec in manifest.hosts.items() if spec.kind == "ssh")]

    @job.command
    def collect(path: str, *, on: str, json: bool = False) -> None:
        """Collect remote evidence for queries, including runs started directly on that node.

        Complete files are immutable. Conflicts preserve the local copy and fail collection.
        Live event snapshots exclude incomplete records; queries deduplicate overlapping frames.
        A new query sees published files. Collection is not one transaction across servers.

        Args:
            path: workspace-relative results file or directory, using forward slashes on every OS.
            on: declared SSH host; root and bootstrap Python come from its profile, a `~` root
                expanded by that Python when no setup has placed it yet. Python is a command in
                that host's SSH login shell, usually python3; quote an absolute interpreter path as
                that shell requires. No remote Mainboard is needed.
            json: print a machine-readable collection summary.
        """
        workspace = board(on)
        root = workspace.plan(container="none").profile.root
        published = workspace.dispatcher.fetch_path(on, root=root, path=path)
        Output(json=json).print_record(
            {"host": on, "path": path, "new_files": published}, title="collection"
        )

    @app.command
    def query(
        sql: str | None = None,
        *,
        file: Path | None = None,
        project: str = "",
        json: bool = False,
        out: Path | None = None,
    ) -> None:
        """Explore collected results across servers; each query sees newly arrived files.

        Views: runs, trials, events, metrics, artifacts, jobs, and every table of the
        workspace's state lake as `lake.<table>` (lake.events, lake.log_lines, lake.costs...).
        Project scopes science views;
        jobs always shows the fleet. Monitor refreshes tracked jobs; collect also imports
        results from native remote runs. Neither requires a shared database service.

        Args:
            sql: one DuckDB SELECT statement; defaults to SELECT * FROM runs without --file.
            file: read SQL from a UTF-8 file instead of the positional statement.
                Both the file and paths inside SQL resolve from the caller's current directory.
            project: restrict scientific rows to this research project.
            json: print JSON when no output file is requested.
            out: export to a new .csv, .parquet, or .json file instead of printing rows.
        """
        source = _query_source(sql, file)
        if source is None:
            source = "SELECT * FROM runs"
        from .results import (
            Results,  # its dataframe engine is paid only by the verbs reading results
        )

        results = Results(workspace_root())
        if out is not None:
            print(results.export(source, out, project=project))
            return
        rows = loads(dumps(results.rows(source, project=project), default=str))
        Output(json=json).print_rows(rows, title="results")

    @paper.command
    def plot(
        sql: str | None = None,
        *,
        file: Path | None = None,
        config: Path | None = None,
        figure: str = "",
        x: str = "",
        y: str = "",
        out: tuple[Path, ...] = (),
        project: str = "",
        hue: str = "",
        kind: Literal["scatter", "line", "bar"] = "scatter",
        style: str = "",
        dpi: int | None = None,
        title: str = "",
    ) -> None:
        """Plot a local SELECT using Seaborn and Matplotlib, without implicit aggregation.

        Args:
            sql: define the table, including any filtering, grouping, and ordering.
            file: read a UTF-8 SQL file instead of the positional statement.
                Both the file and paths inside SQL resolve from the caller's current directory.
            config: overlay project styles and figures from this manifest-format TOML file.
                It does not select an environment or change path resolution.
            figure: render a named [figures.<name>] specification; omit SQL, x, and y.
                Panels use native Seaborn marks and Matplotlib settings, without estimation.
            x: column names; omit hue for one series.
            y: column names; omit hue for one series.
            hue: column names; omit hue for one series.
            out: a new output path; repeat for multiple formats, such as .pdf and .png.
            project: restrict scientific rows to this research project.
            kind: bar requires one row per x/hue group.
            style: a named [plots.<name>] entry; defaults to paper when declared.
            dpi: raster resolution, overriding the style's DPI when supplied.
            title: the chart title, including the measurement scope when appropriate.
        """
        source = _query_source(sql, file)
        if figure and (source is not None or x or y or hue or title):
            raise MissionError("--figure cannot be combined with SQL or individual chart mappings")
        if not figure and source is None:
            raise MissionError("plot requires a SQL statement or --file")
        if not figure and (not x or not y):
            raise MissionError("plot requires --x and --y without --figure")
        # Plotting is a genuine optional dependency boundary; other verbs do not import it.
        try:
            from .plots.figure import FigurePlot
            from .plots.table import Plot
        except ModuleNotFoundError as fault:
            if fault.name not in {"seaborn", "matplotlib", "pandas"}:
                raise
            raise MissionError(
                "plotting requires the plot extra. From the monorepo root run: "
                "uv tool install --reinstall --python 3.14 "
                "--from './packages/mainboard[plot]' mainboard --force"
            ) from fault
        settings = PlotStyle()
        specification = None
        manifest = load_plot_config(Project().manifest(workspace_root()), config)
        if figure:
            try:
                specification = manifest.figures[figure]
            except KeyError:
                raise MissionError(
                    f"no figure {figure!r}; declared figures are {sorted(manifest.figures)}"
                ) from None
            style = style or specification.style
        style = style or ("paper" if "paper" in manifest.plots else "")
        if style:
            try:
                settings = manifest.plots[style]
            except KeyError:
                raise MissionError(
                    f"no plot style {style!r}; declared styles are {sorted(manifest.plots)}"
                ) from None
        from .results import (
            Results,  # its dataframe engine is paid only by the verbs reading results
        )

        results = Results(workspace_root())
        if specification is not None:
            saved = FigurePlot(settings).render(
                specification, partial(results.query, project=project), *out, dpi=dpi
            )
        else:
            assert source is not None
            saved = Plot(results.query(source, project=project), settings).save(
                *out, x=x, y=y, hue=hue, kind=kind, dpi=dpi, title=title
            )
        for path in saved:
            print(path)

    # The cutok artifact's README runs `mainboard plot`; kept until that artifact is frozen.
    app.command(plot, name="plot", show=False)

    @app.command
    def lint(*paths: Path, check: bool = False, only: str = "", json: bool = False) -> int:
        """Fix what can be fixed, then check, over the changed files or everything under PATHS.

        With no path the pass reads every file that differs from HEAD or is new, submodules
        entered and deletions included, so the everyday call costs what the edit did. A path
        widens it to every file git tracks or would track at or beneath it, so `lint .` at the
        root reads the whole workspace. The exit is nonzero when a file was rewritten or a step
        failed, the one answer a person, an agent, a hook and a CI job all act on.

        Args:
            paths: files or directories, relative to the working directory.
            check: write nothing: run each tool's read-only `check` command and report what the
                text hygiene would repair, the mode for CI and for verifying a tree.
            only: the steps to run, comma-separated tool names with `text` for the built-in
                hygiene; every step when empty, so `--only ruff-format` is a formatter alone.
            json: print the report as canonical JSON instead of the findings and a summary line.
        """
        root = workspace_root()
        inventory = Inventory(root)
        files = (
            inventory.under([path.resolve() for path in paths]) if paths else inventory.changed()
        )
        steps = [step.strip() for step in only.split(",") if step.strip()]
        report = Linter(root, load(project.manifest(root)), check=check, only=steps).lint(files)
        if json:
            record(report.model_dump(mode="json"), mode="json", fields=(), title="lint")
        else:
            if report.failures:
                print(report.findings())
            print(report.summary())
        return 0 if report.clean else 1

    ci = App(name="ci", help="Run a package's CI gate, the very steps its GitHub workflow runs.")
    app.command(ci)

    @ci.default
    def ci_run(
        package: Path | None = None, *, matrix: bool = False, output: Output = _COMPACT
    ) -> int:
        """Run the gate `[tool.mainboard.ci]` declares, exactly as the package's CI job runs it.

        The package is the nearest directory at or above PACKAGE whose pyproject.toml declares a
        gate, and needs no workspace. Its steps run in order from the package directory and stop
        at the first failure; each step's output goes to stderr as it settles, the table to
        stdout. With `--matrix` the gate also runs, at the same time, on every `[ci] hosts` entry
        of a supported platform this machine is not, the working tree shipped there as it
        stands, and only failing steps print their output. Exits 1 when any step failed.

        Args:
            package: a directory inside the package, the working directory when omitted.
            matrix: also run on one declared host per other platform, the check before a push.
        """
        found = Package.found((package or Path.cwd()).resolve())
        if matrix:
            planned = Matrix.planned(found, board("local"))
            with progress(f"running the gate on {len(planned.legs)} legs"):
                results = planned.run()
            for result in results:
                if result.failed:
                    _told(result)
            for family in planned.uncovered:
                _said(f"no leg ran on {family}; declare a host of it in [ci] hosts")
        else:
            leg = LocalLeg(found.root)
            results = []
            for result in leg.run(found.definition.on(leg.family)):
                _told(result)
                results.append(result)
        output.print_rows([result.row() for result in results], title="ci", columns=_CI_COLUMNS)
        return 1 if any(result.failed for result in results) else 0

    @app.command(show=False)
    def provide(env: str = "", *, source: str = "", expect: str = "", json: bool = False) -> None:
        """Build the immutable environment a dispatched job activates, and print where it is.

        The verb a host runs for itself, and the one a dispatch runs on it after pinning a
        source tree. An environment is addressed by the content of the compiled manifest and
        the lock beside it, so a directory built for one lock is never written to again and a
        wave queued against it keeps it however often the workspace re-solves meanwhile.

        Building one that already exists touches nothing and prints the same path, which is
        what lets every job of a wave ask and one of them build.

        Args:
            env: the environment to build, the host profile's own when empty.
            source: the directory holding the compiled artifact, workspace-relative or absolute;
                this workspace's own generated environment when empty.
            expect: the digest a dispatch pinned, refused when this machine reads the artifact as a
                different environment rather than building one no queued job will activate.
            json: print the path as canonical JSON instead of a bare line.
        """
        with progress(f"building the environment for {env or 'default'}"):
            built = board("local").provide(env, source, expect)
        if not json:
            print(built)
            return
        record({"prefix": str(built)}, mode="json", fields=(), title="prefix")

    @app.command(name="execute", show=False)
    def job_(record: str) -> int:
        """Run a dispatched job from its record, which every generated job script hands over.

        The job's command, the tree it runs from, the environment it enters, what it exports and
        how long it may take were all decided where it was dispatched and written down as one
        record. This carries the record out the same way on every host: build and enter the
        environment, run the command under its walltime, frame its receipts back and answer its
        exit status.

        Args:
            record: the job record as JSON, or the path of the job script that carries one.
        """
        return Runner(Job.read(record)).run()

    @app.command(show=False)
    def attest(stream: str, *, job: str = "") -> None:
        """Record what this machine looks like right now into a stream's receipts, once.

        The reading a measurement needs in order to say what it was taken under. Two jobs on one
        host run at the same time, so a benchmark can be measuring while another job holds the
        GPU, and nothing about the resulting artifact says so. This publishes one `job.attested`
        receipt carrying the machine's readings and whether the accelerator was idle, which is
        what lets `verdict` flag a row rather than forbid the run.

        A dispatched job runs this for itself before its command starts, so the verb is here for
        a measurement somebody takes by hand and for the job scripts that already call it.

        Args:
            stream: the receipts stream, a batch id or a run's name.
            job: the job inside that stream, the stream itself when omitted.
        """
        board("local").attest(stream, job=job or stream)

    @app.command(show=False)
    def sample(
        stream: str,
        *,
        job: str = "",
        interval: float = 0.0,
        seconds: float = 0.0,
        parent: int = 0,
    ) -> None:
        """Publish this machine's live readings into a stream's receipts until told to stop.

        GPU memory and busyness, host memory, and the enforced cgroup cap that memory is really
        running under, which is the number an OOM kill fires against and the one a hosted
        dashboard never had. Every reading is a `job.sample` receipt, so it lands in the
        workspace lake beside the rest of the stream.

        A dispatched job starts this for itself, so this verb is here for a command somebody
        runs by hand and for the job scripts that already call it.

        Args:
            stream: the receipts stream, a batch id or a run's name.
            job: the job inside that stream, the stream itself when omitted.
            interval: seconds between readings, the manifest's own when 0.
            seconds: stop after this long, 0 to run until interrupted.
            parent: stop when this process does, 0 for none.
        """
        sampler = board("local").samples(
            stream, job=job or stream, interval=interval, seconds=seconds, parent=parent
        )
        with suppress(KeyboardInterrupt), sampler:
            sampler.thread.join()

    @job.command
    def logs(handle: str, *, on: str = "") -> int:
        """Print only what a dispatched job printed, whether or not its host still exists.

        Only the exit code used to survive a run: the output lived on the host or on a rented
        disk that dies with the rental, so a lost terminal lost everything the job said. The
        durable sweep now keeps each settled run's tail beside that run's receipts, and this
        reads that copy first, falling back to the backend for a run still in flight. The frame a
        rented run carries its receipts home in is this tool's and not the job's, so it is left
        out; `verdict` reads the receipts themselves.

        An empty log has two entirely different causes, so a run that printed nothing answers
        with where it stands instead: its verdict, the scheduler's own state word, how long it
        has been waiting, and what the backend says about when it will run, which on PBS is the
        estimated start time the server reports. That line is the difference between a job that
        is queued behind a full cluster and one that started and said nothing.

        Exits 0 having printed output, 2 for a run still in flight that has printed none, and 1
        when nothing was captured and nothing is coming, so a script can tell the three apart.

        Args:
            handle: the job to read, as `submit` printed it or by the name `jobs` prints.
            on: the host alias narrowing a handle recorded on several hosts.
        """
        workspace = board("local")
        captured = printed(workspace.verdicts().captured(handle, host=on))
        if not captured.strip():
            return _unprinted(workspace, handle, host=on)
        # A job that coloured its output for a terminal it never had leaves escape codes in
        # the capture; a pipe or a file gets the plain text, a terminal gets the colours.
        shown = captured if sys.stdout.isatty() else plain(captured)
        print(shown, end="" if shown.endswith("\n") else "\n")
        return 0

    @job.command
    def cancel(handle: str, *, on: str = "", output: Output = _COMPACT) -> int:
        """Stop a dispatched job on whatever took it and settle its record in the same pass.

        The verb a provably doomed run needs. Without it a job could only die at its own
        walltime, and killing it over ssh by hand would stop the job while leaving the dispatch
        record claiming it still ran, so a cancellation lost its receipt trail. This kills
        through the backend the run was dispatched under, whether that is pueue, PBS or a
        provider API, writes the terminal verdict, publishes the settled receipt, and ends the
        rental, which is the only thing that stops a provider charging. A run on a host the
        manifest no longer declares, a released rental say, has nothing left to ask, so it
        settles cancelled with that cause and its evidence marked unverified.

        Exits the settled code, so a cancelled run exits 1: the stop was deliberate, and a
        completion check must still never call a stopped run complete.

        Args:
            handle: the job to cancel, as `submit` printed it or by the name `jobs` prints.
            on: the host alias narrowing a handle recorded on several hosts.
        """
        with progress(f"cancelling {handle}"):
            settled = board("local").verdicts().cancel(handle, host=on)
        return _settled(settled, output)

    proc = App(
        name="proc",
        help="Kill a process tree, bound a command, wait for a file or port, on every system.",
    )
    app.command(proc)

    @proc.command(name="list")
    def proc_list(pattern: str = "", *, on: str = "local", output: Output = _COMPACT) -> None:
        """List processes, started by this tool or not: pid, parent, user, cpu, memory, age.

        Args:
            pattern: a substring of the command line (`train.py`, `vllm`); your own processes
                when empty, everyone's when given.
            on: the host alias, `local` for this machine; a host answers with its own copy.
        """
        if on != "local":
            board(on).interact(project.name, "proc", "list", pattern)
        output.print_rows(Processes().matching(pattern), title="proc")

    @proc.command(name="kill")
    def proc_kill(
        pids: list[int] | None = None,
        *,
        match: str = "",
        on: str = "local",
        force: bool = False,
    ) -> int:
        """Stop each process and everything it started, children first, on any system.

        What `pkill -P`, `kill -- -pgid` and `taskkill /T` each do on one system, for any
        process, a job somebody started by hand included. Exits 1 when a process was gone.

        Args:
            pids: the processes to stop.
            match: also stop every process whose command line contains this (see `proc list`).
            on: the host alias, `local` for this machine; a host runs its own copy.
            force: kill at once instead of asking each process to terminate first.
        """
        if on != "local":
            flags = [*(["--match", match] if match else []), *(["--force"] if force else [])]
            board(on).interact(project.name, "proc", "kill", *map(str, pids or ()), *flags)
        chosen = (
            [*(pids or ()), *(int(row["pid"]) for row in Processes().matching(match))]
            if match
            else list(pids or ())
        )
        if not chosen:
            raise MissionError("name a process: a pid, or --match <text> (see `proc list`)")
        gone = Processes().kill(chosen, force=force)
        for pid in gone:
            print(f"no process {pid}", file=sys.stderr)
        return 1 if gone else 0

    @proc.command(name="timeout", version_flags=[])
    def proc_timeout(seconds: float, *command: str) -> int:
        """Run a command with a hard limit, stopping its whole tree when the limit passes.

        The portable `timeout`: the command runs with this terminal's stdio and without a shell,
        and exits with its own status, or 124 when it had to be stopped, as GNU `timeout` does.
        A tree that ignores the request to stop is killed after a short grace.

        Args:
            command: the program and its arguments, from the first token after the limit.
        """
        return Processes().timeout(seconds, command)

    @proc.command(name="wait")
    def proc_wait(
        *, file: Path | None = None, port: str = "", pid: int = 0, timeout: float = 0.0
    ) -> int:
        """Block until a file exists, a port accepts, and a process has exited, whichever named.

        The loop around `sleep`, `test` and `nc` that no Windows shell runs. Exits 0 once every
        named condition holds and 1 when the timeout passed first.

        Args:
            port: a `host:port` that must accept a TCP connection.
            pid: a process that must have exited.
            timeout: seconds to wait at most, 0 for as long as it takes.
        """
        return 0 if Processes().wait(file=file, port=port, pid=pid, seconds=timeout) else 1

    @lake.command(name="check")
    def lake_check(*, output: Output = _COMPACT) -> int:
        """Compare what the lake's catalog references with what is on disk; exit 1 on a loss.

        A deleted data file breaks only its table and `count(*)` hides it, so this is what finds
        one; a catalog WAL lost after a crash is named too.

        Args:
        """
        health = Lake.at(workspace_root()).ready().check()
        output.print_rows(
            [finding.model_dump() for finding in health.findings],
            title="lake check",
            columns=("table", "kind", "detail"),
        )
        return 0 if health.ok else 1

    @lake.command
    def compact() -> int:
        """Compact the lake: inlined rows to Parquet, small files merged, old snapshots expired.

        Files past their retention are deleted too. Appends keep committing meanwhile; exits 1
        when another process is already compacting.
        """
        if Lake.at(workspace_root()).ready().maintain():
            print("compacted")
            return 0
        print("another process is compacting this lake", file=sys.stderr)
        return 1

    @lake.command(name="upgrade")
    def lake_upgrade() -> None:
        """Migrate the lake's catalog to the newest DuckLake spec this DuckDB writes.

        One way: run it once every machine reading the lake runs a release that reads that spec.
        """
        print(Lake.at(workspace_root()).upgrade())

    @lake.command
    def serve(*, port: int = PORT, token: str = "") -> None:
        """Serve this workspace's lake over DuckDB's Quack protocol until interrupted.

        Listens on localhost only; another machine reaches it through an ssh tunnel (`ssh -R
        <port>:localhost:<port> <host>` from here, or `-L` from there) and attaches it by setting
        `MB_LAKE=quack:localhost:<port>` and `MB_LAKE_TOKEN`. Every query, listing and append
        there then reads and writes this lake, while commands here keep using its files. Both
        ends must run the same DuckDB release (`mb self version` names it).

        Args:
            port: the local port to listen on.
            token: what clients must present; the one kept in the state directory when omitted.
        """
        lake_ = Lake.at(workspace_root())
        with lake_.serving(port, token) as (uri, _):
            kept = "the one given" if token else str(lake_.token)
            said = f"serving {lake_.catalog} at {uri} (token: {kept}); Ctrl-C stops it"
            print(said, flush=True)
            with suppress(KeyboardInterrupt):
                while True:
                    time.sleep(3600)

    @lake.command(name="import")
    def import_(*, again: bool = False, output: Output = _COMPACT) -> int:
        """Import this workspace's file state into its state lake once, and prove it landed.

        Creates the lake (`lake.sqlite` beside a `lake/` data folder in the state directory) and
        appends, in one transaction, the dispatch registry, batch events, receipts and logs,
        cost ledgers, the offer catalog, held machines, study ledgers, the pulse memory, both
        digest memories, job scripts and closure listings. Then reads every source's rows back
        against the files and rebuilds the logs byte for byte, one row per comparison, and exits
        1 when any differs. The files are only read: nothing is moved, rewritten or deleted, and
        wandb folders, pins, source archives, recovery and environments are left out.

        Args:
            again: import even though the lake already holds an import, setting that lake aside
                under `lake.aside/` first rather than appending the same records twice.
        """
        with progress("importing the state directory into the lake"):
            tallies = Importer(Lake.at(workspace_root())).run(again=again)
        output.print_rows([tally.model_dump() for tally in tallies], title="lake import")
        return 0 if all(tally.ok for tally in tallies) else 1

    @paper.command(name="build")
    def paper_build(
        name: str,
        *,
        show: tuple[str, ...] = (),
        dpi: int = 110,
        output: Output = _COMPACT,
    ) -> int:
        """Build a declared manuscript and report everything wrong with it, exiting 1 on any.

        Built with tectonic in the workspace environment, then read back: errors, undefined
        references and citations, multiply defined labels and overfull boxes with the file and
        line each comes from, the page count, the page every section starts on, and whether
        the section `[papers.<name>] ends` names ends by page `limit`.

        Args:
            name: the `[papers.<name>]` manuscript.
            show: a phrase from the manuscript, repeatable; the page it appears on is rendered to a
                PNG beside the build and its path printed.
            dpi: the resolution a shown page renders at.
        """
        manuscript = board("local").paper(name)
        with progress(f"building {name}"):
            report = manuscript.check()
        _report(report, mode=output.mode)
        for phrase in show:
            print(manuscript.show(phrase, dpi=dpi).as_posix())
        return 1 if report.problems else 0

    git = App(
        name="git",
        help="Operate the workspace repository and its owned submodules as one tree.",
    )
    app.command(git)

    @git.command(name="status")
    def git_status(*, output: Output = _COMPACT) -> None:
        """Show every owned repository in the tree on one table, without touching the network.

        Owned means the owner in the remote URL is the workspace root's own or one `[git]
        owners` names; reference code pinned from anybody else is left out. Each row says the
        branch (or `detached`), how far HEAD is ahead of and behind its upstream as last
        fetched, how many paths are changed and untracked, and which remote branch already
        holds HEAD, empty for a commit a parent pointer could not yet be cloned at.

        Args:
        """
        with progress("reading the repository tree"):
            states = board("local").git().status()
        output.print_rows(
            [state.model_dump() for state in states],
            title="git status",
            columns=_GIT_STATUS_COLUMNS,
        )

    @git.command(name="pull")
    def git_pull(*, output: Output = _COMPACT) -> int:
        """Fast-forward every owned repository and bring submodule checkouts along, root first.

        Every owned remote is fetched at once, then the tree is walked from the root down.
        Nothing is merged or rebased: a diverged branch is held and named, and a fast-forward
        that would overwrite local changes is refused by git itself. A detached HEAD is put back
        on its trunk where that moves no commit. A submodule follows its parent's new pointer
        only when it sat on the old one, and one never checked out is cloned at the recorded
        pointer. Exits 1 when any repository was held or failed.

        Args:
        """
        with progress("pulling the repository tree"):
            steps = board("local").git().pull()
        return _stepped(steps, output, title="git pull")

    @git.command(name="commit")
    def git_commit(
        *, message: Annotated[str, Parameter(name=["--message", "-m"])], output: Output = _COMPACT
    ) -> int:
        """Commit every dirty owned repository, submodules first, then the pointers to them.

        Each commit lands on a branch: a detached HEAD is attached to its trunk when that is a
        fast-forward of the branch, and held otherwise, as is a repository behind its upstream
        and a parent whose submodule did not commit. Anything under a `[git] never-commit`
        pattern and files over the size ceiling that Git LFS does not carry stay out of the
        commit, unstaged; the row names the oversized ones and any never-commit path that was
        staged by hand. Exits 1 when any repository was held or failed.

        Args:
            message: the commit message, the same for every repository committed.
        """
        with progress("committing the repository tree"):
            steps = board("local").git().commit(message)
        return _stepped(steps, output, title="git commit")

    @git.command(name="push")
    def git_push(*, output: Output = _COMPACT) -> int:
        """Push every owned repository, submodules before the parents that point at them.

        A parent is pushed only once every submodule pointer its HEAD records is held by a
        branch of that submodule's remote. Git LFS objects are uploaded first. A remote that
        protects the tracked branch gets the commit on a branch named `<tool>/<branch>` after
        this tool instead, and the row asks for the pull request. HTTPS pushes to GitHub can
        use the `gh` login as a credential. Exits 1 when any repository was held or failed.

        Args:
        """
        with progress("pushing the repository tree"):
            steps = board("local").git().push()
        return _stepped(steps, output, title="git push")

    @git.command(name="check")
    def git_check(*, output: Output = _COMPACT) -> int:
        """Verify the tree is safe to clone and push, and exit 1 when anything fails.

        Fetches every owned repository and the foreign submodules they point at, then reports
        each pointer no branch of its remote holds, each diverged branch, each file in HEAD over
        the size ceiling and each LFS repository with no git-lfs here as `fail`, and a detached
        HEAD, unpushed or missing commits, and a checkout off its recorded pointer as `warn`.
        An empty table is a consistent tree.

        Args:
        """
        with progress("checking the repository tree"):
            findings = board("local").git().check()
        output.print_rows(
            [finding.model_dump() for finding in findings],
            title="git check",
            columns=_GIT_CHECK_COLUMNS,
        )
        return 1 if any(finding.verdict is Verdict.FAIL for finding in findings) else 0

    return app


# The columns each table always carries, so an empty one still renders its heading; the batch
# tables' totals row is also summed over this shape, and the job listing lines a settled row's
# empty live columns up under the live rows' own.
_SECTION_ROW = ("section", "verdict", "detail", "fix")
_CHANGE_COLUMNS = ("host", "handle", "outcome", "detail")
_GIT_STATUS_COLUMNS = (
    "repo",
    "owner",
    "branch",
    "head",
    "upstream",
    "ahead",
    "behind",
    "changed",
    "untracked",
    "published",
)
_GIT_STEP_COLUMNS = ("repo", "outcome", "detail")
_CI_COLUMNS = ("leg", "os", "step", "verdict", "seconds")
_GIT_CHECK_COLUMNS = ("repo", "check", "verdict", "detail")
_JOB_COLUMNS = (
    "state",
    "host",
    "project",
    "name",
    "handle",
    "cells",
    "quiet_s",
    "gpu_pct",
    "since",
    "starts",
    "submitted_at",
    "cause",
)
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
_SECTION_COLUMNS = ("number", "title", "page", "within")
_PROBLEM_COLUMNS = ("kind", "where", "detail")
_STATUS_COLUMNS = ("job", "target", "handle", "state", "verdict", "detail")
_VERDICT_COLUMNS = (
    "job",
    "handle",
    "target",
    "node",
    "verdict",
    "settled",
    "exit_code",
    "detail",
    "cause",
    "gates",
    "contended",
    "commit",
    "digest",
)


def _query_source(sql: str | None, file: Path | None) -> str | Path | None:
    """Select explicit SQL text or a file without interpreting strings as paths."""
    if sql is not None and file is not None:
        raise MissionError("SQL statement and --file are mutually exclusive")
    return file if file is not None else sql


def _expected(priced: JobEstimate, *, results: str = "") -> str:
    """One line saying where a submit lands, what it brings home, and what the meter will read.

    A rate means a rented target and carries its tail cost. No rate names why there is none, so
    a machine this workspace owns reads as `owned` while a provider nobody could get a price out
    of says that instead, and neither prints a bare zero that looks like a promise. Where the
    rate came from rides beside it for the same reason: a live offer can be rented at that price
    and a stored one is last week's.

    A run that pulls nothing home says so out loud, since the only moment that is cheap to
    notice is before the job goes out rather than after it wrote a wave of receipts on a cluster.

    results: the path this dispatch pulls back, empty when it declared none.
    """
    where = f"{priced.target} ({priced.kind}{', ' + priced.hardware if priced.hardware else ''})"
    pulled = f"results {results}" if results else "results NOT pulled back (no --fetch, no --node)"
    if not priced.rate_usd_hr:
        return (
            f"submit -> {where}: queue policy ok, {pulled}, {priced.rate_source}, expected $0.00"
        )
    return (
        f"submit -> {where}: queue policy ok, {pulled}, ${priced.rate_usd_hr:.2f}/hr "
        f"({priced.rate_source}), expected ${priced.expected_usd:.2f} (p90 ${priced.p90_usd:.2f})"
    )


def _unprinted(workspace: Board, handle: str, *, host: str) -> int:
    """Say where a run that has printed nothing stands, and exit on whether it still might.

    The durable registry row, always there for a real handle, is what this stands on; the
    backend is asked on top of it for the state and start time only a live scheduler knows, so
    a host that will not answer still gets a line built from what was last recorded.

    host: the alias narrowing a handle recorded on several hosts.
    """
    absent = f"no output on file for {handle}"
    try:
        record = workspace.verdicts().record(handle, host=host)
    except MissionError:
        print(absent, file=sys.stderr)
        return 1
    state = vocabulary.JobState(
        handle=record.handle,
        state=record.state,
        exit_code=record.exit_code,
        verdict=record.verdict or vocabulary.RUNNING,
    )
    with suppress(HostUnreachable, MissionError, OSError):
        state = workspace.job(record.handle, host=record.target).poll()
    if state.verdict in vocabulary.TERMINAL:
        print(absent, file=sys.stderr)
        return 1
    print(standing(state, submitted_at=record.submitted_at, host=record.target), file=sys.stderr)
    return 2


def _settled(settled: StreamVerdict, output: Output) -> int:
    """Print one stream's settled rows under its name, and answer its exit code.

    A stream with no rows says why on stderr, since an empty table looks identical whether the
    run has not started, the evidence landed elsewhere, or the harness wrote a shape this verb
    was never taught; stderr keeps a machine-readable stdout exactly what it was.
    """
    output.print_rows(
        [trial.model_dump() for trial in settled.trials],
        title=f"verdict: {settled.stream}",
        columns=_VERDICT_COLUMNS,
    )
    if settled.note:
        print(settled.note, file=sys.stderr)
    if settled.stalled:
        print(f"stalled: {settled.stalled}", file=sys.stderr)
    return settled.code


def _said(line: str) -> None:
    """One line a wait says while it blocks, on stderr and at once."""
    print(line, file=sys.stderr, flush=True)


def _told(result: CiResult) -> None:
    """One settled step's transcript, on stderr and at once, nothing for a step never run."""
    if transcript := result.transcript:
        _said(transcript)


def _agreed() -> bool:
    """Ask once at the terminal whether to dispatch, and say what was typed back.

    The question shares stderr with the expectation line it follows, since stdout belongs to the
    handle or the document this verb prints once the dispatch has actually happened. An input
    that ends unanswered is a no: Windows reports its null device as a terminal, so a script that
    redirected stdin there reaches this question too, and should pass `--yes`.
    """
    print("dispatch? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        answer = input()
    except EOFError:
        print("no answer; pass --yes to dispatch without asking", file=sys.stderr)
        return False
    return answer.strip().lower() in {"y", "yes"}


def _followed[T](passes: Iterator[T], label: str, show: Callable[[T], None]) -> None:
    """Show each pass as it lands until the passes end or the reader interrupts.

    Each pass is taken inside its own progress block rather than iterated over, so the sweep's
    own noise is diverted the way a single pass's is and each report prints as its own document.
    """
    with suppress(KeyboardInterrupt, StopIteration):
        while True:
            with progress(label):
                passed = next(passes)
            show(passed)


def _judged(sections: list[Section], *, mode: str | None, title: str) -> None:
    """Print a machine's findings as a table of their own, under the record they judge."""
    if mode is None:
        print(f"\n# {title}")
    rows(
        [section.model_dump() for section in sections], mode=mode, fields=_SECTION_ROW, title=title
    )


def _onboarded(workspace: Board, report: HostSetup, output: Output, *, title: str) -> None:
    """Print a setup record, then the findings its read-back census adds up to.

    The JSON mode keeps the record whole and adds the findings under `findings`, so a script
    reads both from one document.
    """
    census = report.hardware.system if report.hardware else System()
    findings = workspace.findings(census)
    if output.mode == "json":
        payload: dict[str, Node] = {
            **report.model_dump(),
            "findings": [row.model_dump() for row in findings],
        }
        output.print_record(payload, title=title)
        return
    output.print_record(report.model_dump(), title=title)
    _judged(findings, mode=output.mode, title=f"findings: {report.host}")


def _heading(output: Output, title: str) -> None:
    """Name the next compact table when a verb prints several (rich tables carry their title)."""
    if output.mode is None:
        print(f"\n# {title}")


def _facts(workspace: Board, alias: str, output: Output) -> None:
    """Print one host's probed facts, then the findings that judge it against the workspace."""
    with progress(f"probing {alias}"):
        found = workspace.facts()
    _heading(output, f"facts: {alias}")
    output.print_record(found.model_dump(), title=f"facts: {alias}")
    if output.mode != "json":
        _judged(workspace.findings(found.system), mode=output.mode, title=f"findings: {alias}")


def _gpus(board: Callable[[str], Board], names: Sequence[str], output: Output) -> None:
    """Print who holds each card on each named host; an unreachable host is a row saying why."""
    listed: list[dict[str, str | int | float | bool]] = []
    readings: dict[str, JsonValue] = {}
    for name in names:
        try:
            with progress(f"reading the cards of {name}"):
                occupancy = board(name).occupancy()
        except (MissionError, OSError, ValueError) as error:
            why = str(error).splitlines()[0][:80]
            listed.append({"host": name, "card": "", "holders": f"unreachable: {why}"})
            continue
        readings[name] = occupancy.model_dump(mode="json")
        listed.extend(occupancy_rows(name, occupancy))
    if output.json:
        # The wire form a remote read parses: readings keyed by host.
        print(dumps(readings))
        return
    _heading(output, "gpus")
    output.print_rows(listed, title="gpus")


def _sectioned(
    sections: list[Section], output: Output, *, title: str, heading: bool = False
) -> int:
    """Print a report's rows and answer its exit status: 1 when any row failed."""
    if heading:
        _heading(output, title)
    output.print_rows(
        [section.model_dump() for section in sections], title=title, columns=_SECTION_ROW
    )
    return 1 if failed(sections) else 0


def _changed(changes: Sequence[Change], output: Output, *, title: str) -> None:
    """Print one edit's constraint move and every pin its solve moved, as one table.

    Both are one fact, a version moving somewhere, so they share one shape; `where` tells a
    manifest table's requirement from the lock's pins the solve dragged along.
    """
    output.print_rows([change.model_dump() for change in changes], title=title)


def _tabled(
    payloads: list[dict[str, Node]],
    columns: Sequence[str],
    *,
    summing: Sequence[str],
    output: Output,
    title: str,
) -> None:
    """Print an analysis table: one row per job, then one row adding up what the batch costs.

    The total rides in the table rather than beside it, since every mode a caller can ask for
    renders rows and a figure printed outside them would be the one number `--json` dropped.
    """
    total = totals(payloads, columns=columns, summing=summing)
    output.print_rows([*payloads, total], title=title, columns=columns)


def _status(status: BatchStatus, output: Output) -> None:
    """Print one pass over a batch, its still-running count in the heading."""
    output.print_rows(
        [job.model_dump() for job in status.jobs],
        title=f"{status.batch}: {status.running} running",
        columns=_STATUS_COLUMNS,
    )


def _stepped(steps: Sequence[Step], output: Output, *, title: str) -> int:
    """Print one row per repository a tree verb walked, exiting 1 when any did not settle."""
    output.print_rows(
        [step.model_dump() for step in steps], title=title, columns=_GIT_STEP_COLUMNS
    )
    return 0 if all(step.outcome.settled for step in steps) else 1


def _held(held: Held, output: Output, *, title: str) -> None:
    """Print one held machine: its alias, where it came from, what it costs, when it ends."""
    payload: dict[str, Node] = {
        "alias": held.alias,
        "provider": held.provider,
        "handle": held.handle,
        "gpu": held.gpu,
        "usd_hr": held.usd_hr,
        "deadline": held.deadline.isoformat(),
        "root": held.profile.root,
    }
    output.print_record(payload, title=title)


def _report(report: PaperReport, *, mode: str | None) -> None:
    """Print one manuscript check: the summary, where each section starts, and every problem.

    The JSON mode prints the report whole, one document a script can read; the other modes
    print three tables, the problem table naming its columns even when it is empty so a clean
    build still says so.
    """
    if mode == "json":
        record(report.model_dump(mode="json"), mode=mode, fields=(), title="paper")
        return
    summary: dict[str, Node] = {
        "pdf": report.pdf,
        "pages": report.pages,
        "limit": report.limit,
        "ends": report.ends,
        "ends_on": report.ends_on,
        "problems": len(report.problems),
    }
    record(summary, mode=mode, fields=(), title=f"paper: {report.paper}")
    last = report.limit or report.pages
    rows(
        [{**section.model_dump(), "within": section.page <= last} for section in report.sections],
        mode=mode,
        fields=_SECTION_COLUMNS,
        title="sections",
    )
    rows(
        [problem.model_dump(mode="json") for problem in report.problems],
        mode=mode,
        fields=_PROBLEM_COLUMNS,
        title="problems",
    )


def _answers(given: Sequence[str]) -> dict[str, str]:
    """The `question=value` pairs a caller passed, refusing one written without its value."""
    split = [pair.partition("=") for pair in given]
    if bare := [pair for pair, separator, _ in split if not separator]:
        raise MissionError(f"answers are written question=value, not {bare[0]!r}")
    return {question: answer for question, _, answer in split}


def _changes(report: MonitorReport) -> list[dict[str, str]]:
    """What moved this pass, one row each: every job that settled or was re-dispatched, every
    dispatch a quota is still holding, and every host that could not be reached.

    A held dispatch is here because nothing else reports it: it has no handle a scheduler knows
    and no verdict to settle, so a sweep silent about it is a job waiting in silence.
    """
    moved = [
        *((run.target, run.handle, "dispatched", run.name) for run in report.resumed),
        *((run.target, run.handle, "held", run.reason) for run in report.held),
        *((job.target, job.handle, "ok", job.pulled_path or "") for job in report.finished),
        *((job.target, job.handle, "failed", job.reason) for job in report.failed),
        *((host.host, "", "unreachable", host.reason) for host in report.unreachable_hosts),
    ]
    return [dict(zip(_CHANGE_COLUMNS, row, strict=True)) for row in moved]


def _forget_openssh_descriptors() -> None:
    """Drop the descriptor table Win32-OpenSSH hands the process it spawns.

    sshd passes its child a `<guid>_POSIX_FD_STATE` naming that child's own handles, and the
    variable rides every environment inherited after it. An `ssh.exe` this tool starts reads the
    stale table as its own and hangs before reading its config, so a center driven over ssh
    reached no host (pedro-home, 2026-09-26). plumbum copied the environment at import, so its
    copy loses the table too.
    """
    for name in [name for name in os.environ if name.endswith("_POSIX_FD_STATE")]:
        del os.environ[name]
        del localhost.env[name]


def main() -> None:
    """Console entry point, `MissionError` printed to stderr without a traceback, exit 1.

    A trailing-command verb gets the `--` its command implies, so nothing typed after the
    command is ever read as this tool's. The verbs a person or an agent types log short plain
    lines; the ones a dispatched job runs keep the JSON its log readers parse.
    """
    _forget_openssh_descriptors()
    if not Project().variable("LOG_FORMAT").read() and sys.argv[1:2] != ["execute"]:
        configure(output="line")
    app = build()
    profiled = Project().variable("PROFILE").read()
    try:
        if profiled:
            _profiled(Path(profiled), lambda: app(Delimiter(app).placed(sys.argv[1:])))
        else:
            app(Delimiter(app).placed(sys.argv[1:]))
    except (MissionError, NoWorkspace) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None


def _profiled(report: Path, command: Callable[[], object]) -> None:
    """Run `command` under this tool's own profiler, every one of its modules instrumented, and
    write the span report to `report`: `MB_PROFILE=prof.txt mb job submit ...` names the slow
    function of any verb, on any machine, with nothing else installed."""
    import pkgutil  # noqa: PLC0415  (only a profiled run pays for walking the package)

    from . import __path__ as package  # noqa: PLC0415
    from .profile import Feature, Profiler  # noqa: PLC0415

    # Entry points and the standard-library agents are left out: importing them runs them.
    skipped = ("__main__", "_refresh", "agent.program", "center.remote", ".profile")
    modules = [
        found.name
        for found in pkgutil.walk_packages(package, f"{__package__}.")
        if not any(skip in found.name for skip in skipped)
    ]
    profiler = Profiler(features=Feature.SPANS, gpus=(), auto=modules)
    try:
        with profiler:
            command()
    finally:
        report.write_text(profiler.report(), encoding="utf-8", newline="\n")
        print(f"profile written to {report}", file=sys.stderr)
