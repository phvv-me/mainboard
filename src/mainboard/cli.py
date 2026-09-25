import sys
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from json import dumps, loads
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn

from cyclopts import App, Parameter
from plumbum import local as localhost
from pydantic import JsonValue

from . import staleness
from .batch.spec import BatchSpec, Selection
from .board import Board
from .center.migrate import Migration
from .center.standalone import Standalone
from .center.verify import Verification
from .ci import LocalLeg, Matrix, Package
from .context.resolver import Resolver
from .core.errors import MissionError
from .core.project import Project
from .core.section import Section, Verdict, failed
from .delimiter import Delimiter
from .dispatch import vocabulary
from .dispatch.commandline import joined
from .dispatch.evidence import printed
from .dispatch.schedulers import HostUnreachable, standing
from .durable import schedule
from .help import Help
from .holds import Holds
from .jobs import lanes as lanes_module
from .lint import Inventory, Linter
from .listing import Listing
from .manifest.loading import composition, load, load_plot_config
from .manifest.schema.plot import PlotStyle
from .probe.occupancy import rows as occupancy_rows
from .probe.system import System
from .proc import Processes
from .render import diverted, install_traceback, mode_of, plain, progress, record, rows, totals
from .results import Results
from .runtime.job import Job
from .runtime.runner import Runner
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
    from .git import Step
    from .manifest.held import Held
    from .manuscript import Report as PaperReport
    from .render.values import Node
    from .verdicts import StreamVerdict


@Parameter(name="*")
@dataclass(frozen=True, kw_only=True)
class Output:
    """How a verb prints its document: the default rich table or one of two compact modes."""

    json: bool = False
    """print canonical JSON instead of the default rich table."""
    agent: bool = False
    """print the compact tabular mode instead of the default rich table."""
    fields: str = ""
    """a comma-separated projection over the printed fields."""

    def __post_init__(self) -> None:
        """Refuse both compact modes at once before the verb does any work."""
        mode_of(json_mode=self.json, agent=self.agent)

    @property
    def mode(self) -> str | None:
        """The render key, `None` for the default rich table."""
        return mode_of(json_mode=self.json, agent=self.agent)

    def projection(self, default: Sequence[str] = ()) -> Sequence[str]:
        """The `--fields` names, trimmed and blanks dropped, `default` when none were given."""
        return tuple(part.strip() for part in self.fields.split(",") if part.strip()) or default

    def print_rows(
        self, payloads: Sequence[Mapping[str, Node]], *, title: str, columns: Sequence[str] = ()
    ) -> None:
        """Print many entities, `columns` keeping an empty table's heading unless projected."""
        rows(payloads, mode=self.mode, fields=self.projection(columns), title=title)

    def print_record(self, payload: Mapping[str, Node], *, title: str) -> None:
        """Print one entity."""
        record(payload, mode=self.mode, fields=self.projection(), title=title)


_RICH = Output()


@Parameter(name="*")
@dataclass(frozen=True, kw_only=True)
class Declared:
    """A batch's declaration beyond its spec file: inline jobs, a selection, `[vars]` values."""

    job: tuple[str, ...] = ()
    """a `target:command` job, repeatable, for a batch declared without a file."""
    name: str = ""
    """the batch's name when declared with `--job` rather than a file."""
    only: str = ""
    """the plan's jobs to act on, names or `kind-*` globs, comma-separated; all when unset."""
    set_: Annotated[tuple[str, ...], Parameter(name="--set", negative="")] = ()
    """a `name=value` filling one of the spec file's `[vars]`, repeatable."""

    def batch(self, board: Board, spec: str) -> Batch:
        """The declared batch over `board`'s workspace, `spec` relative to its root."""
        return board.batch(self._spec(board.root, spec), selection=Selection.of(self.only))

    def _spec(self, root: Path, spec: str) -> BatchSpec:
        if spec:
            return BatchSpec.load(root / spec, _answers(self.set_))
        if self.set_:
            raise MissionError("--set fills a spec file's [vars]; a --job batch declares none")
        if not self.job:
            raise MissionError("declare a batch: a spec file, or --job target:command")
        return BatchSpec.inline(self.name or "batch", self.job)


_SPEC_ONLY = Declared()


def build(root: Path | None = None) -> App:
    """The CLI application, workspace discovery deferred until a verb runs.

    root: an explicit workspace root, discovered from the cwd when None.
    """
    project = Project()
    app = App(name=project.name, help="One interface for environments, dispatch, and hardware.")

    def workspace_root() -> Path:
        return root or project.find_root(Path.cwd())

    def board(on: str) -> Board:
        return Board(workspace_root(), host=on)

    @app.command(name="help")
    def help_(*query: str) -> None:
        """Show command help or search shipped docs and Python docstrings without a workspace.

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

        command: the command tokens, from the first token that is not an option of this verb,
            passed on verbatim with its own flags; a job's arguments follow `--`.
        on: the host alias, `local` for this machine.
        env: an environment name overriding the profile's choice.
        container: a container override, `none` forcing bare.
        """
        return board(on).run(command, env=env, container=container)

    @app.command(version_flags=[])
    def submit(
        *command: str,
        on: str,
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
        yes: bool = False,
        output: Output = _RICH,
    ) -> None:
        """Dispatch a command, or a job spelled `path/to/file.py::name`, on a host.

        A job ships exactly the code it imports and the directory it lives in, runs through the
        one runner in the host's environment, and stamps its receipts with a provenance scoped
        to those files. A command ships the mirror and keeps the whole-tree provenance. Either
        way the handle is printed, bare unless a compact mode asks for the whole record.

        The expectation prints first, the same manners a batch has: the resolved target, the
        queue policy's admission, and what the meter will say, a provider's rate for a rented
        host and zero for owned hardware. At a terminal the dispatch then asks once; in a
        script or under `--yes` it proceeds, and the line is printed either way.

        command: the command tokens, from the first token that is not an option of this verb,
            or `path/to/file.py::name` and, after `--`, the arguments its application takes.
        gpu_name: the GPU type to rent, for a metered provider host.
        max_usd: the spend cap a provider host refuses to submit without.
        attempt: the 1-based try number feeding expression defaults.
        fetch: a results path recorded for later `pull`, the node's own evidence directory when
            unset and `--node` names one.
        node: the ledger slug this run serves, carried into its record and receipts.
        needs: a workspace-relative data path the job reads on the host, repeatable, joining
            the ones the job file declares; refused for a command, which reaches the mirror.
        yes: dispatch without asking, what a script passes.
        """
        line = joined(command)
        workspace = board(on)
        priced = workspace.expectation(
            line,
            queue=queue,
            walltime=walltime,
            mem_gb=mem_gb,
            gpus=gpus,
            gpu_name=gpu_name,
            max_usd=max_usd,
            attempt=attempt,
        )
        pulled = workspace.results(fetch, node=node, command=line)
        print(_expected(priced, results=pulled), file=sys.stderr)
        if not yes and sys.stdin.isatty() and not _agreed():
            raise SystemExit(1)
        with progress(f"submitting on {on}") as stage:
            job = workspace.submit(
                line,
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
            print(job.handle.id)
            return
        output.print_record(job.handle.model_dump(), title="handle")

    @app.command
    def add(
        spec: str,
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "conda",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
        output: Output = _RICH,
    ) -> None:
        """Declare a dependency in the manifest and re-solve, showing what the lock did.

        A bare name is pinned to whatever the ecosystem's index publishes right now, and a name
        carrying its own constraint is written exactly as given. The table it lands in is the
        one the flags name, and where the manifest already writes that kind of requirement in a
        particular table, the edit joins it there.

        lang: the ecosystem whose resolver installs it.
        env: an environment name, the workspace-wide table when omitted.
        dev: declare it as a development-only requirement.
        resolve: `--no-resolve` stages several edits to solve once.
        fields: a comma-separated projection over name/where/before/after.
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
        output: Output = _RICH,
    ) -> None:
        """Drop a dependency from the manifest and re-solve, showing what the lock did.

        With no flags the whole manifest is searched, so dropping a requirement never asks
        which table it was written into. `--lang`, `--env` and `--dev` narrow that search to one
        ecosystem's, one environment's or the development-only tables, which is also how a name
        declared in more than one table is told apart.

        fields: a comma-separated projection over name/where/before/after.
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
        output: Output = _RICH,
    ) -> None:
        """Move one dependency to its newest release, or the whole lock forward in its bounds.

        Named, the constraint itself is rewritten to what the ecosystem publishes now, which is
        the only way past a ceiling the manifest declares. Unnamed, the manifest is untouched
        and the lock is re-solved against the indexes, moving every pin as far as the declared
        constraints already allow. `--lang`, `--env` and `--dev` narrow the search for the name
        to one ecosystem's, one environment's or the development-only tables.

        fields: a comma-separated projection over name/where/before/after.
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
        output: Output = _RICH,
    ) -> None:
        """Scaffold a project from one of this workspace's declared templates.

        Which templates exist is the manifest's `[templates]` table, and the first declared one
        is what this renders when none is named. Every answer a template asks for comes from
        the name, from what the workspace already declared, or from the template's own default,
        so a project stays one argument while `--answer` covers the rest. The task rows a
        template generates are printed rather than pasted, since the root manifest's task table
        is hand-curated and the same project has to reach the type checker's search path beside
        it, and half of that edit landing on its own is worse than none of it.

        name: the project name, which becomes its slug, its package and its task prefix.
        template: a declared template name or any location copier accepts.
        description: the one sentence the README and the task rows carry.
        dest: where to render it, under the template's own declared home when omitted.
        answer: a further `question=value` for the template, repeatable.
        fields: a comma-separated projection over project/path/tasks/paste/snippet.
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
    def doctor(env: str = "", *, output: Output = _RICH) -> int:
        """Say whether this workspace is fit to work in, and exit nonzero when it is not.

        The questions asked at once and bounded: does the manifest still say something
        coherent, is what is installed the environment it describes, what compute answers right
        now, do onboarded hosts still match the manifest, does a periodic pass settle jobs, and
        does every declared gate still hold. A section reports the one command that repairs it,
        and only a genuinely broken workspace fails, so a sleeping host or a provider nobody has
        a key for is a word rather than a nonzero exit. Whether this machine can be the center
        is `center verify`'s question.

        env: the environment to examine, the local profile's own when omitted.
        fields: a comma-separated projection over section/verdict/detail/fix.
        """
        with progress("examining the workspace"):
            sections = board("local").doctor(env).sections()
        return _sectioned(sections, output, title="doctor")

    @app.command
    def install(env: str = "", *, resolve: bool = False, profile: str = "") -> None:
        """Compile the manifest and install the environment on this machine.

        Another machine is onboarded with `setup`, which ends by running this verb there.

        env: the environment name, this machine's declared profile choice when omitted.
        resolve: allow a fresh dependency solve when the lock is stale.
        profile: the declared host profile describing this machine, so the generated activation
            carries that host's modules; what `setup` passes when a host installs its own.
        """
        with progress(f"installing {env or 'the environment'}") as stage:
            board("local").install(env, resolve=resolve, profile=profile, watch=stage)

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

    @app.command
    def setup(
        host: str,
        *,
        env: str = "",
        resolve: bool = False,
        sync_only: bool = False,
        output: Output = _RICH,
    ) -> None:
        """Onboard a host until it can run jobs, then show what it became and what that means.

        The host is provisioned with the environment its declared profile names, so a host that
        runs `serving` is set up for serving without repeating the name here. The lock this
        workspace solved ships with the mirror and the host installs from it. The record is
        followed by the findings `facts` shows, judged from the software census read back
        through the new activation.

        env: an environment name overriding the host profile's own.
        resolve: let the host run its own dependency solve instead of installing the shipped
            lock, which puts that host's compiler in the resolution path.
        sync_only: re-mirror and re-provision a host already set up, skipping the tool
            reinstall and the hardware probe, what `sync` does.
        fields: a comma-separated projection over the setup record's fields.
        """
        workspace = board(host)
        with progress(f"setting up {host}") as stage:
            report = workspace.install(env, resolve=resolve, watch=stage, sync_only=sync_only)
        _onboarded(workspace, report, output, title="setup")

    @app.command
    def sync(host: str, *, env: str = "", output: Output = _RICH) -> None:
        """Re-mirror a host already set up and re-provision it from the shipped lock.

        The fast path back after source moved: the tool is not reinstalled and the hardware is
        not probed again, so a Python edit reaches the host in the time the mirror takes. A
        host never set up needs `setup` first, which is where the probe and the tool come from.

        env: an environment name overriding the host profile's own.
        fields: a comma-separated projection over the setup record's fields.
        """
        workspace = board(host)
        with progress(f"syncing {host}") as stage:
            report = workspace.install(env, resolve=False, watch=stage, sync_only=True)
        _onboarded(workspace, report, output, title="sync")

    @app.command
    def hold(
        provider: str,
        *,
        for_: Annotated[str, Parameter(name="--for")],
        as_: Annotated[str, Parameter(name="--as")] = "",
        gpu_name: str = "",
        gpus: int = 0,
        max_usd: float = 0.0,
        env: str = "",
        output: Output = _RICH,
    ) -> None:
        """Rent a machine and keep it as an ssh host until a deadline, set up and ready for jobs.

        A rental per job rebuilds the environment every time; a held machine is set up once and
        then takes `submit --on <alias>` in seconds. The machine gets an alias in the ssh config,
        onboards like `setup`, and is recorded with its deadline, which `monitor` and `compute`
        both enforce by releasing it, so a forgotten hold stops billing on time.

        provider: the provider host to rent through, `vast` say.
        for_: how long to keep it once it is ready, `3h`, `90m` or `1h30m`.
        as_: the alias to reach it by, `<provider>-<card>` when omitted.
        gpu_name: the card to rent, in the provider's own spelling.
        gpus: cards per machine, the provider profile's default when 0.
        max_usd: the spend cap over the whole hold, landing included, the provider's default
            when 0.
        env: the environment to set up, the provider profile's own when omitted.
        fields: a comma-separated projection over the hold's fields.
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

    @app.command
    def release(alias: str, *, output: Output = _RICH) -> None:
        """End a held machine now: stop its billing, settle its record and drop its alias.

        alias: the held machine's alias, as `hold` printed it.
        fields: a comma-separated projection over the hold's fields.
        """
        with progress(f"releasing {alias}"):
            held = Holds(board("local")).release(alias)
        _held(held, output, title="release")

    @app.command
    def compute(*, output: Output = _RICH) -> None:
        """List every compute path this workspace can reach, with prices and credit where cheap.

        Held machines past their deadline are released first. Then this machine, then each
        declared host with whether it answers and whether it was ever set up, then each provider
        backend with whether its credentials are present here, what the account has left to
        spend, and a live rate where asking for one is cheap, followed by every machine the
        provider says this account is renting, named by its hold when this workspace holds it.
        Every probe is bounded and runs beside the others, so the whole fleet answers in the
        time the slowest one takes, and a host that is down or a provider with no key is a row
        rather than a failure. No credential is ever printed, only whether one was found.

        Provisioned means cached setup, not current job readiness. Hardware may be stale;
        cached_at names its onboarding observation and observed_at names this live survey.
        GPU availability is not checked. PBS/Slurm reachability concerns the login endpoint,
        not an allocated compute node. Inspect jobs and facts before scheduling experiments.

        fields: comma-separated name/kind/access/detail/usd_hr/credit_usd/observed_at/cached_at.
        """
        workspace = board("local")
        for released in Holds(workspace).expire():
            gone = f"{released.provider} {released.handle}"
            print(f"released {released.alias}, {gone}", file=sys.stderr)
        with progress("probing every compute path"):
            paths = workspace.compute().paths()
        output.print_rows([path.model_dump() for path in paths], title="compute")

    @app.command
    def collect(path: str, *, on: str, json: bool = False) -> None:
        """Collect remote evidence for queries, including runs started directly on that node.

        Complete files are immutable. Conflicts preserve the local copy and fail collection.
        Live event snapshots exclude incomplete records; queries deduplicate overlapping frames.
        A new query sees published files. Collection is not one transaction across servers.

        path: workspace-relative results file or directory, using forward slashes on every OS.
        on: declared SSH host; root and bootstrap Python come from its profile, a `~` root
            expanded by that Python when no setup has placed it yet. Python is a command in that
            host's SSH login shell, usually python3; quote an absolute interpreter path as that
            shell requires. No remote Mainboard is needed.
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

        Views: runs, trials, events, metrics, artifacts, jobs. Project scopes science views;
        jobs always shows the fleet. Monitor refreshes tracked jobs; collect also imports
        results from native remote runs. Neither requires a shared database service.

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
        results = Results(workspace_root())
        if out is not None:
            print(results.export(source, out, project=project))
            return
        frame = results.query(source, project=project)
        Output(json=json).print_rows(loads(frame.write_json()), title="results")

    @app.command
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

        sql: define the table, including any filtering, grouping, and ordering.
        file: read a UTF-8 SQL file instead of the positional statement.
            Both the file and paths inside SQL resolve from the caller's current directory.
        config: overlay project styles and figures from this manifest-format TOML file.
            It does not select an environment or change path resolution.
        figure: render a named [figures.<name>] specification; omit SQL, x, and y.
            Panels use native Seaborn marks and Matplotlib settings, without estimation.
        x, y, hue: column names; omit hue for one series.
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
                "uv tool install --from './packages/mainboard[wandb,plot]' mainboard --force"
            ) from fault
        settings = PlotStyle()
        specification = None
        manifest = load_plot_config(workspace_root() / Project().manifest, config)
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

    @app.command
    def monitor(*, every: str = "", watch: float = 0.0, output: Output = _RICH) -> None:
        """Settle every dispatched job that ended since the last pass, then exit.

        The durable sweep a periodic cron runs. It resolves every job the dispatch cache still
        owes an outcome for, pulls back the results of the ones that just finished, records their
        verdicts in the study ledgers that own them, and reports only what changed, so a second
        pass with nothing new says exactly that. A host that cannot be reached is reported with
        why and its jobs are left for the next pass, so no outcome ever depends on the process
        that dispatched the job still being alive. A compact mode prints the whole report.

        `--every` is what makes that last sentence true of the schedule as well as of the pass.
        It hands the sweep to this machine's own service manager, so the period outlives the
        session that asked for it, and `--every 0` hands it back.

        every: install the periodic pass at this period (`20m`), `0` removing what is installed.
        watch: seconds between repeated passes in the foreground, one pass and exit when 0.
        fields: a comma-separated projection over the report's fields.
        """
        if every:
            settling = schedule(workspace_root(), every)
            print(f"{project.name}: {settling.detail}")
            if settling.fix:
                print(f"{project.name}: run `{settling.fix}`")
            return
        sweep = board("local").monitor()
        label = "sweeping dispatched jobs"
        show = partial(_present, output=output)
        if not watch:
            with progress(label):
                report = sweep.once()
            show(report)
            return
        _followed(sweep.watch(watch), label, show)

    @app.command
    def facts(on: str = "local", *, output: Output = _RICH) -> None:
        """Show the host's probed hardware and software facts, then what they mean here.

        The facts are the hardware inventory beside the software census: operating system and
        version, shells, filesystem case sensitivity, symbolic link and long path support, the
        git settings a clone inherits, every tool with its version, and the NVIDIA driver, its
        CUDA, each card's compute capability and memory. The findings table below them judges
        that machine against this workspace, its platforms, its lock, the CUDA floor and the
        card memory its profile declares, one row each with the command that repairs it.
        `--json` prints the facts alone, the wire snapshot one machine answers another with.

        on: the host alias to probe, `local` for this machine.
        fields: a comma-separated projection over the fact fields.
        """
        workspace = board(on)
        with progress(f"probing {on}"):
            found = workspace.facts()
        output.print_record(found.model_dump(), title="facts")
        if output.mode != "json":
            _judged(workspace.findings(found.system), mode=output.mode, title=f"findings: {on}")

    @app.command
    def gpus(
        on: str = "local", *, every: bool = False, json: bool = False, agent: bool = False
    ) -> None:
        """Show who holds each card right now: utilization, memory and the processes on it.

        The screen that says whether a card can take an acquisition. `facts` describes the
        hardware and `jobs` what this workspace dispatched; a resident server or another user's
        run appears only here. `--json` prints the readings keyed by host, one line for a single
        host, which is what a remote read parses.

        on: the host alias to read, `local` for this machine.
        every: read this machine and every declared ssh host instead of one host.
        json: print the readings as JSON instead of the table.
        agent: print the compact tabular mode instead of the default rich table.
        """
        manifest = load(workspace_root() / project.manifest)
        remote = [alias for alias, profile in manifest.hosts.items() if profile.kind == "ssh"]
        names = ["local", *remote] if every else [on]
        listed: list[dict[str, str | int | float | bool]] = []
        readings: dict[str, JsonValue] = {}
        for name in names:
            try:
                with progress(f"reading the cards of {name}"):
                    occupancy = board(name).occupancy()
            except (MissionError, OSError, ValueError) as error:
                why = str(error).splitlines()[0][:80]
                listed.append(
                    {
                        "host": name,
                        "card": "",
                        "util_pct": 0,
                        "memory_gb": 0.0,
                        "of_gb": 0.0,
                        "free": False,
                        "holders": f"unreachable: {why}",
                    }
                )
                continue
            readings[name] = occupancy.model_dump(mode="json")
            listed.extend(occupancy_rows(name, occupancy))
        if json:
            print(dumps(readings, indent=2) if every else dumps(readings.get(on, {})))
            return
        Output(agent=agent).print_rows(listed, title="gpus")

    @app.command
    def check(*, on: str = "", env: str = "", container: str = "", output: Output = _RICH) -> None:
        """Validate the workspace manifest, showing what it declares or what a host resolves to.

        on: a host alias, `local` for this machine, to show the execution plan it resolves to
            instead of the manifest's declarations.
        env: with `--on`, an environment name overriding the profile's choice.
        container: with `--on`, a container name overriding the profile's, `none` for bare.
        fields: a comma-separated projection over the declared or planned fields.
        """
        manifest = load(workspace_root() / project.manifest)
        if on:
            resolved = Resolver(manifest).plan(on, env=env, container=container)
            output.print_record(resolved.model_dump(), title="plan")
            return
        if env or container:
            raise MissionError("--env and --container override a host's plan; pass --on too")
        payload: dict[str, Node] = {
            "workspace": manifest.workspace.name,
            "environments": tuple(sorted(manifest.envs)),
            "containers": tuple(sorted(manifest.containers)),
            "hosts": tuple(sorted(manifest.profiles())),
            "papers": tuple(sorted(manifest.papers)),
            "tasks": tuple(sorted(manifest.tasks)),
        }
        output.print_record(payload, title="check")

    @app.command
    def lint(*paths: Path, check: bool = False, only: str = "", json: bool = False) -> int:
        """Fix what can be fixed, then check, over the changed files or everything under PATHS.

        With no path the pass reads every file that differs from HEAD or is new, submodules
        entered and deletions included, so the everyday call costs what the edit did. A path
        widens it to every file git tracks or would track at or beneath it, so `lint .` at the
        root reads the whole workspace. The exit is nonzero when a file was rewritten or a step
        failed, the one answer a person, an agent, a hook and a CI job all act on.

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
        report = Linter(root, load(root / project.manifest), check=check, only=steps).lint(files)
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
        package: Path | None = None, *, matrix: bool = False, output: Output = _RICH
    ) -> int:
        """Run the gate `[tool.mainboard.ci]` declares, exactly as the package's CI job runs it.

        The package is the nearest directory at or above PACKAGE whose pyproject.toml declares a
        gate, and needs no workspace. Its steps run in order from the package directory and stop
        at the first failure; each step's output goes to stderr as it settles, the table to
        stdout. With `--matrix` the gate also runs, at the same time, on every `[ci] hosts` entry
        of a supported platform this machine is not, the working tree shipped there as it
        stands, and only failing steps print their output. Exits 1 when any step failed.

        package: a directory inside the package, the working directory when omitted.
        matrix: also run on one declared host per other platform, the check before a push.
        fields: a comma-separated projection over leg/os/step/verdict/seconds.
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

    batch = App(name="batch", help="Prepare, price, dispatch and watch many jobs as one flow.")
    app.command(batch)

    lanes = App(name="lanes", help="Run one pytest lane on many hosts, a job per group of cells.")
    app.command(lanes)

    @lanes.command(name="run")
    def lanes_run(
        target: str,
        *,
        on: str = "local",
        group: str = "",
        per_job: int = 0,
        rerun: bool = False,
        timeout: float = 900.0,
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        gpu_name: str = "",
        max_usd: float = 0.0,
        node: str = "",
        dry_run: bool = False,
        wait: bool = False,
        yes: bool = False,
        agent: bool = False,
    ) -> int:
        """Run every cell of a lane on each named host, one dispatched job per group of cells.

        The lane's own parametrization is the plan: its cells are collected here, grouped by a
        parametrize value or sliced, and each group becomes one job that runs its cells as
        fresh processes through the runner's `--fresh` mode. `local` runs the groups in place;
        every other host gets a submission with the lane's declared needs and pins shipped, and
        `--wait` blocks on every handle through the same durable sweep `wait` runs, which
        pulls the receipts home. A Windows host cannot take a submission yet; a roster
        containing one is refused before any host is dispatched.

        target: the lane, `path/to/file.py::test`.
        on: comma-separated host aliases, `local` for this machine.
        group: a parametrize name whose value names each job's cells, `model` say.
        per_job: how many cells one job takes when no name groups them, 0 for all in one.
        rerun: run cells whose data is already complete.
        timeout: seconds one cell may take before its process is killed.
        queue, walltime, mem_gb: what a queued host's scheduler is asked for.
        gpus: cards per job, the host profile's default when 0.
        gpu_name: the card a provider host rents, in the provider's own spelling.
        max_usd: the spend cap of one rental on a provider host; one group is one rental.
        node: the ledger slug the receipts serve, the directory under `experiments` when unset.
        dry_run: print the plan and dispatch nothing.
        wait: block until every dispatched job settles.
        yes: dispatch without asking.
        agent: print the compact tabular mode instead of the default rich table.
        """
        manifest = load(workspace_root() / project.manifest)
        hosts = [alias.strip() for alias in on.split(",") if alias.strip()]
        if unsupported := [
            host
            for host in hosts
            if host in manifest.hosts and manifest.hosts[host].platform == "win-64"
        ]:
            raise MissionError(
                f"queued lanes do not support Windows hosts: {', '.join(unsupported)}; "
                "no jobs were dispatched. Use a bounded native run there and collect its receipts."
            )
        with progress(f"collecting {target}"):
            probe = ["run", "--", "python", "-m", "mainboard.jobs.lanes", "collect", target]
            cells = lanes_module.parsed(localhost[project.name][probe]())
        if not cells:
            raise MissionError(f"{target} collected no cells")
        groups = lanes_module.grouped(cells, by=group, per_job=per_job)
        served = node or lanes_module.node_of(target)
        fresh = ["--fresh", "--timeout", str(timeout)]
        pytest_args = ["-p", "no:randomly", "-q", "--no-header", *(["--rerun"] if rerun else [])]
        shown = Output(agent=agent)
        shown.print_rows(lanes_module.summary(hosts, groups), title="lanes")
        if dry_run:
            return 0
        if not yes and sys.stdin.isatty() and not _agreed():
            raise SystemExit(1)
        dispatched: list[tuple[str, str, str]] = []
        exit_code = 0
        for host in hosts:
            for chosen in groups:
                line = [target, "--", *fresh, *chosen.ids, "--", *pytest_args]
                if host == "local":
                    code = board("local").run(line)
                    exit_code = exit_code or code
                    dispatched.append((host, chosen.name, f"local exit {code}"))
                    continue
                with progress(f"submitting {chosen.name} on {host}") as stage:
                    job = board(host).submit(
                        joined(line),
                        watch=stage,
                        name=f"lanes-{host}-{chosen.name}",
                        queue=queue,
                        walltime=walltime,
                        mem_gb=mem_gb,
                        gpus=gpus,
                        gpu_name=gpu_name,
                        max_usd=max_usd,
                        node=served,
                    )
                dispatched.append((host, chosen.name, job.handle.id))
        shown.print_rows(
            [{"host": h, "group": g, "handle": i} for h, g, i in dispatched], title="dispatched"
        )
        if not wait:
            return exit_code
        for host, name, identity in dispatched:
            if identity.startswith("local exit"):
                continue
            with progress(f"waiting on {identity} ({host}, {name})"):
                settled = board("local").verdicts().wait(identity, host=host, say=_said)
            exit_code = exit_code or settled.code
        return exit_code

    @batch.command(name="prepare")
    def batch_prepare(
        spec: str = "", *, declared: Declared = _SPEC_ONLY, output: Output = _RICH
    ) -> None:
        """Measure what each job must still put on its target, and record the measurement.

        The mirror a host already carries is not shipped again, so what a job actually sends is
        the workspace's changes since that mirror plus whatever data the job itself names. Both
        sizes are reported, on disk and compressed, because compressed is what crosses the wire.
        Nothing is dispatched.

        spec: the batch spec file, relative to the workspace root.
        fields: a comma-separated projection over the transfer columns.
        """
        batched = declared.batch(board("local"), spec)
        with progress(f"measuring {batched.id}"):
            measured = [transfer.model_dump() for transfer in batched.prepare()]
        _tabled(
            measured,
            _TRANSFER_COLUMNS,
            summing=("files", "raw_bytes", "wire_bytes"),
            output=output,
            title=f"prepare: {batched.id}",
        )

    @batch.command(name="estimate")
    def batch_estimate(
        spec: str = "", *, declared: Declared = _SPEC_ONLY, output: Output = _RICH
    ) -> None:
        """Price every job of a batch before any of it runs, one row each and a total.

        What each job ships, what hardware it lands on, how long that target has actually taken
        to start work, and what the meter says about that. The setup times are fitted from this
        workspace's own recorded dispatches, so a target nobody has measured is priced with a
        deliberately pessimistic assumption and says so in its sample count. Nothing is
        dispatched, nothing is rented, and no target is even contacted.

        spec: the batch spec file, relative to the workspace root.
        fields: a comma-separated projection over the estimate columns.
        """
        batched = declared.batch(board("local"), spec)
        with progress(f"pricing {batched.id}"):
            priced = [row.model_dump() for row in batched.estimate().jobs]
        _tabled(
            priced,
            _ESTIMATE_COLUMNS,
            summing=("wire_bytes", "runtime_s", "expected_usd", "p90_usd"),
            output=output,
            title=f"estimate: {batched.id}",
        )

    @batch.command(name="run")
    def batch_run(
        spec: str = "", *, declared: Declared = _SPEC_ONLY, output: Output = _RICH
    ) -> None:
        """Dispatch every job of a batch to its own target, printing the batch id and each handle.

        One target refusing is that job's row and the rest still go, since a batch spread over a
        fleet routinely meets one machine that is asleep or was never declared. Watch the batch
        by the id printed here.

        A target that refuses on its own count quota is the one refusal that is not final: the
        row says `held`, the request stays in the run registry, and the durable sweep offers it
        again every pass until the queue has room, so a wave is never quietly shorter than the
        plan.

        `--only` dispatches part of the plan, which is what a plan worked through in waves needs:
        the nine jobs whose data is ready go now, and the four that are not are recorded as
        skipped so neither `batch watch` nor `monitor` ever waits for them. The batch keeps its
        identity, so tomorrow's wave writes to the same receipts stream.

        spec: the batch spec file, relative to the workspace root.
        fields: a comma-separated projection over job/target/state/handle/kind/reason.
        """
        batched = declared.batch(board("local"), spec)
        with progress(f"dispatching {batched.id}") as stage:
            dispatched = batched.run(watch=stage)
        if output.mode is None:
            print(batched.id)
        output.print_rows(
            [entry.model_dump() for entry in dispatched],
            title=f"run: {batched.id}",
            columns=_DISPATCH_COLUMNS,
        )

    @batch.command(name="watch")
    def batch_watch(batch_id: str, *, interval: float = 0.0, output: Output = _RICH) -> None:
        """Show every job of a dispatched batch, on every target, as the durable sweep settles it.

        Each pass runs the same sweep a cron runs, so results are pulled back and provider
        rentals are cancelled whether or not anyone is watching, and every change becomes a line
        in the batch's own receipts. One pass and exit by default.

        batch_id: the batch to watch, as `run` printed it.
        interval: seconds between passes, following until every job settles; one pass when 0.
        fields: a comma-separated projection over job/target/handle/state/verdict/detail.
        """
        watcher = board("local").watch(batch_id)
        label = f"sweeping {batch_id}"
        show = partial(_status, output=output)
        if not interval:
            with progress(label):
                status = watcher.once()
            show(status)
            return
        _followed(watcher.follow(interval), label, show)

    @batch.command(name="wait")
    def batch_wait(
        batch_id: str,
        *,
        timeout: float = vocabulary.WAIT_SECONDS,
        interval: float = 0.0,
        stall: float = STALL_SECONDS,
        output: Output = _RICH,
    ) -> int:
        """Block until every job of a batch settles, print the batch's verdict, exit its code.

        The same durable sweep `wait` runs on one handle, over the whole batch: results are
        pulled back and rentals released as each job lands, cells and a heartbeat stream to
        stderr, and the answer is read off the batch's receipts, 0 when every job settled
        clean, 1 on any failure, 2 at the timeout with work still in flight, 4 when a job
        stalled.

        batch_id: the batch to wait on, as `run` printed it.
        timeout: give up after this many seconds, exiting 2 with jobs still in flight, an hour
            unless said otherwise; 0 waits as long as it takes.
        interval: seconds between sweeps, the dispatch default when 0.
        stall: seconds a running job may print nothing on an idle card before the wait stops
            and exits 4; 0 never calls a job stalled.
        fields: a comma-separated projection over the verdict columns.
        """
        return wait(batch_id, timeout=timeout, interval=interval, stall=stall, output=output)

    @app.command(show=False)
    def provide(env: str = "", *, source: str = "", expect: str = "", json: bool = False) -> None:
        """Build the immutable environment a dispatched job activates, and print where it is.

        The verb a host runs for itself, and the one a dispatch runs on it after pinning a
        source tree. An environment is addressed by the content of the compiled manifest and
        the lock beside it, so a directory built for one lock is never written to again and a
        wave queued against it keeps it however often the workspace re-solves meanwhile.

        Building one that already exists touches nothing and prints the same path, which is
        what lets every job of a wave ask and one of them build.

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

    @app.command(name="job", show=False)
    def job_(record: str) -> int:
        """Run a dispatched job from its record, which every generated job script hands over.

        The job's command, the tree it runs from, the environment it enters, what it exports and
        how long it may take were all decided where it was dispatched and written down as one
        record. This carries the record out the same way on every host: build and enter the
        environment, run the command under its walltime, frame its receipts back and answer its
        exit status.

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
        workspace's own file first and reaches whatever `[tracking]` declared second.

        A dispatched job starts this for itself, so this verb is here for a command somebody
        runs by hand and for the job scripts that already call it.

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

    @app.command
    def wait(
        handle: str,
        *,
        on: str = "",
        timeout: float = vocabulary.WAIT_SECONDS,
        interval: float = 0.0,
        stall: float = STALL_SECONDS,
        output: Output = _RICH,
    ) -> int:
        """Block until a dispatched job settles, print its receipts-derived outcome, exit its code.

        Every poll is the same durable pass `monitor` runs, so waiting here pulls results back,
        cancels rentals and writes receipts exactly as the cron would, and a wait killed halfway
        loses nothing. What prints at the end is read back off the on-disk receipts rather than
        remembered from the loop, which is what makes this the sanctioned completion check.

        While it blocks, stderr carries each test cell's outcome as it lands and a heartbeat:
        cells done, failures, how long since the output last grew, and the busiest card where
        that is cheap to read. A job whose pytest session ended while its process lingers is
        settled on the session's own outcome, and a running job silent past `--stall` on an idle
        card ends the wait with exit 4 rather than holding it to the timeout.

        handle: the job to wait on, as `submit` printed it or by the name `jobs` prints, or a
            batch id as `batch run` printed it, which waits for every job of the batch.
        on: the host alias narrowing a handle recorded on several hosts.
        timeout: give up after this many seconds, exiting 2 with the job still in flight, an
            hour unless said otherwise; 0 waits as long as it takes.
        interval: seconds between polls, the dispatch default when 0.
        stall: seconds a running job may print nothing on an idle card before the wait stops
            and exits 4; 0 never calls a job stalled.
        fields: a comma-separated projection over the verdict columns.
        """
        print(f"waiting on {handle}", file=sys.stderr, flush=True)
        with diverted():
            settled = (
                board("local")
                .verdicts()
                .wait(
                    handle,
                    host=on,
                    timeout=timeout,
                    interval=interval or vocabulary.POLL_SECONDS,
                    stall=stall,
                    say=_said,
                )
            )
        return _settled(settled, output)

    @app.command
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

    @app.command
    def cancel(handle: str, *, on: str = "", output: Output = _RICH) -> int:
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

        handle: the job to cancel, as `submit` printed it or by the name `jobs` prints.
        on: the host alias narrowing a handle recorded on several hosts.
        fields: a comma-separated projection over the verdict columns.
        """
        with progress(f"cancelling {handle}"):
            settled = board("local").verdicts().cancel(handle, host=on)
        return _settled(settled, output)

    @app.command
    def verdict(target: str, *, on: str = "", run: str = "", output: Output = _RICH) -> int:
        """Print the settled truth the on-disk receipts hold, and exit with what it adds up to.

        The anti-fabrication verb. Dashboards, notification digests and progress summaries are
        sinks; this reads only the receipts they point at, one row per trial with its outcome,
        its gate sweep and the ledger node it serves, and never a scheduler, a service or a
        memory of the session that dispatched. The exit status is the completion check: 0 when
        every row settled clean, 1 on any failure, 2 while anything is still in flight, 3 when
        the receipts prove nothing.

        A receipts STORE is scored one run at a time, its newest by default, because a store
        holds every run a harness ever took and reading them as one stream lets a failure from
        months ago condemn a clean re-run today.

        target: a receipts store directory, a stream id, a receipts file, or a dispatched handle
            or job name.
        on: the host alias narrowing a handle recorded on several hosts.
        run: which run of a receipts store to score, its newest when unset.
        fields: a comma-separated projection over the verdict columns.
        """
        with progress(f"reading {target}"):
            settled = board("local").verdicts().of(target, host=on, run=run)
        return _settled(settled, output)

    @app.command
    def jobs(*, limit: int = 20, output: Output = _RICH) -> None:
        """List every dispatched job still in flight, then the most recently settled ones.

        A live job is never left out and never answered from memory. Each host is asked once
        about every run it still owes an answer on, one `qstat`, one `squeue`, one `pueue
        status`, so a wave of thirty five says which of them are running and which are queued
        behind them, since when, and where the scheduler estimates a start. A running job also
        shows its test cells landed out of its total, the seconds since its output last grew, and
        the busiest card on its host where that is one cheap command away. The limit bounds only
        the settled tail, and a listing that had to leave anything out says so on stderr rather
        than stopping quietly at twenty rows.

        limit: how many settled runs to show behind the live ones, newest first.
        fields: a comma-separated projection over the row's columns, cells/quiet_s/gpu_pct
            among them.
        """
        with progress("asking every host about its live jobs"):
            listed = Listing(board("local"), limit=limit).taken()
        output.print_rows(
            [row.model_dump() for row in listed.rows], title="jobs", columns=_JOB_COLUMNS
        )
        if listed.note:
            print(listed.note, file=sys.stderr)

    proc = App(
        name="proc",
        help="Kill a process tree, bound a command, wait for a file or port, on every system.",
    )
    app.command(proc)

    @proc.command(name="kill")
    def proc_kill(pids: list[int], *, force: bool = False) -> int:
        """Stop each process and everything it started, children first, on any system.

        What `pkill -P`, `kill -- -pgid` and `taskkill /T` each do on one system. Exits 1 when a
        process was already gone, naming it.

        force: kill at once instead of asking each process to terminate first.
        """
        gone = Processes().kill(pids, force=force)
        for pid in gone:
            print(f"no process {pid}", file=sys.stderr)
        return 1 if gone else 0

    @proc.command(name="timeout", version_flags=[])
    def proc_timeout(seconds: float, *command: str) -> int:
        """Run a command with a hard limit, stopping its whole tree when the limit passes.

        The portable `timeout`: the command runs with this terminal's stdio and without a shell,
        and exits with its own status, or 124 when it had to be stopped, as GNU `timeout` does.
        A tree that ignores the request to stop is killed after a short grace.

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

        port: a `host:port` that must accept a TCP connection.
        pid: a process that must have exited.
        timeout: seconds to wait at most, 0 for as long as it takes.
        """
        return 0 if Processes().wait(file=file, port=port, pid=pid, seconds=timeout) else 1

    center = App(
        name="center",
        help="Manage the monorepo from the one machine that holds it; targets never need these.",
    )
    app.command(center)

    @center.command
    def verify(*, output: Output = _RICH) -> int:
        """Say whether this machine is ready to be the center, and exit 1 when it is not.

        Every question at once, each row with the one command that repairs it: this machine's
        git tooling (git, git-lfs and its filters, a credential helper, and on Windows symlinks
        and long paths, the safe settings applied in place), the machine judged against the
        workspace the way `facts` judges any host, the `doctor` report, the plan `check`
        resolves here, a smoke run of Python, torch and CUDA in the default environment,
        whether every lint tool can start, the repository tree, every agent's configuration
        (AGENTS.md, CLAUDE.md, the `.claude` and `.codex` links, `.mcp.json`, `opencode.json`),
        the default environment put on the PATH every agent shell starts from and proven from
        each shell kind, and the tracked scripts that would behave differently here, each named
        with its portable replacement.

        fields: a comma-separated projection over section/verdict/detail/fix.
        """
        with progress("verifying this center"):
            sections = Verification(board("local")).sections()
        return _sectioned(sections, output, title="verify")

    @center.command
    def migrate(destination: str, *, root: str = "", output: Output = _RICH) -> int:
        """Move the center to another machine ssh reaches, Windows, macOS or Linux.

        Probes the destination (operating system, shells, filesystem, links, long paths, disk,
        git, git-lfs, gh, pixi, the NVIDIA driver and its CUDA) and stops early on a platform
        the workspace or its lock cannot serve. Then signs gh in with this machine's login,
        carries the ssh config blocks and keys the host profiles use, clones the monorepo at
        this HEAD with every owned submodule at its recorded pointer, carries what git does not
        hold (the `.env`, the `.mainboard/` registry and ledgers and every environment's lock,
        Claude Code's memory re-keyed to the new workspace path and its project settings,
        Codex's and opencode's config, credentials and memories), installs this tool and the
        default environment from the lock this center solved, and ends with the destination
        running `center verify` on itself. Secrets ride ssh's stdin only and are never printed.

        Every step converges on what is already there, so running it again after an
        interruption continues, and running it after it finished re-verifies and changes
        nothing. Exits 1 when any row fails.

        destination: the ssh alias of the machine becoming the center.
        root: where the workspace goes there, `~/projects` when omitted; an existing directory
            is used only when it is empty or already this repository.
        fields: a comma-separated projection over section/verdict/detail/fix.
        """
        with progress(f"moving the center to {destination}") as stage:
            sections = Migration(board("local"), destination, root=root, watch=stage).run()
        return _sectioned(sections, output, title="migrate")

    @center.command
    def paper(
        name: str,
        *,
        show: tuple[str, ...] = (),
        dpi: int = 110,
        json: bool = False,
        agent: bool = False,
    ) -> int:
        """Build a declared manuscript and report everything wrong with it, exiting 1 on any.

        Built with tectonic in the workspace environment, then read back: errors, undefined
        references and citations, multiply defined labels and overfull boxes with the file and
        line each comes from, the page count, the page every section starts on, and whether
        the section `[papers.<name>] ends` names ends by page `limit`.

        name: the `[papers.<name>]` manuscript.
        show: a phrase from the manuscript, repeatable; the page it appears on is rendered to a
            PNG beside the build and its path printed.
        dpi: the resolution a shown page renders at.
        json: print the whole report as canonical JSON instead of the default rich tables.
        agent: print the compact tabular mode instead of the default rich tables.
        """
        manuscript = board("local").paper(name)
        with progress(f"building {name}"):
            report = manuscript.check()
        _report(report, mode=Output(json=json, agent=agent).mode)
        for phrase in show:
            print(manuscript.show(phrase, dpi=dpi).as_posix())
        return 1 if report.problems else 0

    @center.command
    def members(*names: str, output: Output = _RICH) -> int:
        """Check that every member works for somebody who clones it alone; exit 1 on a failure.

        A member is a project `[workspace] members` composes into this workspace. Each one is
        read for what ties it to the monorepo: no installable `pyproject.toml`, no repository
        of its own, a path in its own files climbing out of it, a task depending on one it does
        not declare, and an import only the monorepo satisfies, from a sibling member it does
        not require, a directory on the root's `PYTHONPATH` or, under a `src/` layout, beside
        the package it installs, each a `fail`. Root-only
        settings it declares, and root tasks, papers or variables reaching into it, are a
        `warn`. Then it is cloned alone into a temporary directory, installed by `uv` into an
        empty environment and every package it installs imported, the one `pass`.

        names: the members to check, by name or path; every member when omitted.
        fields: a comma-separated projection over section/verdict/detail/fix.
        """
        standalone = Standalone(composition(workspace_root() / project.manifest))
        with progress("checking the members alone"):
            sections = standalone.sections(names)
        return _sectioned(sections, output, title="members")

    git = App(
        name="git",
        help="Operate the workspace repository and its owned submodules as one tree.",
    )
    center.command(git)

    @git.command(name="status")
    def git_status(*, output: Output = _RICH) -> None:
        """Show every owned repository in the tree on one table, without touching the network.

        Owned means the owner in the remote URL is the workspace root's own or one `[git]
        owners` names; reference code pinned from anybody else is left out. Each row says the
        branch (or `detached`), how far HEAD is ahead of and behind its upstream as last
        fetched, how many paths are changed and untracked, and which remote branch already
        holds HEAD, empty for a commit a parent pointer could not yet be cloned at.

        fields: a comma-separated projection over the status columns.
        """
        with progress("reading the repository tree"):
            states = board("local").git().status()
        output.print_rows(
            [state.model_dump() for state in states],
            title="git status",
            columns=_GIT_STATUS_COLUMNS,
        )

    @git.command(name="pull")
    def git_pull(*, output: Output = _RICH) -> int:
        """Fast-forward every owned repository and bring submodule checkouts along, root first.

        Every owned remote is fetched at once, then the tree is walked from the root down.
        Nothing is merged or rebased: a diverged branch is held and named, and a fast-forward
        that would overwrite local changes is refused by git itself. A detached HEAD is put back
        on its trunk where that moves no commit. A submodule follows its parent's new pointer
        only when it sat on the old one, and one never checked out is cloned at the recorded
        pointer. Exits 1 when any repository was held or failed.

        fields: a comma-separated projection over repo/outcome/detail.
        """
        with progress("pulling the repository tree"):
            steps = board("local").git().pull()
        return _stepped(steps, output, title="git pull")

    @git.command(name="commit")
    def git_commit(
        *, message: Annotated[str, Parameter(name=["--message", "-m"])], output: Output = _RICH
    ) -> int:
        """Commit every dirty owned repository, submodules first, then the pointers to them.

        Each commit lands on a branch: a detached HEAD is attached to its trunk when that is a
        fast-forward of the branch, and held otherwise, as is a repository behind its upstream
        and a parent whose submodule did not commit. Anything under a `[git] never-commit`
        pattern and files over the size ceiling that Git LFS does not carry stay out of the
        commit, unstaged; the row names the oversized ones and any never-commit path that was
        staged by hand. Exits 1 when any repository was held or failed.

        message: the commit message, the same for every repository committed.
        fields: a comma-separated projection over repo/outcome/detail.
        """
        with progress("committing the repository tree"):
            steps = board("local").git().commit(message)
        return _stepped(steps, output, title="git commit")

    @git.command(name="push")
    def git_push(*, output: Output = _RICH) -> int:
        """Push every owned repository, submodules before the parents that point at them.

        A parent is pushed only once every submodule pointer its HEAD records is held by a
        branch of that submodule's remote. Git LFS objects are uploaded first. A remote that
        protects the tracked branch gets the commit on a branch named `<tool>/<branch>` after
        this tool instead, and the row asks for the pull request. HTTPS pushes to GitHub can
        use the `gh` login as a credential. Exits 1 when any repository was held or failed.

        fields: a comma-separated projection over repo/outcome/detail.
        """
        with progress("pushing the repository tree"):
            steps = board("local").git().push()
        return _stepped(steps, output, title="git push")

    @git.command(name="check")
    def git_check(*, output: Output = _RICH) -> int:
        """Verify the tree is safe to clone and push, and exit 1 when anything fails.

        Fetches every owned repository and the foreign submodules they point at, then reports
        each pointer no branch of its remote holds, each diverged branch, each file in HEAD over
        the size ceiling and each LFS repository with no git-lfs here as `fail`, and a detached
        HEAD, unpushed or missing commits, and a checkout off its recorded pointer as `warn`.
        An empty table is a consistent tree.

        fields: a comma-separated projection over repo/check/verdict/detail.
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
_TRANSFER_COLUMNS = ("job", "target", "files", "raw_bytes", "wire_bytes", "since")
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
    handle or the document this verb prints once the dispatch has actually happened.
    """
    print("dispatch? [y/N] ", end="", file=sys.stderr, flush=True)
    return input().strip().lower() in {"y", "yes"}


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


def _sectioned(sections: list[Section], output: Output, *, title: str) -> int:
    """Print a report's rows and answer its exit status: 1 when any row failed."""
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


def _present(report: MonitorReport, output: Output) -> None:
    """Print one sweep's report, the whole document in the compact modes, else what moved.

    A cron reads the full report, counts and `changed` flag included, and branches on it; a
    person at a terminal wants the jobs that actually settled this pass, one row each, with the
    still-running count in the heading and the columns named even when nothing moved.
    """
    if output.mode is not None:
        output.print_record({**report.model_dump(), "changed": report.changed}, title="monitor")
        return
    output.print_rows(
        _changes(report),
        title="monitor: sweep skipped; another monitor owns settlement"
        if report.running is None
        else f"monitor: {report.running} running",
        columns=_CHANGE_COLUMNS,
    )


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


def main() -> None:
    """Console entry point, `MissionError` printed to stderr without a traceback, exit 1.

    The snapshot is brought up to its source first, which re-executes this same command on the
    new code when the source moved, and says so on stderr only. A trailing-command verb then gets
    the `--` its command implies, so nothing typed after the command is ever read as this tool's.
    """
    install_traceback()
    staleness.current()
    app = build()
    try:
        app(Delimiter(app).placed(sys.argv[1:]))
    except MissionError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None
