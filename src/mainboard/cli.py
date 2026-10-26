import os
import sys
from collections.abc import Callable
from contextlib import ExitStack
from io import TextIOWrapper
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, NoReturn

from cyclopts import App, Parameter
from rich.console import Console

from .apple import Apple
from .core.errors import MissionError, NoWorkspace
from .core.execution.modules import submodules
from .core.project import Project
from .dispatch.schedulers import HostUnreachable
from .log import configure, logger
from .profile import Feature, Profiler
from .verbs.apps.apple import AppleVerbs
from .verbs.apps.core import AppVerbs
from .verbs.delimiter import Delimiter
from .verbs.evidence.lake import LakeVerbs
from .verbs.evidence.papers import PaperVerbs
from .verbs.machines.hosts import HostVerbs
from .verbs.machines.procs import ProcVerbs
from .verbs.machines.rentals import RentalVerbs
from .verbs.session import Session
from .verbs.work.jobs import JobVerbs
from .verbs.work.running import RunningVerbs
from .verbs.work.submit import SubmitVerbs
from .verbs.workspace.agents import AgentVerbs
from .verbs.workspace.core import WorkspaceVerbs
from .verbs.workspace.environments import EnvironmentVerbs
from .verbs.workspace.git import GitVerbs
from .verbs.workspace.tool import ToolVerbs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .verbs.base import Verbs


# A verb class's method, as its class holds it: its names find the class's one instance and the
# bound twin cyclopts registers.
type _Method = Callable[..., int | None]


# The command groups under the top level, by the name each is typed as; `{tool}` is this tool.
_GROUPS = {
    "host": "Set up, list, inspect and reach the machines jobs run on.",
    "job": "Dispatch jobs to hosts and follow them until they settle.",
    "lake": "Keep the workspace's state lake and its evidence: check, compact, serve.",
    "paper": "Build a declared manuscript and plot figures from results.",
    "self": "Manage the {tool} installation itself.",
    "probe": "Inspect device code and preserve normalized analysis evidence.",
    "ci": "Run a package's CI gate, the very steps its GitHub workflow runs.",
    "agents": (
        "Configure every coding agent alike from `.agents`: files, links, servers, releases."
    ),
    "binary": "Install manifest-pinned executable ZIP dependencies.",
    "apple": "Inspect Xcode, simulators, build diagnostics and test results.",
    "app": "Build and test the apps a manifest declares (`[apps.<name>]`), from anywhere.",
    "proc": "Kill a process tree, bound a command, wait for a file or port, on every system.",
    "git": "Operate the workspace repository and its owned submodules as one tree.",
}

# Every verb: the group it is typed under (the top level when empty), and the name it is typed
# as when that is not the one cyclopts derives from its method.
_ROUTES: tuple[tuple[_Method, str, str], ...] = (
    (PaperVerbs.device, "probe", ""),
    (WorkspaceVerbs.help_, "", "help"),
    (RunningVerbs.run, "", ""),
    (SubmitVerbs.submit, "job", ""),
    (JobVerbs.show, "job", ""),
    (JobVerbs.jobs, "job", "list"),
    (JobVerbs.monitor, "job", ""),
    (JobVerbs.digest, "job", ""),
    (JobVerbs.events, "job", ""),
    (EnvironmentVerbs.add, "", ""),
    (EnvironmentVerbs.remove, "", ""),
    (EnvironmentVerbs.update_lock, "", "update"),
    (EnvironmentVerbs.upgrade, "", ""),
    (WorkspaceVerbs.new, "", ""),
    (WorkspaceVerbs.doctor, "", ""),
    (EnvironmentVerbs.install, "", ""),
    (EnvironmentVerbs.lock, "", ""),
    (ToolVerbs.update, "self", ""),
    (ToolVerbs.version, "self", ""),
    (WorkspaceVerbs.completion, "", ""),
    (HostVerbs.host_upgrade, "host", "upgrade"),
    (EnvironmentVerbs.pack_, "", "pack"),
    (EnvironmentVerbs.list_, "", "list"),
    (EnvironmentVerbs.tree, "", ""),
    (RunningVerbs.shell, "", ""),
    (HostVerbs.setup, "host", ""),
    (HostVerbs.sync, "host", ""),
    (RentalVerbs.hold, "host", ""),
    (RentalVerbs.offers, "host", ""),
    (RentalVerbs.release, "host", ""),
    (HostVerbs.serving, "host", "serve"),
    (HostVerbs.compute, "host", "list"),
    (JobVerbs.collect, "job", ""),
    (WorkspaceVerbs.query, "", ""),
    (WorkspaceVerbs.lint, "", ""),
    (ToolVerbs.ci_run, "ci", ""),
    (RunningVerbs.provide, "", ""),
    (RunningVerbs.job_, "", "execute"),
    (RunningVerbs.attest, "", ""),
    (RunningVerbs.sample, "", ""),
    (JobVerbs.logs, "job", ""),
    (JobVerbs.cancel, "job", ""),
    (AgentVerbs.agents_sync, "agents", "sync"),
    (AgentVerbs.agents_check, "agents", "check"),
    (AgentVerbs.agents_update, "agents", "update"),
    (AgentVerbs.binary_sync, "binary", "sync"),
    (AppleVerbs.apple_toolchain, "apple", "toolchain"),
    (AppleVerbs.apple_simulators, "apple", "simulators"),
    (AppleVerbs.apple_launch, "apple", "launch"),
    (AppleVerbs.apple_bundle, "apple", "bundle"),
    (AppleVerbs.apple_font_previews, "apple", "font-previews"),
    (AppleVerbs.apple_execute, "apple", "execute"),
    (AppVerbs.app_list, "app", "list"),
    (AppVerbs.app_test, "app", "test"),
    (AppVerbs.app_build, "app", "build"),
    (AppleVerbs.apple_diagnostics, "apple", "diagnostics"),
    (AppleVerbs.apple_results, "apple", "results"),
    (ProcVerbs.proc_list, "proc", "list"),
    (ProcVerbs.proc_kill, "proc", "kill"),
    (ProcVerbs.proc_timeout, "proc", "timeout"),
    (ProcVerbs.proc_wait, "proc", "wait"),
    (LakeVerbs.lake_check, "lake", "check"),
    (LakeVerbs.ingest, "lake", ""),
    (LakeVerbs.materialize, "lake", ""),
    (LakeVerbs.evict, "lake", ""),
    (LakeVerbs.dedup, "lake", ""),
    (LakeVerbs.replicate, "lake", ""),
    (LakeVerbs.compact, "lake", ""),
    (LakeVerbs.lake_upgrade, "lake", "upgrade"),
    (LakeVerbs.serve, "lake", ""),
    (PaperVerbs.paper_build, "paper", "build"),
    (GitVerbs.git_status, "git", "status"),
    (GitVerbs.git_pull, "git", "pull"),
    (GitVerbs.git_commit, "git", "commit"),
    (GitVerbs.git_push, "git", "push"),
    (GitVerbs.git_check, "git", "check"),
)

# The verbs a dispatched job or a keeper runs, never typed by a person, so help leaves them out,
# with the group each is typed under.
_HIDDEN = frozenset(
    {(JobVerbs.monitor, "job"), (HostVerbs.serving, "host"), (RunningVerbs.job_, "")}
    | {(RunningVerbs.provide, ""), (RunningVerbs.attest, ""), (RunningVerbs.sample, "")}
)

# The verbs a group runs when typed alone, rather than one of its commands.
_DEFAULTS = frozenset({ToolVerbs.ci_run})

# The verbs whose trailing command is passed on verbatim, so `--version` there is the command's.
_TRAILING = frozenset(
    {RunningVerbs.run, SubmitVerbs.submit, EnvironmentVerbs.list_, EnvironmentVerbs.tree}
    | {RunningVerbs.shell, ProcVerbs.proc_timeout}
)

# Entry points and the standard-library agents are left out of a profile: importing them runs
# them.
_UNPROFILED = ("__main__", "agent.program", "center.remote", ".profile")


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
    groups = {"": app} | {
        name: App(name=name, help=text.format(tool=project.name)) for name, text in _GROUPS.items()
    }
    for heading, group in groups.items():
        if heading:
            app.command(group)
    # Imported on first use: the engine brings DuckDB and the whole rule catalog.
    app.command("mainboard.analysis.cli:refactor", name="refactor")
    # Imported on first use too: the plotting stack takes over a second to import.
    groups["paper"].command("mainboard.verbs.evidence.plotting:plot", name="plot")
    # The cutok artifact's README runs `mainboard plot`; kept until that artifact is frozen.
    app.command("mainboard.verbs.evidence.plotting:plot", name="plot", show=False)
    verbs = {type(verb).__name__: verb for verb in _verbs(Session(root=root), app)}
    for method, under, name in _ROUTES:
        _route(groups, verbs, method, group=under, name=name)
    return app


def main() -> None:
    """Console entry point, `MissionError` printed to stderr without a traceback, exit 1.

    A trailing-command verb gets the `--` its command implies, so nothing typed after the
    command is ever read as this tool's. The verbs a person or an agent types log short plain
    lines; the ones a dispatched job runs keep the JSON its log readers parse.
    """
    _prepare()
    try:
        _run(build())
    except BrokenPipeError:
        _closed()
    except (MissionError, NoWorkspace, HostUnreachable) as error:
        _refused(error)


def _closed() -> NoReturn:
    """End quietly on a reader that went away, before Python's shutdown flush tries again."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        os.dup2(sink.fileno(), sys.stdout.fileno())
    raise SystemExit(0) from None


def _prepare() -> None:
    """Ready this process to print and to build.

    Streams never fail on a character, the Xcode in use is selected, and every verb but the one
    a dispatched job runs logs short plain lines.
    """
    # A job's log carries whatever it printed (a progress bar's block characters), which a pipe
    # in a narrower encoding cannot print; one such character killed `job logs` mid-print
    # (2026-09-29).
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, TextIOWrapper):
            stream.reconfigure(errors="replace")
    Apple.select_xcode()
    if not Project().variable("LOG_FORMAT").read() and sys.argv[1:2] != ["execute"]:
        configure(output="line")


def _profiled(report: Path, app: App, argv: Sequence[str]) -> None:
    """Run `argv` through `app` under this tool's own profiler, its span report in `report`.

    Every one of its modules is instrumented, so `MB_PROFILE=prof.txt mb job submit ...` names the
    slow function of any verb, on any machine, with nothing else installed.
    """
    package = __name__.partition(".")[0]
    modules = [
        name
        for name in submodules(package)
        if name != package and not any(skip in name for skip in _UNPROFILED)
    ]
    profiler = Profiler(features=Feature.SPANS, gpus=(), auto=modules)
    with ExitStack() as finish:
        finish.callback(_reported, profiler, report)
        with profiler:
            app(argv)


def _refused(error: Exception) -> NoReturn:
    """End on a refusal with its one sentence, never a traceback.

    A host that will not answer is an answer, not a crash: it used to end a verb as a traceback
    forty lines long with the one sentence that mattered last.
    """
    logger.error(str(error))
    raise SystemExit(1) from None


def _reported(profiler: Profiler, report: Path) -> None:
    """Write the profiler's span report to `report`, and say where."""
    report.write_text(profiler.report(), encoding="utf-8", newline="\n")
    logger.info(f"profile written to {report}")


def _route(
    groups: Mapping[str, App],
    verbs: Mapping[str, Verbs],
    method: _Method,
    *,
    group: str,
    name: str,
) -> None:
    """Register `method`, bound to its class's one instance, under `group` as `name`."""
    if not isinstance(method, FunctionType):
        raise TypeError(f"{method!r} is no verb method a route can bind")
    owner = verbs[method.__qualname__.partition(".")[0]]
    flags: dict[str, list[str]] = {"version_flags": []} if method in _TRAILING else {}
    bound = getattr(owner, method.__name__)
    if method in _DEFAULTS:
        groups[group].default(bound)
        return
    shown = (method, group) not in _HIDDEN
    groups[group].command(bound, name=name or None, show=shown, **flags)


def _run(app: App) -> None:
    """Run the verb the command line names, under the profiler when `MB_PROFILE` asks."""
    argv = Delimiter(app).placed(sys.argv[1:])
    if profiled := Project().variable("PROFILE").read():
        _profiled(Path(profiled), app, argv)
    else:
        app(argv)


def _verbs(session: Session, app: App) -> list[Verbs]:
    """One instance of every verb class, each bound to the one session of this run."""
    jobs = JobVerbs(session=session)
    apple = AppleVerbs(session=session)
    plain = (EnvironmentVerbs, RunningVerbs, HostVerbs, RentalVerbs, ToolVerbs, AgentVerbs)
    tools = (ProcVerbs, LakeVerbs, GitVerbs)
    return [
        jobs,
        apple,
        *(kind(session=session) for kind in (*plain, *tools)),
        PaperVerbs(session=session, jobs=jobs),
        SubmitVerbs(session=session, jobs=jobs),
        AppVerbs(session=session, apple=apple),
        WorkspaceVerbs(session=session, app=app),
    ]
