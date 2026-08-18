import shlex
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

from cyclopts import App, Parameter

from .board import Board
from .context.resolver import Resolver
from .core.errors import MissionError
from .core.project import Project
from .manifest.loading import load
from .render import install_traceback, mode_of, progress, record, rows

if TYPE_CHECKING:
    from .dispatch.state import MonitorReport


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

    # Everything after `--` is another program's argv and must reach it untouched. cyclopts
    # honours the `--` delimiter for its own help flags but not for its version flag, so the two
    # passthrough verbs give up `--version` entirely (the root app still answers it) rather than
    # answering `run -- python --version` with this tool's version.
    @app.command(version_flags=[])
    def run(
        *command: Annotated[str, Parameter(allow_leading_hyphen=True)],
        on: str = "local",
        env: str = "",
        container: str = "",
    ) -> int:
        """Run a command through the host's activated plan, exiting with its code.

        command: the command tokens, everything after `--`, its own flags included.
        on: the host alias the command runs on, `local` for this machine.
        env: an environment name overriding the profile's choice.
        container: a container override, `none` forcing bare.
        """
        return board(on).run(shlex.join(command), env=env, container=container)

    @app.command(version_flags=[])
    def submit(
        *command: Annotated[str, Parameter(allow_leading_hyphen=True)],
        on: str,
        name: str = "",
        queue: str = "",
        walltime: str = "",
        mem_gb: int = 0,
        gpus: int = 0,
        attempt: int = 1,
        fetch: str = "",
        env: str = "",
        container: str = "",
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Dispatch a command as a job on a host, printing its handle.

        command: the command tokens, everything after `--`.
        on: the host alias the job targets.
        attempt: the 1-based try number feeding expression defaults.
        fetch: a results path recorded for later `pull`.
        json: print the handle as canonical JSON instead of the bare id.
        agent: print the handle in the compact tabular mode instead of the bare id.
        fields: a comma-separated projection over the handle's fields.
        """
        with progress(f"submitting on {on}"):
            job = board(on).submit(
                shlex.join(command),
                name=name,
                queue=queue,
                walltime=walltime,
                mem_gb=mem_gb,
                gpus=gpus,
                attempt=attempt,
                fetch=fetch or None,
                env=env,
                container=container,
            )
        if not json and not agent:
            print(job.handle.id)
            return
        record(
            job.handle.model_dump(),
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="handle",
        )

    @app.command
    def install(
        env: str = "default", *, on: str = "local", resolve: bool = False, profile: str = ""
    ) -> None:
        """Compile the manifest and install the environment, here or on a host.

        Targeting a host alias runs the whole onboarding there: mirror the workspace, install
        the tool from that mirror, provision the environment, and probe what the host became.

        env: the environment name, `default` for the root surface.
        on: the host alias to install on, `local` for this machine.
        resolve: allow a fresh dependency solve when the lock is stale.
        profile: the declared host profile describing this machine, so the generated activation
            carries that host's modules; used when a host installs its own environment.
        """
        with progress(f"installing {env} on {on}") as stage:
            board(on).install(env, resolve=resolve, profile=profile, watch=stage)

    @app.command
    def setup(
        host: str,
        *,
        env: str = "default",
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Onboard a host until it can run jobs, then show what it became.

        host: the host alias to set up.
        env: the environment name to provision there.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the setup record's fields.
        """
        with progress(f"setting up {host}") as stage:
            report = board(host).install(env, watch=stage)
        record(
            report.model_dump(),
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="setup",
        )

    @app.command
    def hosts(*, json: bool = False, agent: bool = False, fields: str = "") -> None:
        """List the hosts already set up, newest first, from the dispatch state.

        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over host/root/env/installer/tool/onboarded_at.
        """
        payloads = [
            {
                "host": setup.host,
                "root": setup.root,
                "env": setup.env,
                "installer": setup.installer,
                "tool": setup.tool,
                "onboarded_at": setup.onboarded_at,
            }
            for setup in board("local").dispatcher.cache.hosts()
        ]
        rows(
            payloads,
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="hosts",
        )

    @app.command
    def monitor(
        *, watch: float = 0.0, json: bool = False, agent: bool = False, fields: str = ""
    ) -> None:
        """Settle every dispatched job that ended since the last pass, then exit.

        The durable sweep a periodic cron runs. It resolves every job the dispatch cache still
        owes an outcome for, pulls back the results of the ones that just finished, records their
        verdicts in the study ledgers that own them, and reports only what changed, so a second
        pass with nothing new says exactly that. A host that cannot be reached is reported with
        why and its jobs are left for the next pass, so no outcome ever depends on the process
        that dispatched the job still being alive.

        watch: seconds between repeated passes in the foreground, one pass and exit when 0.
        json: print the whole report as canonical JSON instead of the default rich table.
        agent: print the whole report in the compact tabular mode instead of the rich table.
        fields: a comma-separated projection over the report's fields.
        """
        sweep = board("local").monitor()
        mode = mode_of(json_mode=json, agent=agent)
        chosen = _fields(fields)
        if not watch:
            with progress("sweeping dispatched jobs"):
                report = sweep.once()
            _present(report, mode=mode, fields=chosen)
            return
        with suppress(KeyboardInterrupt):
            for report in sweep.watch(watch):
                _present(report, mode=mode, fields=chosen)

    @app.command
    def facts(
        on: str = "local", *, json: bool = False, agent: bool = False, fields: str = ""
    ) -> None:
        """Show the host's probed hardware facts.

        on: the host alias to probe, `local` for this machine.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the fact fields.
        """
        with progress(f"probing {on}"):
            payload = board(on).facts().model_dump()
        record(
            payload,
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="facts",
        )

    @app.command
    def plan(
        host: str = "local",
        *,
        env: str = "",
        container: str = "",
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Show the resolved execution plan for a host.

        host: the host alias, `local` for this machine.
        env: an environment name overriding the profile's choice.
        container: a container name overriding the profile's, `none` for bare.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the plan fields.
        """
        base = workspace_root()
        manifest = load(base / project.manifest)
        resolved = Resolver(manifest).plan(host, env=env, container=container)
        record(
            resolved.model_dump(),
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="plan",
        )

    @app.command
    def check(*, json: bool = False, agent: bool = False, fields: str = "") -> None:
        """Validate the workspace manifest, showing what it declares.

        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the declared fields.
        """
        base = workspace_root()
        manifest = load(base / project.manifest)
        payload = {
            "workspace": manifest.workspace.name,
            "environments": sorted(manifest.envs),
            "containers": sorted(manifest.containers),
            "hosts": sorted(manifest.profiles()),
            "tasks": sorted(manifest.tasks),
        }
        record(
            payload,
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="check",
        )

    @app.command
    def jobs(
        *, limit: int = 20, json: bool = False, agent: bool = False, fields: str = ""
    ) -> None:
        """List recently dispatched jobs from the shared dispatch cache.

        limit: how many recent runs to show, newest first.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over state/host/name/handle/submitted_at.
        """
        recent = board("local").dispatcher.cache.recent(limit)
        payloads = [
            {
                "state": run.state,
                "host": run.target,
                "name": run.name,
                "handle": run.handle,
                "submitted_at": run.submitted_at,
            }
            for run in recent
        ]
        rows(
            payloads,
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="jobs",
        )

    return app


# The columns a sweep's change table always carries, so an empty pass still renders its heading.
_CHANGE_COLUMNS = ("host", "handle", "outcome", "detail")


def _exit_on_mission_error(error: MissionError) -> NoReturn:
    """Print `error` to stderr without a traceback, then exit 1."""
    print(error, file=sys.stderr)
    raise SystemExit(1) from None


def _fields(raw: str) -> tuple[str, ...]:
    """`raw`'s comma-separated field names, trimmed and blank entries dropped."""
    return tuple(part.strip() for part in raw.split(",") if part.strip()) if raw else ()


def _present(report: MonitorReport, *, mode: str | None, fields: tuple[str, ...]) -> None:
    """Print one sweep's report, the whole document in the compact modes, else what moved.

    A cron reads the full report, counts and `changed` flag included, and branches on it; a
    person at a terminal wants the jobs that actually settled this pass, one row each, with the
    still-running count in the heading. The change table names its columns even when nothing
    moved, so a quiet pass still prints its heading instead of nothing at all.
    """
    if mode is not None:
        payload = {**report.model_dump(), "changed": report.changed}
        record(payload, mode=mode, fields=fields, title="monitor")
        return
    rows(
        _changes(report),
        mode=mode,
        fields=fields or _CHANGE_COLUMNS,
        title=f"monitor: {report.running} running",
    )


def _changes(report: MonitorReport) -> list[dict[str, str]]:
    """Every job that settled this pass and every host that could not be reached, one row each."""
    return [
        *(
            {
                "host": job.target,
                "handle": job.handle,
                "outcome": "ok",
                "detail": job.pulled_path or "",
            }
            for job in report.finished
        ),
        *(
            {"host": job.target, "handle": job.handle, "outcome": "failed", "detail": job.reason}
            for job in report.failed
        ),
        *(
            {"host": host.host, "handle": "", "outcome": "unreachable", "detail": host.reason}
            for host in report.unreachable_hosts
        ),
    ]


def main() -> None:
    """Console entry point, `MissionError` printed without a traceback."""
    install_traceback()
    try:
        build()(sys.argv[1:])
    except MissionError as error:
        _exit_on_mission_error(error)
