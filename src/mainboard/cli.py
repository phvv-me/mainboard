import shlex
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

from cyclopts import App, Parameter

from .batch.spec import BatchSpec
from .board import Board
from .context.resolver import Resolver
from .core.errors import MissionError
from .core.project import Project
from .doctor import Verdict
from .manifest.loading import load
from .render import install_traceback, mode_of, progress, record, rows, totals

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .batch.watch import BatchStatus
    from .deps import Change
    from .dispatch.state import MonitorReport
    from .render.values import Node


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
        gpu_name: str = "",
        max_usd: float = 0.0,
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
        gpu_name: the GPU type to rent, for a metered provider host.
        max_usd: the spend cap a provider host refuses to submit without.
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
                gpu_name=gpu_name,
                max_usd=max_usd,
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
    def add(
        spec: str,
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "conda",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Declare a dependency in the manifest and re-solve, showing what the lock did.

        A bare name is pinned to whatever the ecosystem's index publishes right now, and a name
        carrying its own constraint is written exactly as given. The table it lands in is the
        one the flags name, and where the manifest already writes that kind of requirement in a
        particular table, the edit joins it there.

        spec: the requirement, a bare name or a name with the constraint it carries.
        lang: the ecosystem whose resolver installs it, `conda` by default.
        env: an environment name, the workspace-wide table when omitted.
        dev: declare it as a development-only requirement.
        resolve: re-solve after the edit, `--no-resolve` to stage several edits and solve once.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over name/where/before/after.
        """
        with progress(f"adding {spec}"):
            changes = (
                board("local").deps().add(spec, ecosystem=lang, env=env, dev=dev, resolve=resolve)
            )
        _changed(changes, json_mode=json, agent=agent, fields=fields, title="add")

    @app.command
    def remove(
        name: str,
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "",
        env: str = "",
        dev: bool = False,
        resolve: bool = True,
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Drop a dependency from the manifest and re-solve, showing what the lock did.

        With no flags the whole manifest is searched, so dropping a requirement never asks
        which table it was written into. Flags narrow that search, which is also how a name
        declared in more than one table is told apart.

        name: the dependency to drop.
        lang: narrow the search to one ecosystem's tables.
        env: narrow the search to one environment's tables.
        dev: narrow the search to development-only tables.
        resolve: re-solve after the edit.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over name/where/before/after.
        """
        with progress(f"removing {name}"):
            changes = (
                board("local")
                .deps()
                .remove(name, ecosystem=lang, env=env, dev=dev, resolve=resolve)
            )
        _changed(changes, json_mode=json, agent=agent, fields=fields, title="remove")

    @app.command
    def upgrade(
        name: str = "",
        *,
        lang: Annotated[str, Parameter(name=["--lang", "-l"])] = "",
        env: str = "",
        dev: bool = False,
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Move one dependency to its newest release, or the whole lock forward in its bounds.

        Named, the constraint itself is rewritten to what the ecosystem publishes now, which is
        the only way past a ceiling the manifest declares. Unnamed, the manifest is untouched
        and the lock is re-solved against the indexes, moving every pin as far as the declared
        constraints already allow.

        name: the dependency to bump, every declared one inside its bounds when omitted.
        lang: narrow the search to one ecosystem's tables.
        env: narrow the search to one environment's tables.
        dev: narrow the search to development-only tables.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over name/where/before/after.
        """
        with progress(f"upgrading {name or 'the lock'}"):
            changes = board("local").deps().upgrade(name, ecosystem=lang, env=env, dev=dev)
        _changed(changes, json_mode=json, agent=agent, fields=fields, title="upgrade")

    @app.command
    def new(
        name: str,
        *,
        template: str = "",
        description: str = "",
        dest: str = "",
        answer: tuple[str, ...] = (),
        json: bool = False,
        agent: bool = False,
        fields: str = "",
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
        template: the template to render, a declared name or any location copier accepts.
        description: the one sentence the README and the task rows carry.
        dest: where to render it, under the template's own declared home when omitted.
        answer: a further `question=value` for the template, repeatable.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over project/path/tasks/paste/snippet.
        """
        mode = mode_of(json_mode=json, agent=agent)
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
        # The rows print whole and pasteable at a terminal, so repeating them wrapped inside a
        # table cell would only make them harder to copy back out. A compact mode keeps the
        # field, since a caller reading the record has nowhere else to get them.
        if mode is None and made.snippet:
            print(made.snippet)
            payload.pop("snippet")
        record(payload, mode=mode, fields=_fields(fields), title="new")

    @app.command
    def doctor(*, json: bool = False, agent: bool = False, fields: str = "") -> int:
        """Say whether this workspace is fit to work in, and exit nonzero when it is not.

        Four questions asked at once and bounded: does the manifest still say something
        coherent, is what is installed the environment it describes, what compute answers right
        now, and does the mathematics still hold. A section reports the one command that
        repairs it, and only a genuinely broken workspace fails, so a sleeping host or a
        provider nobody has a key for is a word rather than a nonzero exit.

        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over section/verdict/detail/fix.
        """
        with progress("examining the workspace"):
            sections = board("local").doctor().sections()
        rows(
            [section.model_dump() for section in sections],
            mode=mode_of(json_mode=json, agent=agent),
            fields=_fields(fields),
            title="doctor",
        )
        return 1 if any(section.verdict is Verdict.FAIL for section in sections) else 0

    @app.command
    def install(
        env: str = "", *, on: str = "local", resolve: bool = False, profile: str = ""
    ) -> None:
        """Compile the manifest and install the environment, here or on a host.

        Targeting a host alias runs the whole onboarding there: mirror the workspace, install
        the tool from that mirror, provision the environment, and probe what the host became.

        env: the environment name, the target's declared profile choice when omitted.
        on: the host alias to install on, `local` for this machine.
        resolve: allow a fresh dependency solve when the lock is stale.
        profile: the declared host profile describing this machine, so the generated activation
            carries that host's modules; used when a host installs its own environment.
        """
        with progress(f"installing {env} on {on}") as stage:
            board(on).install(env, resolve=resolve, profile=profile, watch=stage)

    @app.command
    def shell(env: str = "") -> NoReturn:
        """Open an interactive shell with this workspace's environment already activated.

        The daily way in, and the one verb that works from a terminal where nothing is
        activated yet. This process becomes the shell, so quitting it returns to the terminal
        that asked.

        env: the environment name, the profile's declared choice when omitted.
        """
        board("local").shell(env)

    @app.command(version_flags=[])
    def interact(
        *command: Annotated[str, Parameter(allow_leading_hyphen=True)],
        on: str,
        env: str = "",
        queue: str = "",
        walltime: str = "",
    ) -> NoReturn:
        """Open an interactive session on a host, inside its mirrored workspace.

        `shell` for a machine that is not this one. This process becomes the ssh, so quitting
        the session returns to the terminal that asked. A queued host is asked for an
        interactive allocation first, so the terminal lands on a compute node rather than on the
        login node the request was made from.

        command: a command to run instead of handing over the terminal, everything after `--`.
        on: the host alias the session opens on.
        env: an environment name overriding the profile's choice.
        queue: the queue the allocation targets, the profile's declared choice when omitted.
        walltime: the session's wall-clock limit, the profile's declared choice when omitted.
        """
        board(on).interact(*command, env=env, queue=queue, walltime=walltime)

    @app.command
    def setup(
        host: str,
        *,
        env: str = "",
        resolve: bool = False,
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Onboard a host until it can run jobs, then show what it became.

        The host is provisioned with the environment its declared profile names, so a host that
        runs `serving` is set up for serving without repeating the name here. The lock this
        workspace solved ships with the mirror and the host installs from it.

        host: the host alias to set up.
        env: an environment name overriding the host profile's own.
        resolve: let the host run its own dependency solve instead of installing the shipped
            lock, which puts that host's compiler in the resolution path.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the setup record's fields.
        """
        with progress(f"setting up {host}") as stage:
            report = board(host).install(env, resolve=resolve, watch=stage)
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
    def compute(*, json: bool = False, agent: bool = False, fields: str = "") -> None:
        """List every compute path this workspace can reach, with prices and credit where cheap.

        This machine first, then each declared host with whether it answers and whether it was
        ever set up, then each provider backend with whether its credentials are present here,
        what the account has left to spend, and a live rate where asking for one is cheap. Every
        probe is bounded and runs beside the others, so the whole fleet answers in the time the
        slowest one takes, and a host that is down or a provider with no key is a row rather than
        a failure. No credential is ever printed, only whether one was found.

        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over name/kind/access/detail/usd_hr/credit_usd.
        """
        mode = mode_of(json_mode=json, agent=agent)
        with progress("probing every compute path"):
            paths = board("local").compute().paths()
        rows(
            [path.model_dump() for path in paths],
            mode=mode,
            fields=_fields(fields),
            title="compute",
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

    batch = App(name="batch", help="Prepare, price, dispatch and watch many jobs as one flow.")
    app.command(batch)

    def declared(spec: str, job: tuple[str, ...], name: str) -> BatchSpec:
        """The batch the caller declared, a spec file or repeated `target:command` flags."""
        if spec:
            return BatchSpec.load(workspace_root() / spec)
        if not job:
            raise MissionError("declare a batch: a spec file, or --job target:command")
        return BatchSpec.inline(name or "batch", job)

    @batch.command(name="prepare")
    def batch_prepare(
        spec: str = "",
        *,
        job: tuple[str, ...] = (),
        name: str = "",
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Measure what each job must still put on its target, and record the measurement.

        The mirror a host already carries is not shipped again, so what a job actually sends is
        the workspace's changes since that mirror plus whatever data the job itself names. Both
        sizes are reported, on disk and compressed, because compressed is what crosses the wire.
        Nothing is dispatched.

        spec: the batch spec file, relative to the workspace root.
        job: a `target:command` job, repeatable, for a batch declared without a file.
        name: the batch's name when declared with `--job` rather than a file.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the transfer columns.
        """
        batched = board("local").batch(declared(spec, job, name))
        with progress(f"measuring {batched.id}"):
            measured = [transfer.model_dump() for transfer in batched.prepare()]
        _tabled(
            measured,
            _TRANSFER_COLUMNS,
            ("files", "raw_bytes", "wire_bytes"),
            json_mode=json,
            agent=agent,
            fields=fields,
            title=f"prepare: {batched.id}",
        )

    @batch.command(name="estimate")
    def batch_estimate(
        spec: str = "",
        *,
        job: tuple[str, ...] = (),
        name: str = "",
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Price every job of a batch before any of it runs, one row each and a total.

        What each job ships, what hardware it lands on, how long that target has actually taken
        to start work, and what the meter says about that. The setup times are fitted from this
        workspace's own recorded dispatches, so a target nobody has measured is priced with a
        deliberately pessimistic assumption and says so in its sample count. Nothing is
        dispatched, nothing is rented, and no target is even contacted.

        spec: the batch spec file, relative to the workspace root.
        job: a `target:command` job, repeatable, for a batch declared without a file.
        name: the batch's name when declared with `--job` rather than a file.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over the estimate columns.
        """
        batched = board("local").batch(declared(spec, job, name))
        with progress(f"pricing {batched.id}"):
            priced = [row.model_dump() for row in batched.estimate().jobs]
        _tabled(
            priced,
            _ESTIMATE_COLUMNS,
            ("wire_bytes", "runtime_s", "expected_usd", "p90_usd"),
            json_mode=json,
            agent=agent,
            fields=fields,
            title=f"estimate: {batched.id}",
        )

    @batch.command(name="run")
    def batch_run(
        spec: str = "",
        *,
        job: tuple[str, ...] = (),
        name: str = "",
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Dispatch every job of a batch to its own target, printing the batch id and each handle.

        One target refusing is that job's row and the rest still go, since a batch spread over a
        fleet routinely meets one machine that is asleep or was never declared. Watch the batch
        by the id printed here.

        spec: the batch spec file, relative to the workspace root.
        job: a `target:command` job, repeatable, for a batch declared without a file.
        name: the batch's name when declared with `--job` rather than a file.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over job/target/handle/kind/reason.
        """
        batched = board("local").batch(declared(spec, job, name))
        mode = mode_of(json_mode=json, agent=agent)
        with progress(f"dispatching {batched.id}"):
            dispatched = batched.run()
        if mode is None:
            print(batched.id)
        rows(
            [entry.model_dump() for entry in dispatched],
            mode=mode,
            fields=_fields(fields) or _DISPATCH_COLUMNS,
            title=f"run: {batched.id}",
        )

    @batch.command(name="watch")
    def batch_watch(
        batch_id: str,
        *,
        interval: float = 0.0,
        json: bool = False,
        agent: bool = False,
        fields: str = "",
    ) -> None:
        """Show every job of a dispatched batch, on every target, as the durable sweep settles it.

        Each pass runs the same sweep a cron runs, so results are pulled back and provider
        rentals are cancelled whether or not anyone is watching, and every change becomes a line
        in the batch's own receipts. One pass and exit by default.

        batch_id: the batch to watch, as `run` printed it.
        interval: seconds between passes, following until every job settles; one pass when 0.
        json: print canonical JSON instead of the default rich table.
        agent: print the compact tabular mode instead of the default rich table.
        fields: a comma-separated projection over job/target/handle/state/verdict/detail.
        """
        watcher = board("local").watch(batch_id)
        mode = mode_of(json_mode=json, agent=agent)
        chosen = _fields(fields) or _STATUS_COLUMNS
        if not interval:
            with progress(f"sweeping {batch_id}"):
                status = watcher.once()
            _status(status, mode=mode, fields=chosen)
            return
        with suppress(KeyboardInterrupt):
            for status in watcher.follow(interval):
                _status(status, mode=mode, fields=chosen)

    @app.command
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

        stream: the receipts stream the samples belong to, a batch id or a run's name.
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

# The columns each batch table carries, named here so an empty batch still renders its heading and
# so the totals row is summed over the same shape the rows are printed in.
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
    "expected_usd",
    "p90_usd",
)
_DISPATCH_COLUMNS = ("job", "target", "handle", "kind", "reason")
_STATUS_COLUMNS = ("job", "target", "handle", "state", "verdict", "detail")


def _exit_on_mission_error(error: MissionError) -> NoReturn:
    """Print `error` to stderr without a traceback, then exit 1."""
    print(error, file=sys.stderr)
    raise SystemExit(1) from None


def _changed(
    changes: Sequence[Change], *, json_mode: bool, agent: bool, fields: str, title: str
) -> None:
    """Print one edit's constraint move and every pin its solve moved, as one table.

    Both are the same fact, something moved from one version to another somewhere, so they
    render as one shape rather than two tables a reader has to align by eye. Where the move
    happened is the column that tells them apart, a manifest table for the requirement and the
    lock for everything the solve dragged with it.
    """
    rows(
        [change.model_dump() for change in changes],
        mode=mode_of(json_mode=json_mode, agent=agent),
        fields=_fields(fields),
        title=title,
    )


def _tabled(
    payloads: list[dict[str, Node]],
    columns: Sequence[str],
    summing: Sequence[str],
    *,
    json_mode: bool,
    agent: bool,
    fields: str,
    title: str,
) -> None:
    """Print an analysis table: one row per job, then one row adding up what the batch costs.

    The total rides in the table rather than beside it, since every mode a caller can ask for
    renders rows and a figure printed outside them would be the one number `--json` dropped.
    """
    rows(
        [*payloads, totals(payloads, columns=columns, summing=summing)],
        mode=mode_of(json_mode=json_mode, agent=agent),
        fields=_fields(fields) or columns,
        title=title,
    )


def _status(status: BatchStatus, *, mode: str | None, fields: Sequence[str]) -> None:
    """Print one pass over a batch, its still-running count in the heading."""
    rows(
        [job.model_dump() for job in status.jobs],
        mode=mode,
        fields=fields,
        title=f"{status.batch}: {status.running} running",
    )


def _fields(raw: str) -> tuple[str, ...]:
    """`raw`'s comma-separated field names, trimmed and blank entries dropped."""
    return tuple(part.strip() for part in raw.split(",") if part.strip()) if raw else ()


def _answers(given: tuple[str, ...]) -> dict[str, str]:
    """The `question=value` pairs a caller passed, refusing one written without its value."""
    split = [pair.partition("=") for pair in given]
    if bare := [pair for pair, separator, _ in split if not separator]:
        raise MissionError(f"answers are written question=value, not {bare[0]!r}")
    return {question: answer for question, _, answer in split}


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
