from contextlib import suppress
from typing import NoReturn

from ...core.errors import MissionError
from ...dispatch.jobs.vocabulary import Allocation
from ...engines.compile.pixi.policy import Preparation
from ...log import logger
from ...render import progress, record
from ...runtime.job import Job
from ...runtime.runner import Runner
from ..base import Verbs

# The verbs' defaults: no allocation asked for, and the lock and environment as pixi brings them.
_ALLOCATION = Allocation()
_PREPARATION = Preparation()


class RunningVerbs(Verbs):
    """The `running` verbs."""

    def attest(self, stream: str, *, job: str = "") -> None:
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
        self.session.board("local").attest(stream, job=job or stream)

    def job_(self, record: str) -> int:
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

    def provide(
        self, env: str = "", *, source: str = "", expect: str = "", json: bool = False
    ) -> None:
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
            built = self.session.board("local").provide(env, source=source, expect=expect)
        if not json:
            logger.result(built.as_posix())
            return
        record({"prefix": str(built)}, mode="json", fields=(), title="prefix")

    def run(
        self,
        *command: str,
        on: str = "local",
        env: str = "",
        container: str = "",
        preparation: Preparation = _PREPARATION,
    ) -> int:
        """Run a command, or a job spelled `path/to/file.py::name`, through the host's plan.

        As `pixi run` does, a local run first solves a lock that no longer answers the manifest
        and installs the environment when it is not in line with that lock; `--locked`,
        `--frozen`, `--no-install` and `--as-is` say so as pixi does. Native file targets run
        locally with the same runner and closure format as submitted jobs; use submit for remote
        file targets, and run collection and help locally. Plain remote diagnostic commands
        execute over SSH, on a cluster's login endpoint rather than in a batch allocation, in
        what that host's setup installed. The exit code is the command's own.

        Args:
            command: the command tokens, from the first token that is not an option of this verb,
                passed on verbatim with its own flags; a job's arguments follow `--`.
            on: the host alias, `local` for this machine.
            env: an environment name overriding the profile's choice.
            container: a container override, `none` forcing bare.
        """
        return self.session.board(on).run(
            command,
            env=env,
            container=container,
            policy=preparation.policy,
            install=preparation.installs,
        )

    def sample(
        self,
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
        sampler = self.session.board("local").samples(
            stream, job=job or stream, interval=interval, seconds=seconds, parent=parent
        )
        with suppress(KeyboardInterrupt), sampler:
            sampler.thread.join()

    def shell(
        self,
        *command: str,
        on: str = "local",
        env: str = "",
        allocation: Allocation = _ALLOCATION,
        preparation: Preparation = _PREPARATION,
    ) -> NoReturn:
        """Open an interactive shell in this workspace's environment, here or on a host.

        The daily way in, and the one verb that works from a terminal where nothing is
        activated yet. This process becomes the shell, so quitting it returns to the terminal
        that asked. Here, as `pixi shell` does, a lock that no longer answers the manifest is
        solved and the environment installed first, unless `--locked`, `--frozen`,
        `--no-install` or `--as-is` say otherwise. On a host the shell opens inside its mirrored
        workspace, and a queued host is asked for an interactive allocation first, so the
        terminal lands on a compute node rather than on the login node the request was made
        from.

        Args:
            command: on a host, a command to run instead of handing over the terminal, from the
                first token that is not an option of this verb.
            on: the host alias, `local` for this machine.
            env: the environment name, the profile's declared choice when omitted.
        """
        if on != "local":
            self.session.board(on).interact(*command, env=env, allocation=allocation)
        elif command or allocation != _ALLOCATION:
            raise MissionError(
                "a command, a queue, a walltime, --nodes and --keep belong to a host's shell; run "
                f"a command here with `{self.session.project.name} run -- <command>`"
            )
        else:
            self.session.board("local").shell(
                env, policy=preparation.policy, install=preparation.installs
            )
