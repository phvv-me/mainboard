import shlex
from copy import copy
from functools import partial
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Self

from ...batch.receipts import Journal
from ...context.expressions import evaluate
from ...context.resolver import Resolver
from ...core.errors import MissionError
from ...core.project import Project
from ...dispatch.dispatcher import Dispatcher
from ...dispatch.jobs.vocabulary import Resources
from ...dispatch.transport.targets import resolve as resolved_profile
from ...dispatch.transport.targets import rooted
from ...engines.compile.provisioner import Provisioner
from ...git import Tree
from ...jobs.targets.target import Target
from ...manifest.schema.root import Manifest
from ...manuscript import Manuscript
from ...state.storage.holds import holding
from ..nodes import evidence_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...context.plan import ExecutionPlan


# How long one manuscript build may take. A first build downloads the engine's bundle, so the
# ceiling is minutes, and a TeX run looping on a bad macro still ends inside it.
_PAPER_SECONDS = 900.0

type Shared = Dispatcher | Manifest | Resolver


class Workspace:
    """One workspace, bound to a host: its manifest, plans, shared caches and results paths."""

    @property
    def dispatcher(self) -> Dispatcher:
        """The dispatch core, rooted at this workspace and shared across every host pivot.

        Rooted rather than left to the cwd, so a command typed in a subdirectory reads the same
        run registry, stages into the same jobs directory and mirrors the same tree.
        """
        return self.once("dispatcher", Dispatcher, lambda: Dispatcher(root=self.root))

    @property
    def floor(self) -> str:
        """The tool version this workspace declares, empty when it declares none.

        A workspace vendoring the tool's source needs no version; one consuming it from an index
        names it like any other dependency, and a host with no vendored source installs that.
        """
        declared = self.manifest.requirement(self.project.package)
        return declared.version if declared is not None else ""

    @property
    def local(self) -> bool:
        """Whether this board is bound to the current machine."""
        return self.host == "local"

    @property
    def manifest(self) -> Manifest:
        """The loaded workspace manifest, shared across every `on` pivot."""
        return self.once("manifest", Manifest, lambda: holding(self.project.manifest(self.root)))

    @property
    def resolver(self) -> Resolver:
        """The plan resolver over this workspace's manifest."""
        return self.once("resolver", Resolver, lambda: Resolver(self.manifest))

    def __init__(self, root: Path | None = None, *, host: str = "local") -> None:
        """root: the workspace root, discovered upward from the cwd when None.

        host: the host alias this board is bound to, `local` for here.
        """
        self.project = Project()
        self.root = root or self.project.find_root(Path.cwd())
        self.host = host
        self.shared: dict[str, Shared] = {}
        self.guard = RLock()

    def declares(self, host: str) -> bool:
        """Whether `host` is this machine, a declared host or a held rental, one worth asking.

        A run recorded on any other host outlived its machine (a released rental, a host taken
        out of the manifest), so no sweep contacts it and a cancel settles it without asking.
        """
        return host == "local" or host in self.manifest.profiles()

    def git(self) -> Tree:
        """This workspace's checkout: the root repository and every submodule, as one tree."""
        return Tree(self.root, self.manifest.git)

    def on(self, host: str) -> Self:
        """This workspace bound to `host`, sharing the loaded manifest and caches.

        host: a declared host alias, or any ssh-config alias for defaults.
        """
        bound = copy(self)
        bound.host = host
        return bound

    def once[Built: Shared](
        self, key: str, kind: type[Built], build: Callable[[], Built]
    ) -> Built:
        """The one `key` this workspace shares, built on first ask and never a second time.

        kind: the class a slot's value must be an instance of, so any other value is rebuilt.

        Locked because the first ask routinely comes from a worker thread (a doctor asks four
        questions at once, a survey probes a fleet in a pool), and two threads building a
        dispatcher would open a second SQLite connection owned by whichever thread won. The lock
        is reentrant since one build reads another. A build that raises is not remembered, so an
        unparsable manifest is re-read and re-refused. Emptying a slot makes the values derived
        from it be rebuilt (a test rewriting the manifest).
        """
        with self.guard:
            found = self.shared.get(key)
            if isinstance(found, kind):
                return found
            built = build()
            self.shared[key] = built
            return built

    def paper(self, name: str) -> Manuscript:
        """The declared manuscript `name` (`[papers.<name>]`), built through this environment."""
        try:
            declared = self.manifest.papers[name]
        except KeyError:
            raise MissionError(
                f"no paper {name!r}; declared papers are {sorted(self.manifest.papers)}"
            ) from None

        provisioner = Provisioner(self.root, self.manifest)
        return Manuscript(
            name,
            declared,
            root=self.root,
            run=partial(provisioner.capture, env=self.plan().env, timeout=_PAPER_SECONDS),
        )

    def plan(self, *, env: str = "", container: str = "") -> ExecutionPlan:
        """The resolved execution plan for this board's host.

        A bound host's profile is completed from what its setup probed, so a platform the
        manifest never spelled out is still the one every later command stages for, and a `~`
        root is the one path under that host's home.
        """
        plan = self.resolver.plan(self.host, env=env, container=container)
        if self.local:
            return plan
        try:
            recorded = self.dispatcher.cache.hosts.recorded(self.host)
        except LookupError:
            return plan
        if recorded.capabilities is None:
            return plan
        profile = resolved_profile(plan.profile, recorded.capabilities, recorded=recorded.root)
        return plan.model_copy(update={"profile": profile})

    def receipts(self, stream: str) -> Journal:
        """Where one stream's events go: this workspace's lake, the one record of them.

        stream: the receipts stream, a batch id, or one run's own name.
        """
        return Journal(self.dispatcher.cache.session, stream)

    def remote_root(self) -> str:
        """The workspace root on the bound host, refusing one its setup never placed."""
        return rooted(self.plan().profile, host=self.host)

    def resources(self, asked: Resources, *, plan: ExecutionPlan | None = None) -> Resources:
        """`asked` resolved for this host, the profile's defaults filling what it leaves unset.

        Shared by `dispatch` and `expectation`, so a dispatch is priced at what it asks for.
        Expression-valued defaults are evaluated against the attempt, so a retry escalates
        instead of dying to the same ceiling twice.

        plan: an already-resolved execution plan, this board's own when None.
        """
        resolved = plan or self.plan()
        defaults = resolved.profile.defaults
        memory = asked.mem_gb or (
            evaluate(defaults.mem_gb, attempt=asked.attempt) if defaults.mem_gb else 0
        )
        return asked.model_copy(
            update={
                "queue": asked.queue or defaults.queue,
                "walltime": asked.walltime or defaults.walltime,
                "mem_gb": memory,
                "gpus": asked.gpus or defaults.gpus,
                # An asked card or architecture replaces the profile's default of the other kind.
                "gpu_name": asked.gpu_name or ("" if asked.arch else defaults.gpu_name),
                "max_usd": asked.max_usd or defaults.max_usd,
                "account": resolved.profile.account,
                "spot": asked.spot or defaults.spot,
                "datacenter": asked.datacenter or defaults.datacenter,
                "arch": asked.arch or ("" if asked.gpu_name else defaults.arch),
            }
        )

    def results(self, fetch: str | None, *, node: str = "", command: str = "") -> str:
        """What this dispatch pulls back: `fetch`, the job's declared path, or `node`'s evidence.

        A node is a directory and its evidence the directory inside it, so it answers for
        itself: making callers repeat it as `--fetch` is how a whole GH200 wave's receipts
        stayed on the cluster (2026-09-05). An explicit path still wins, unless it is a folder
        inside a test's home, which the receipts' store shares (`Target.results`).
        """
        target = Target.spelled(shlex.split(command), self.root) if command else None
        if target is not None:
            fetch = target.results(fetch or target.declaration(self.root).fetch)
        return fetch or evidence_of(self.root, node)
