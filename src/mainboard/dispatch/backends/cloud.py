# Machines rented from any cloud through one template, no server between.
#
# Studied against SkyPilot (a provisioner per cloud: run, wait, query, stop, terminate instances,
# behind an API server) and dstack (a `Compute` base, `get_offers` / `create_instance` /
# `terminate_instance`, with capability mixins, also behind a server). Both keep the machine's
# lifecycle small and put the work around it (pricing, keys, waiting for ssh, cleaning up a
# machine that failed) in shared code. Here that shared code is `CloudBackend.rent`, and a
# provider supplies five primitives: `create`, `machine`, `machines`, `terminate`, plus its
# catalog name. Everything after ssh answers is mb's ordinary host path (mirror, tool, lock,
# pueue), which is why a held machine is set up once and then takes jobs like any host.
#
# A cloud machine has no entrypoint mb can rely on (a Lambda VM has none at all), so once ssh
# answers the rent starts the landing's waiter itself, detached, its output and exit status
# written beside the launch script. A held machine parks that waiter on an idle command; a
# one-shot `job submit` hands it the job, and `state` and `logs` read the two files over ssh.
#
# Offers come from gpuhunt, the catalog dstack publishes as its own library (MPL-2.0): public
# price lists and live marketplaces across AWS, GCP, Azure, Lambda, RunPod, Vast, Verda, Nebius
# and more, most of them readable without an account. It is the same row whichever cloud rents
# it, so `mb host offers` compares every cloud even where mb cannot rent yet.

import abc
import json
import logging
import os
import shlex
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.request import Request

from patos import FrozenModel

from ...core.errors import MissionError
from ...costs.catalog import Offer
from ...log import logger
from ..arch import Arch, arch, capability, card
from ..lease import Lease
from ..rentals import LANDING_SECONDS, Rental, identity, reachable, waiting
from ..transport import Endpoint, HostUnreachable, SshTransport
from ..vocabulary import JobState
from .base import (
    Account,
    Credentials,
    Delivery,
    Inventory,
    LogSource,
    Market,
    ProviderBackend,
    Rentable,
    Rented,
    Standing,
    hourly_cap,
    http_transport,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...context.plan import ExecutionPlan
    from ..allocation import Allocation
    from ..vocabulary import Resources
    from .base import Transport

# How long a created machine may take to publish an address before it is ended unused.
_BOOT_SECONDS = 900
_POLL_SECONDS = 10.0
# How many offers a rent tries when the first ones are taken or out of capacity.
_OFFERS_TRIED = 5
# Where the detached waiter leaves the job's output and exit status, beside the launch script.
_LOG_PATH = "/tmp/mainboard.log"
_EXIT_PATH = "/tmp/mainboard.exit"


class Machine(FrozenModel):
    """One machine as its cloud reports it.

    status: `pending` until it can take ssh, `running`, or `gone` once ended or lost.
    host / port / user: where ssh reaches it, empty until the cloud publishes an address.
    """

    handle: str
    status: str
    host: str = ""
    port: int = 22
    user: str = "root"
    label: str = ""
    gpu: str = ""
    usd_hr: float | None = None


class CapacityGone(MissionError):
    """The offer chosen was taken or has no capacity left; another one may still rent."""


def hunt(
    *,
    providers: Sequence[str] = (),
    gpu_name: str = "",
    gpus: int = 0,
    max_usd_hr: float = 0.0,
    spot: bool | None = None,
    arch: Arch | None = None,
) -> list[Any]:
    """gpuhunt's offers for the filters, cheapest first; providers with no key are skipped.

    Imported here: listing the market is the one thing that needs it, and it reads a dozen
    catalogs, so a command that never asks pays nothing. The architecture is filtered here
    rather than by gpuhunt, whose table has B300 at sm_100.

    arch: the capabilities a card must have, any card of them when `gpu_name` is empty.
    """
    import gpuhunt  # type: ignore[import-untyped]  # noqa: PLC0415

    # It logs a warning per catalog it cannot read offline (azure), which says nothing to act on.
    logging.getLogger("gpuhunt").setLevel(logging.ERROR)
    query: dict[str, Any] = {}
    if providers:
        query["provider"] = list(providers)
    if gpu_name:
        query["gpu_name"] = [card(gpu_name)]
    if gpus:
        query.update(min_gpu_count=gpus, max_gpu_count=gpus)
    if max_usd_hr:
        query["max_price"] = max_usd_hr
    if spot is not None:
        query["spot"] = spot
    try:
        found = gpuhunt.query(**query)
    except OSError:
        found = _answering(gpuhunt, query)
    if arch is not None:
        found = [item for item in found if item.gpu_name and arch.holds(capability(item.gpu_name))]
    return sorted(found, key=lambda item: item.price)


def _answering(gpuhunt: Any, query: dict[str, Any]) -> list[Any]:
    """The offers of every catalog that answers, asked one at a time, each silent one named.

    gpuhunt asks its catalogs together and one failing fails them all: a 502 from Vultr's API
    ended `host offers` as a traceback and hid every other cloud's prices (2026-09-30).
    """
    from gpuhunt._internal.catalog import (  # type: ignore[import-untyped]  # noqa: PLC0415
        OFFLINE_PROVIDERS,
    )

    loaded = (provider.NAME for provider in gpuhunt.default_catalog().providers)
    found: list[Any] = []
    for name in query.get("provider") or [*OFFLINE_PROVIDERS, *loaded]:
        try:
            found += gpuhunt.query(**{**query, "provider": [name]})
        except OSError as silent:
            logger.warning("{} listed no offers: {}", name, silent)
    return found


def as_offer(item: Any) -> Offer:
    """A gpuhunt row as the offer every lease and estimate prices."""
    return Offer(
        provider=item.provider,
        gpu=item.gpu_name or "cpu",
        gpu_count=item.gpu_count or 0,
        spot=bool(item.spot),
        region=item.location or "",
        rate_usd_hr=item.price,
        source=f"gpuhunt:{item.provider}:{item.instance_name}",
    )


class CloudBackend(ProviderBackend, Account, Inventory, LogSource, Market, Rentable):
    """A cloud that rents ssh machines by the hour, held (`mb host hold`) or one per job.

    A subclass names its gpuhunt catalog and its key variables and implements the primitives
    below; renting, leasing, reaching, reporting and ending a machine are shared. A job
    submitted to a cloud host lands on a machine rented for it, ended once the job settles; a
    held machine is set up once and takes any number of jobs through its alias.
    """

    # The name gpuhunt lists this cloud under, and the variables its key may be set in.
    catalog_name: ClassVar[str] = ""
    key_variables: ClassVar[tuple[str, ...]] = ()
    # The image a container cloud boots: plain Ubuntu, since the environment brings its own CUDA
    # libraries and only the host driver comes from the machine.
    image: ClassVar[str] = "ubuntu:24.04"

    lacks = {
        Delivery: "a cloud machine is an ssh host: `mb job collect {path} --on <alias>`",
    }

    def __init__(
        self,
        *,
        transport: Transport = http_transport,
        sleeper: Callable[[float], None] = sleep,
        disk_gb: int = 100,
    ) -> None:
        self.transport = transport
        self.sleeper = sleeper
        self.disk_gb = disk_gb

    # The primitives a cloud implements.

    @abc.abstractmethod
    def create(self, offer: Any, *, name: str, public_key: str, disk_gb: int) -> str:
        """Start a machine for gpuhunt `offer` with `public_key` authorized; its handle.

        Raises `CapacityGone` when that offer can no longer be rented, so the next is tried.
        """

    @abc.abstractmethod
    def machine(self, handle: str) -> Machine:
        """`handle` as the cloud reports it now; `gone` when the cloud no longer knows it."""

    @abc.abstractmethod
    def machines(self) -> list[Machine]:
        """Every machine this account holds on the cloud, rented here or not."""

    @abc.abstractmethod
    def terminate(self, handle: str) -> None:
        """End `handle`, tolerating one already gone."""

    # Shared.

    def call(self, method: str, url: str, body: object = None) -> Any:
        """One JSON request to this cloud's API with its key; the parsed answer, {} when empty.

        HTTP errors propagate, so a caller reads the refusal (capacity, a gone handle) itself.
        """
        data = None if body is None else json.dumps(body).encode()
        request = Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.key()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "mainboard",
            },
        )
        raw = self.transport(request).read()
        return json.loads(raw) if raw.strip() else {}

    def key(self) -> str:
        """This cloud's API key from the environment or the workspace `.env`."""
        Credentials().load()
        for variable in self.key_variables:
            if found := os.environ.get(variable, ""):
                return found
        raise MissionError(
            f"set {' or '.join(self.key_variables)} in the workspace .env to rent on {self.name}"
        )

    def keyed(self) -> bool:
        """Whether a key is present, asked without reaching the network."""
        try:
            self.key()
        except MissionError:
            return False
        return True

    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        rows = [
            as_offer(item)
            for item in hunt(providers=[self.catalog_name], gpu_name=gpu_name, gpus=gpus)
        ]
        return rows[:limit] if limit else rows

    def standing(self) -> Standing:
        if not self.keyed():
            return Standing(note=f"set {' or '.join(self.key_variables)} to rent here")
        return Standing(keyed=True, note="gpuhunt prices; `mb host offers` lists them")

    def rentals(self) -> list[Rented]:
        return [
            Rented(handle=m.handle, label=m.label, gpu=m.gpu, status=m.status, usd_hr=m.usd_hr)
            for m in self.machines()
        ]

    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """The cheapest offer under the budget, created, reachable over ssh, or nothing billing.

        Each offer tried is leased first, so a crash between create and record still leaves the
        run registry holding the handle the sweep ends.
        """
        self.admit(plan, resources)
        key = identity(plan.profile.vars.get("ssh-key", ""))
        gpus = max(resources.gpus, 1)
        span = arch(resources.arch) if resources.arch else None
        cap = hourly_cap(resources, landing=LANDING_SECONDS)
        found = hunt(
            providers=[self.catalog_name],
            gpu_name=resources.gpu_name,
            gpus=gpus,
            max_usd_hr=cap,
            # On-demand unless asked: an interruptible machine can be taken back mid-run, which
            # a job survives only through its checkpoint (`mb job submit --resume`).
            spot=resources.spot,
            arch=span,
        )
        if not found:
            asked = " ".join(
                part
                for part in (
                    resources.gpu_name,
                    span.spelled if span else "",
                    "spot" if resources.spot else "",
                )
                if part
            )
            raise MissionError(
                f"{self.name} lists no {gpus}x {asked or 'GPU'} offer"
                + (f" under ${cap:.2f}/h" if cap else "")
                + " right now; `mb host offers` shows every cloud's"
            )
        for offer in found[:_OFFERS_TRIED]:
            allocation.begin(
                lease=Lease.priced(as_offer(offer), resources, setup_s=LANDING_SECONDS)
            )
            try:
                handle = self.create(
                    offer, name=allocation.label, public_key=key.public, disk_gb=self.disk_gb
                )
            except CapacityGone as gone:
                allocation.refused()
                logger.warning(
                    "{} offer {} is gone ({}); trying the next",
                    self.name,
                    offer.instance_name,
                    gone,
                )
                continue
            allocation.created(handle)
            break
        else:
            raise MissionError(
                f"{self.name} had no capacity for the {len(found[:_OFFERS_TRIED])} cheapest offers"
            )
        try:
            endpoint = reachable(self.booted(handle, key.private), sleeper=self.sleeper)
            self.shell(endpoint, _detached(_waiter()), operation="start the waiter")
            return Rental(handle=handle, endpoint=endpoint)
        except BaseException:
            logger.warning("{} machine {} never became reachable; ending it", self.name, handle)
            self.terminate(handle)
            raise

    def booted(self, handle: str, private: str) -> Endpoint:
        """The ssh endpoint of `handle` once its cloud publishes one."""
        deadline = monotonic() + _BOOT_SECONDS
        while monotonic() < deadline:
            found = self.machine(handle)
            if found.status == "gone":
                raise MissionError(f"{self.name} machine {handle} ended while booting")
            if found.status == "running" and found.host:
                return Endpoint(
                    address=found.host, port=found.port, user=found.user, identity=private
                )
            self.sleeper(_POLL_SECONDS)
        raise MissionError(
            f"{self.name} machine {handle} published no address in {_BOOT_SECONDS}s"
        )

    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        found = self.machine(handle)
        if not found.host:
            raise MissionError(f"{self.name} machine {handle} has no address ({found.status})")
        return Endpoint(address=found.host, port=found.port, user=found.user, identity=key)

    def cancel(self, handle: str) -> None:
        self.terminate(handle)

    def logs(self, handle: str) -> str:
        return self.read(handle, _LOG_PATH)

    def read(self, handle: str, path: str) -> str:
        """The file at `path` on `handle`, empty while it is not there yet."""
        endpoint = self.endpoint(handle, key=identity().private)
        return self.shell(
            endpoint, f"cat {shlex.quote(path)} 2>/dev/null || true\n", operation="read"
        )

    def shell(self, endpoint: Endpoint, script: str, *, operation: str) -> str:
        """Run `script` under bash on `endpoint`, answering its stdout; a failure is refused."""
        policy = SshTransport(endpoint=endpoint)
        where = endpoint.destination
        code, out, err = policy.invoke(
            (*policy.command(where), "bash -s"),
            where,
            operation=operation,
            input_text=script,
        )
        if code:
            raise MissionError(f"{self.name} {operation} on {where} exited {code}: {err.strip()}")
        return out

    def state(self, handle: str) -> JobState:
        """The machine's status, then the job's own exit file once the machine runs.

        A machine the cloud ended reads `vanished`, which is also what an interruptible machine
        taken back reads as; its job resumes from its checkpoint with `mb job submit --resume`.
        """
        found = self.machine(handle)
        if found.status == "gone":
            return JobState(handle=handle, state=found.status, verdict="vanished")
        if found.status != "running" or not found.host:
            return JobState(handle=handle, state=found.status, verdict="running")
        try:
            said = self.read(handle, _EXIT_PATH).strip()
        except (HostUnreachable, MissionError) as unanswered:
            logger.debug("{} {} gave no exit file: {}", self.name, handle, unanswered)
            said = ""
        if not said.lstrip("-").isdigit():
            return JobState(handle=handle, state=found.status, verdict="running")
        code = int(said)
        verdict = "ok" if code == 0 else "failed"
        return JobState(handle=handle, state=found.status, exit_code=code, verdict=verdict)

    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        raise MissionError(
            f"{self.name} runs no prebuilt image: drop the container from {plan.host!r} so the "
            "job lands on a machine rented for it, or hold one with `mb host hold`"
        )


def _waiter() -> str:
    """The landing's waiter, its output and exit status kept where `state` and `logs` read."""
    return (
        f"{{\n{waiting()}\n}} > {_LOG_PATH} 2>&1\n"
        f"echo $status > {_EXIT_PATH}.part && mv {_EXIT_PATH}.part {_EXIT_PATH}\n"
    )


def _detached(script: str) -> str:
    """`script` started so it outlives the ssh session that starts it."""
    return f"nohup setsid bash -c {shlex.quote(script)} >/dev/null 2>&1 </dev/null &\n"


def keyed(kind: str) -> bool:
    """Whether the provider backend registered as `kind` has its credentials here."""
    backend = ProviderBackend.find(kind)()
    if isinstance(backend, CloudBackend):
        return backend.keyed()
    return isinstance(backend, Account) and backend.standing().keyed
