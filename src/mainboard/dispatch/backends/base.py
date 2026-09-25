# The provider backend contract, plus `route`, which decides by kind whether a host stays on the
# ssh-family `Scheduler` path or resolves to a registered `ProviderBackend`.
#
# `ProviderBackend` carries only the lifecycle every provider has (launch, poll, cancel); every
# other verb is a `Capability` a backend opts into by inheriting it, so a caller asks
# `isinstance(backend, LogSource)` before asking for a log instead of discovering mid-sweep that
# the provider never had one.
#
# Nothing here pins a source tree on purpose: a provider rents a machine per job and ships it its
# own tree no later dispatch can reach, so it already has the immutability the ssh family buys
# with a per-dispatch snapshot of its mirror (`dispatch.snapshots`).

import abc
import os
import re
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol
from urllib.error import HTTPError
from urllib.request import urlopen

from patos import FrozenModel, Registry, Singleton
from pydantic import field_validator

from ...core.errors import MissionError
from ...core.project import Project
from ..rentals import Rental

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from urllib.request import Request

    from ...context.plan import ExecutionPlan
    from ...costs.catalog import Offer
    from ..allocation import Allocation
    from ..transport import Endpoint
    from ..vocabulary import JobState, Resources

    class HttpResponse(Protocol):
        """The subset of `http.client.HTTPResponse` a transport callable must return."""

        status: int

        def read(self) -> bytes: ...

    type Transport = Callable[[Request], HttpResponse]

# Kinds the `Scheduler` path dispatches; `auto` stays here, as `Scheduler.pick` treats an
# unprobed kind as ssh.
_SSH_FAMILY_KINDS = frozenset({"auto", "local", "pbs", "slurm", "ssh"})
# A provider that stops answering costs one slow row rather than a wedged command: enough for a
# cold offer search, short enough that a fleet survey finishes while someone watches.
_TIMEOUT_S = 10.0
_ENV_FILE = ".env"
# A CUDA version as `nvidia/cuda:13.3.1-devel-ubuntu24.04`, `vastai/base-image:cuda-13.3.1-auto`
# and `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` spell it. Only major and minor are read,
# since a patch level never decides whether an image loads on a driver.
_IMAGE_CUDA = re.compile(r"cuda[:_-]?(\d+)\.(\d+)", re.IGNORECASE)


class Credentials(Singleton):
    """The workspace `.env`, merged into this process's environment once per process.

    Every backend refusal names a variable to set there, and while nothing read the file a survey
    called a provider unkeyed whose keys sat at the workspace root; every backend crosses this
    seam before looking a key up. The environment always wins, so a stale line never shadows an
    exported key. The file is data, never shell: `NAME=value` lines, blanks and `#` comments
    skipped, one matching pair of surrounding quotes off, nothing expanded or executed. Only the
    names defined are returned, never a value, and nothing is logged.
    """

    def __init__(self) -> None:
        self.project = Project()
        self.loaded = False
        self.lock = Lock()

    def load(self) -> tuple[str, ...]:
        """Define what the workspace `.env` declares and this environment lacks, by name.

        Empty after the first call, outside a workspace (found by walking up to the manifest) and
        in one without a `.env`. The whole merge is one critical section rather than a flag
        flipped up front: a compute survey probes every provider at once, and a second provider
        finding nothing mid-merge reported a keyed account unkeyed (seen live 2026-08-19).
        """
        with self.lock:
            if self.loaded:
                return ()
            self.loaded = True
            return self.merged()

    def merged(self) -> tuple[str, ...]:
        """Read the workspace `.env` and define what it declares, returning the names."""
        try:
            text = (self.project.find_root(Path.cwd()) / _ENV_FILE).read_text(encoding="utf-8")
        except OSError:
            return ()
        defined: list[str] = []
        for line in text.splitlines():
            entry = line.strip().removeprefix("export ").strip()
            name, assigned, value = entry.partition("=")
            name = name.strip()
            if entry.startswith("#") or not name or not assigned or name in os.environ:
                continue
            value = value.strip()
            quoted = len(value) > 1 and value[0] == value[-1] and value[0] in "\"'"
            os.environ[name] = value[1:-1] if quoted else value
            defined.append(name)
        return tuple(defined)


class Standing(FrozenModel):
    """What a provider cheaply says about itself, the only thing a compute survey asks it.

    keyed: whether this provider's credentials are present here; never the credential itself.
    credit_usd: the balance the provider reports, None when it publishes spend but no balance.
    usd_hr: the cheapest live rate a sample search found, None when no price is a cheap question.
    note: the row's one human line, the variable to set or why there is no credit. Never a secret.
    """

    keyed: bool = False
    credit_usd: float | None = None
    usd_hr: float | None = None
    note: str = ""

    @field_validator("credit_usd", "usd_hr")
    @classmethod
    def rounded(cls, value: float | None) -> float | None:
        """Money to four places, since hourly GPU rates differ by fractions of a cent."""
        return None if value is None else round(value, 4)


class Capability:
    """One optional half of the backend contract, opted into by inheriting it.

    A backend lacking one names it in `ProviderBackend.lacks` with the refusal line, so the gap
    is a typed absence seen before calling. The root declares nothing; it keys `lacks` by
    contract rather than by a bare name, keeping a declared gap and its class in step.
    """


class Account(Capability, abc.ABC):
    """A provider that answers for its own account rather than for one run."""

    @abc.abstractmethod
    def standing(self) -> Standing:
        """Whether this provider is usable from here, priced and credited where that is cheap.

        A new backend joins the compute survey by implementing this. It never reveals a
        credential and reaches the network only once it found one, so an unconfigured provider
        is free to list.
        """


class Delivery(Capability):
    """A provider that can bring a finished run's artifacts back to this machine."""

    @abc.abstractmethod
    def deliver(self, handle: str, *, path: str) -> None:
        """Fetch `handle`'s output at `path` back to the local filesystem."""


class Rented(FrozenModel):
    """One machine a provider says this account is renting right now.

    handle: the provider's id for the rental, which ends it.
    label: the label it was created under, `mainboard-<id>` for one this tool rented.
    gpu: the cards it carries, as the provider names them.
    usd_hr: what it bills per hour, None when the listing carries no rate.
    """

    handle: str
    label: str = ""
    gpu: str = ""
    status: str = ""
    usd_hr: float | None = None


class Inventory(Capability, abc.ABC):
    """A provider that lists every machine the account is renting, however it was rented."""

    @abc.abstractmethod
    def rentals(self) -> list[Rented]:
        """Every live rental on the account, as the provider reports it.

        Only the provider's listing catches a machine rented from another checkout, by hand, or
        by a dispatch whose record was lost, all of which bill the same.
        """


class LogSource(Capability):
    """A provider that keeps a run's captured output and will hand it back."""

    @abc.abstractmethod
    def logs(self, handle: str) -> str:
        """`handle`'s captured log so far."""


class Market(Capability):
    """A provider that quotes a live rentable market, priced offer by offer."""

    @abc.abstractmethod
    def catalog(self, *, gpu_name: str = "", gpus: int = 0, limit: int = 0) -> list[Offer]:
        """Live offers as catalog rows, the authed refresh of an imported price feed.

        gpu_name: the provider's GPU name, empty for the whole market.
        gpus: the GPU count per machine, 0 for any.
        limit: how many offers to bring back, 0 for the backend's own page size.
        """


class Rentable(Capability, abc.ABC):
    """A provider whose rental answers ssh, so a dispatch lands on it as on a host.

    It hands back a machine with its entrypoint waiting, and the ordinary mirror, install,
    provision and pin path does the rest (`dispatch.rentals` says why a bare rental cannot run a
    command). A backend without it runs the raw command as its container's entrypoint.
    """

    @abc.abstractmethod
    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Reconnect to a rental, including after the submitting process has exited."""

    @abc.abstractmethod
    def rent(self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation) -> Rental:
        """Rent a machine for one job and return it once ssh answers on it.

        The job is not started: the entrypoint waits for the landing's launch script, so a caller
        that never lands must cancel the handle or the rental bills until the entrypoint gives up.
        A rental that cannot be opened is ended before this raises, as nothing else holds it yet.
        """


class ProviderBackend(Registry, abc.ABC):
    """Registry root for non-ssh provider backends, one concrete class per `HostProfile.kind`.

    Unlike `Scheduler`, no method takes a `remote`/`root`: a backend owns its transport (an HTTP
    session, an SDK client) end to end. A new backend subclasses this, implements the three
    lifecycle methods, inherits whichever capabilities it can honor, and `route` finds it.
    """

    # What this backend cannot do, and the line to print instead, keyed by the contract it does
    # not inherit. `{handle}` and `{path}` are filled in where the refusal is raised.
    lacks: ClassVar[Mapping[type[Capability], str]] = {}

    # The CUDA version this house builds, measures and rents on, a floor rather than a pin (a
    # 13.3 driver runs a 13.0 image, a 12.6 one neither). A rented machine is the one place
    # nothing else enforces it: the provider destroys an instance whose image its driver cannot
    # load and bills for the boot. Five vast dispatches went that way on 2026-08-27 on Tesla T4
    # offers reporting `cuda_max_good = 12.6` under a 12.9 image, so a mismatch is a refusal
    # naming both versions rather than a skip.
    CUDA_FLOOR: ClassVar[float] = 13.0

    # The oldest compute capability CUDA still builds for, as a provider spells it (`750` is
    # `sm_75`, Turing), read off `nvcc --list-gpu-arch` of the house CUDA 13.3 toolchain (12.9
    # also offered compute_50 through compute_72). A separate floor because it fails differently:
    # a driver too old kills the instance at boot, while an architecture too old boots, bills and
    # dies at the first kernel launch with "no kernel image is available for execution on the
    # device". A live vast search on 2026-08-27 found Volta and Pascal machines reporting
    # `cuda_max_good` of 13.0, exactly that trap.
    CAPABILITY_FLOOR: ClassVar[int] = 750

    def admit(self, plan: ExecutionPlan, resources: Resources) -> None:
        """Every refusal a metered dispatch owes before it reaches a provider's API at all.

        Every `submit` and `rent` opens with it, so a house rule on what may be rented is one
        edit. Each refusal is free here and otherwise costs a whole rental, billed from the boot.
        """
        if not resources.max_usd:
            raise MissionError("provider dispatch needs an explicit max-usd budget")
        container = plan.container
        if container is None:
            return
        named = image_cuda(container.image)
        if named is not None and named < self.CUDA_FLOOR:
            raise MissionError(
                f"image {container.image!r} names CUDA {named}, below this house's CUDA "
                f"{self.CUDA_FLOOR} floor; rent a CUDA {self.CUDA_FLOOR} or newer image instead"
            )

    @abc.abstractmethod
    def cancel(self, handle: str) -> None:
        """End `handle` on the provider, tolerating one it already forgot.

        The durable sweep cancels every run it settles and may settle one twice (a pass killed
        before advancing its cursor), and someone may have ended the rental by hand, so a gone
        rental is the state asked for rather than a fault.
        """

    def refusal(self, capability: type[Capability], **facts: str) -> str:
        """Why this backend cannot answer `capability`: its own `lacks` advice, else a plain line.

        facts: the `handle` and `path` the advice names.
        """
        advice = self.lacks.get(capability)
        if advice is None:
            return f"the {self.name} backend does not implement {capability.__name__}"
        return advice.format(**facts)

    @abc.abstractmethod
    def state(self, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict."""

    @abc.abstractmethod
    def submit(
        self, plan: ExecutionPlan, command: str, resources: Resources, *, allocation: Allocation
    ) -> str:
        """Launch `command` under `resources`; return the provider's opaque handle id."""


def forgotten(error: HTTPError) -> dict:
    """An empty row for the 404 a gone instance answers, re-raising every other refusal."""
    if error.status != 404:
        raise error
    return {}


def http_transport(request: Request) -> HttpResponse:
    """Send `request` over urllib, the default transport of every REST backend.

    The one audited url open in the package: each backend builds its `Request` from a constant
    https root and tests swap this callable out, so no unvetted scheme reaches urllib. The
    deadline is ours, since urllib has none.
    """
    return urlopen(request, timeout=_TIMEOUT_S)  # ruff:ignore[suspicious-url-open-usage]  reason=the package's single audited seam, every caller builds its Request from a constant https root and tests inject a double since=2026-08-18


def image_cuda(image: str) -> float | None:
    """The CUDA version `image`'s reference names, None when it names none.

    The text is all anyone has before the rental; pulling the image means paying for it. An
    image that names nothing passes, since refusing a CPU job or an NGC calendar tag would prove
    nothing: the floor refuses only what it can prove wrong.
    """
    found = _IMAGE_CUDA.search(image)
    return float(f"{found[1]}.{found[2]}") if found else None


def route(kind: str) -> Literal["ssh-family"] | type[ProviderBackend]:
    """Whether `kind` runs the ssh-family `Scheduler` path or a registered `ProviderBackend`.

    Takes the kind rather than a profile, since a dispatched run is rebuilt from the kind the
    dispatch cache recorded and no profile is left by then. An unknown kind is refused naming
    the known ones.
    """
    if kind in _SSH_FAMILY_KINDS:
        return "ssh-family"
    try:
        return ProviderBackend.find(kind)
    except KeyError:
        raise MissionError(
            f"no provider backend for kind {kind!r}; known kinds are {ProviderBackend.names()}"
        ) from None
