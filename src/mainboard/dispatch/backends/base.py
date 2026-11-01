# The provider backend contract, plus `route`, which decides by kind whether a host stays on the
# ssh-family `Scheduler` path or resolves to a registered `ProviderBackend`.
#
# `ProviderBackend` carries only the lifecycle every provider has (launch, poll, cancel); every
# other verb is a `Capability` (`capabilities.py`) a backend opts into by inheriting it, so a
# caller asks `isinstance(backend, LogSource)` before asking for a log instead of discovering
# mid-sweep that the provider never had one.
#
# Nothing here pins a source tree on purpose: a provider rents a machine per job and ships it its
# own tree no later dispatch can reach, so it already has the immutability the ssh family buys
# with a per-dispatch snapshot of its mirror (`dispatch.snapshots`).

import abc
import os
import re
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, ClassVar, Literal
from urllib.error import HTTPError
from urllib.request import urlopen

from patos import Registry, Singleton

from ...core.constants.units import SECONDS_PER_HOUR
from ...core.environ import Dotenv
from ...core.errors import MissionError, NoWorkspace
from ...core.project import Project
from ...enums import Bound
from ..hosts.lease import Cap
from ..hosts.rentals import Rental
from .capabilities import Capability

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from urllib.request import Request

    from ...context.plan import ExecutionPlan
    from ...costs.catalog import Offer
    from ..hosts.allocation import Allocation
    from ..jobs.vocabulary import JobState, Resources
    from ..transport.core import Endpoint

    # What a REST backend sends a request through, answering the response body.
    type Transport = Callable[[Request], bytes]

# Kinds the `Scheduler` path dispatches; `auto` stays here, as `Scheduler.pick` treats an
# unprobed kind as ssh. `held` is the kind a queue declaring that scheduler records on its runs.
_SSH_FAMILY_KINDS = frozenset({"auto", "held", "local", "pbs", "slurm", "ssh"})
# A provider that stops answering costs one slow row rather than a wedged command: enough for a
# cold offer search, short enough that a fleet survey finishes while someone watches.
_TIMEOUT_S = 10.0
# A CUDA version as `nvidia/cuda:13.3.1-devel-ubuntu24.04`, `vastai/base-image:cuda-13.3.1-auto`
# and `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` spell it. Only major and minor are read,
# since a patch level never decides whether an image loads on a driver.
_IMAGE_CUDA = re.compile(r"cuda[:_-]?(\d+)\.(\d+)", re.IGNORECASE)


class Credentials(Singleton):
    """The workspace `.env`, merged into this process's environment once per process.

    Every backend refusal names a variable to set there, and while nothing read the file a survey
    called a provider unkeyed whose keys sat at the workspace root; every backend crosses this
    seam before looking a key up. The environment always wins, so a stale line never shadows an
    exported key. The file is read as `Dotenv` reads it, as data and never as shell. Only the
    names defined are returned, never a value, and nothing is logged.
    """

    def __init__(self) -> None:
        self.project = Project()
        self.loaded = False
        self.lock = Lock()

    def key(self, *names: str, hint: str) -> str:
        """The first of the variables `names` set here or in the workspace `.env`.

        Refuses with `hint` (where such a key comes from) when none is.
        """
        self.load()
        if found := next(filter(None, (os.environ.get(name, "") for name in names)), ""):
            return found
        raise MissionError(f"set {' or '.join(names)} in the workspace .env, {hint}")

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
            root = self.project.find_root(Path.cwd())
        except NoWorkspace:
            return ()
        defined = {
            name: value for name, value in Dotenv.at(root).values.items() if name not in os.environ
        }
        os.environ.update(defined)
        return tuple(defined)


class Rentable(Capability, abc.ABC):
    """A provider whose rental answers ssh, so a dispatch lands on it as on a host.

    It hands back a machine with its entrypoint waiting, and the ordinary mirror, install,
    provision and pin path does the rest (`dispatch.rentals` says why a bare rental cannot run a
    command). A backend without it runs the raw command as its container's entrypoint.
    """

    @abc.abstractmethod
    def cancel(self, handle: str) -> None:
        """Cancel `handle` on the provider, which is what stops the meter."""

    @abc.abstractmethod
    def endpoint(self, handle: str, *, key: str = "") -> Endpoint:
        """Reconnect to a rental, including after the submitting process has exited."""

    @abc.abstractmethod
    def rent(
        self, plan: ExecutionPlan, resources: Resources, *, allocation: Allocation, cap: Cap
    ) -> Rental:
        """Rent a machine for one job and return it once ssh answers on it.

        The job is not started: the entrypoint waits for the landing's launch script, so a caller
        that never lands must cancel the handle or the rental bills until the entrypoint gives up.
        A rental that cannot be opened is ended before this raises, as nothing else holds it yet.

        cap: the provider's bound the warden let the rental through under (`Warden.rent`).
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

    # What stops this provider billing when nothing here is left to release its machine. Every
    # concrete backend names one, so a rental is created only behind a cap (`cap`).
    bound: ClassVar[Bound]

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

    # The download rate, in Mbps, a rented host must offer. A cold rental pulls a multi-gigabyte
    # image or environment before its job can start: hosts below this spent the whole address
    # wait pulling and ended with no container (three vast rentals, 2026-09-12), while hosts at a
    # few Gbps were running inside two minutes. Providers that publish the rate (vast's
    # `inet_down`, runpod's `minDownloadMbps`) filter on it before renting.
    DOWNLOAD_FLOOR_MBPS: ClassVar[float] = 500.0

    # The seconds a machine of this provider bills at least, whatever its job used, which no
    # catalog row carries (Hot Aisle bills 30 minutes). A five-minute job on a 60-minute minimum
    # costs an hour, so the leases and the scorecard read it from here.
    BILLED_MINIMUM_S: ClassVar[int] = 0

    @abc.abstractmethod
    def cancel(self, handle: str) -> None:
        """End `handle` on the provider, tolerating one it already forgot.

        The durable sweep cancels every run it settles and may settle one twice (a pass killed
        before advancing its cursor), and someone may have ended the rental by hand, so a gone
        rental is the state asked for rather than a fault.
        """

    @abc.abstractmethod
    def state(self, handle: str) -> JobState:
        """Post-mortem `handle`: its state, exit code, and a verdict."""

    @abc.abstractmethod
    def submit(
        self,
        plan: ExecutionPlan,
        command: str,
        resources: Resources,
        *,
        allocation: Allocation,
        cap: Cap,
    ) -> str:
        """Launch `command` under `resources`; return the provider's opaque handle id.

        cap: the provider's bound the warden let the run through under (`Warden.rent`).
        """

    @classmethod
    def billed(cls, offer: Offer) -> Offer:
        """`offer` with the billing terms this provider adds to its catalog row."""
        return offer.model_copy(update={"minimum_s": cls.BILLED_MINIMUM_S})

    @classmethod
    def cap(cls, *, card_usd: float) -> Cap:
        """What bounds this provider's bill outside this process, refusing a provider with none.

        Pure (the backend's `bound` and the card limit given, no provider call), so the warden
        resolves it once and hands the rent the very cap it checked.

        card_usd: the limit the owner declared for the card paying this provider (`Grants.card`),
            0 while none; read only for a `CARD` bound.
        """
        if cls.bound is not Bound.CARD:
            return Cap(provider=cls.name, bound=cls.bound)
        if not card_usd:
            raise MissionError(
                f"{cls.name} bills a card after the fact, and that card's limit is the only cap "
                f"on the bill: the owner declares it with `{_declaring(cls.name)}` from a "
                "terminal, or rent from a prepaid or threshold-capped provider"
            )
        return Cap(provider=cls.name, bound=Bound.CARD, limit_usd=card_usd)

    @classmethod
    def carded(cls) -> list[str]:
        """The kinds whose bill only the limit of the card paying them caps, sorted."""
        return sorted(name for name in cls.names() if cls.find(name).bound is Bound.CARD)

    def admit(self, plan: ExecutionPlan, resources: Resources, *, cap: Cap) -> None:
        """Every refusal a metered dispatch owes before it reaches a provider's API.

        Every `submit` and `rent` opens with it, so a house rule on what may be rented is one
        edit. Each refusal is free here and otherwise costs a whole rental, billed from the boot.

        cap: the bound the warden resolved for this rental, which a budget may not exceed.
        """
        if not resources.max_usd:
            raise MissionError("provider dispatch needs an explicit max-usd budget")
        if (left := cap.remaining()) is not None and resources.max_usd > left:
            raise MissionError(
                f"max-usd ${resources.max_usd:.2f} is more than the ${left:.2f} {cap.bound} cap "
                f"on {self.name}; lower max-usd, or the owner raises the card's limit with "
                f"`{_declaring(self.name)}` from a terminal"
            )
        container = plan.container
        if container is None:
            return
        named = image_cuda(container.image)
        if named is not None and named < self.CUDA_FLOOR:
            raise MissionError(
                f"image {container.image!r} names CUDA {named}, below this house's CUDA "
                f"{self.CUDA_FLOOR} floor; rent a CUDA {self.CUDA_FLOOR} or newer image instead"
            )

    def refusal(self, capability: type[Capability], **facts: str) -> str:
        """Why this backend cannot answer `capability`: its own `lacks` advice, else a plain line.

        facts: the `handle` and `path` the advice names.
        """
        advice = self.lacks.get(capability)
        if advice is None:
            return f"the {self.name} backend does not implement {capability.__name__}"
        return advice.format(**facts)


def _declaring(provider: str) -> str:
    """The owner's words declaring the limit of the card paying `provider` (`Grants.card`)."""
    return f"{Project().name} serve grant card {provider} --usd N"


def forgotten(error: HTTPError) -> dict:
    """An empty row for the 404 a gone instance answers, re-raising every other refusal."""
    if error.status != 404:
        raise error
    return {}


def hourly_cap(resources: Resources, *, landing: int = 0) -> float:
    """The hourly ceiling `resources` implies, 0 for a walltime-less request.

    A spend cap bounds an hourly rental only once the job says how long it runs; without a
    walltime the search takes the whole market and leans on `max_usd` alone.

    landing: seconds billed before the job starts (mirror, tool install, environment), 0 for
        a prebuilt container that runs the command the moment it boots.
    """
    seconds = resources.walltime_s
    return resources.max_usd * SECONDS_PER_HOUR / (seconds + landing) if seconds else 0.0


def http_transport(request: Request) -> bytes:
    """Send `request` over urllib and read its body, the default transport of every REST backend.

    The one audited url open in the package: each backend builds its `Request` from a constant
    https root and tests swap this callable out, so no unvetted scheme reaches urllib. The
    deadline is ours, since urllib has none.
    """
    with urlopen(request, timeout=_TIMEOUT_S) as response:  # ruff:ignore[suspicious-url-open-usage]  reason=the package's single audited seam, every caller builds its Request from a constant https root and tests inject a double since=2026-08-18
        return response.read()


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
