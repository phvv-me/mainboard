"""A cloud's REST API as the four calls a `CloudBackend` rents through, under its bearer key."""

import abc
import hashlib
import json
from typing import TYPE_CHECKING, ClassVar
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request

from packaging.version import Version
from pydantic import TypeAdapter

from ..base import Credentials, http_transport
from ..capabilities import Rented
from ..exceptions import CapacityGone

if TYPE_CHECKING:
    import re
    from collections.abc import Iterable, Mapping

    from gpuhunt import CatalogItem
    from pydantic import JsonValue

    from ...hosts.lease import Lease
    from ..base import Transport

# What every request says of itself.
_SAYS = {"Content-Type": "application/json", "User-Agent": "mainboard"}


class Machine(Rented):
    """One machine as its cloud reports it.

    status: `pending` until it can take ssh, `running`, or `gone` once ended or lost.
    host / port / user: where ssh reaches it, empty until the cloud publishes an address.
    """

    status: str
    host: str = ""
    port: int = 22
    user: str = "root"


class CloudApi(abc.ABC):
    """One cloud's machines: created for a gpuhunt offer, read, listed and ended.

    transport: sends a prepared `Request`, returning its response, injectable for tests.
    """

    # The variables its key may be set in, and where such a key comes from.
    key_variables: ClassVar[tuple[str, ...]] = ()
    hint: ClassVar[str] = ""
    # What its refusals say, lowercased, when the offer chosen has no machine left.
    exhausted: ClassVar[tuple[str, ...]] = ()
    # The statuses that mean the same whatever the body says, each with its message (`{reason}`).
    exhausted_statuses: ClassVar[Mapping[int, str]] = {}
    # Where one machine is read and ended (`{}` takes its quoted handle), and the key its answer
    # holds it under, none when the answer is the machine itself.
    machine_url: ClassVar[str] = ""
    machine_key: ClassVar[str] = ""

    @abc.abstractmethod
    def create(
        self, offer: CatalogItem, *, lease: Lease, name: str, public_key: str, disk_gb: int
    ) -> str:
        """Start a machine for `offer` with `public_key` authorized; its handle.

        Raises `CapacityGone` when that offer can no longer be rented, so the next is tried.

        lease: the terms accepted for it, whose deadline and budget a cloud that ends machines
            itself is sent with the create, so a machine nothing here releases still stops.
        """

    @abc.abstractmethod
    def machines(self) -> list[Machine]:
        """Every machine this account holds on the cloud, rented here or not."""

    @staticmethod
    @abc.abstractmethod
    def _machine(found: Mapping) -> Machine:
        """One machine as the cloud's answer describes it."""

    @staticmethod
    def _named(public_key: str) -> str:
        """The name this machine's key is registered under, fixed by the key itself."""
        return "mainboard-" + hashlib.sha256(public_key.strip().encode()).hexdigest()[:12]

    @staticmethod
    def _newest(names: Iterable[str], *, pattern: re.Pattern[str]) -> str | None:
        """The first of `names` that `pattern` matches with the newest versions, None if none do.

        pattern: its version groups, `cuda`, `driver` (if any) and `ubuntu`, rank in name order.
        """
        newest = max(
            (match for name in names if (match := pattern.match(name))),
            key=lambda each: [Version(text) for _, text in sorted(each.groupdict().items())],
            default=None,
        )
        return None if newest is None else newest.string

    @staticmethod
    def _same(known: str, *, public_key: str) -> bool:
        """Whether the registered key text `known` is `public_key`, whatever its comment says."""
        return known.split()[:2] == public_key.split()[:2]

    def __init__(self, *, transport: Transport = http_transport) -> None:
        self.transport = transport

    def _authorized(self) -> dict[str, str]:
        """The header this cloud authenticates a request by, a bearer token unless it says else."""
        return {"Authorization": f"Bearer {self.bearer()}"}

    def _call[T](
        self,
        method: str,
        *,
        url: str,
        answer: type[T],
        body: Mapping[str, JsonValue] | None = None,
    ) -> T:
        """One JSON request under this cloud's bearer token, its answer read as `answer`.

        An empty answer reads as `{}`. HTTP errors propagate, so a caller reads the refusal
        (capacity, a gone handle) itself.
        """
        return TypeAdapter(answer).validate_json(self._send(method, url=url, body=body) or b"{}")

    def _ended(
        self, method: str, *, url: str, body: Mapping[str, JsonValue] | None = None
    ) -> None:
        """A request that ends a machine, tolerating one the cloud no longer has (404)."""
        self._fetched(url, method=method, body=body)

    def _exhausted(self, refused: HTTPError) -> None:
        """Raise `CapacityGone` when `refused` says the offer has no machine left."""
        if (message := self.exhausted_statuses.get(refused.code)) is not None:
            raise CapacityGone(message.format(reason=refused.reason)) from refused
        said = refused.read().decode(errors="replace")
        if any(word in said.lower() for word in self.exhausted):
            raise CapacityGone(said[:200]) from refused

    def _fetched(
        self, url: str, *, method: str = "GET", body: Mapping[str, JsonValue] | None = None
    ) -> dict | None:
        """The JSON object `method` gets at `url`, None once the cloud says it is gone (404)."""
        try:
            return self._call(method, url=url, answer=dict, body=body)
        except HTTPError as error:
            if error.code != 404:
                raise
            return None

    def _launched(self, *, url: str, body: Mapping[str, JsonValue]) -> dict:
        """The answer to the create posted to `url`; `CapacityGone` once no machine is left."""
        try:
            return self._call("POST", url=url, answer=dict, body=body)
        except HTTPError as refused:
            self._exhausted(refused)
            raise

    def _locate(self, handle: str) -> str:
        """The URL `handle` is read and ended at."""
        return self.machine_url.format(quote(handle, safe=""))

    def _secret(self, variables: tuple[str, ...]) -> str:
        """The first of `variables` set in the environment or `.env`; this cloud's hint if none."""
        return Credentials().key(*variables, hint=self.hint)

    def _send(
        self,
        method: str,
        *,
        url: str,
        body: Mapping[str, JsonValue] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        """One JSON request under this cloud's key, its raw answer.

        A cloud that answers a create with a bare id rather than JSON reads it from here, and
        one that trades its key for a token sends that call under `headers` of its own.
        """
        if headers is None:
            headers = {**self._authorized(), "Accept": "application/json"}
        data = None if body is None else json.dumps(body).encode()
        return self.transport(Request(url, data, {**headers, **_SAYS}, method=method)).strip()

    def bearer(self) -> str:
        """The token every request sends: the API key itself, unless a cloud exchanges it first."""
        return self.key()

    def key(self) -> str:
        """This cloud's API key from the environment or the workspace `.env`."""
        return self._secret(self.key_variables)

    def machine(self, handle: str) -> Machine:
        """`handle` as the cloud reports it now; `gone` when the cloud no longer knows it."""
        if (found := self._fetched(self._locate(handle))) is None:
            return Machine(handle=handle, status="gone")
        return self._machine(found[self.machine_key] if self.machine_key else found)

    def offers(self) -> list[CatalogItem]:
        """The offers this cloud lists itself, for a cloud gpuhunt publishes no catalog of."""
        return []

    def terminate(self, handle: str) -> None:
        """End `handle`, tolerating one already gone."""
        self._ended("DELETE", url=self._locate(handle))
