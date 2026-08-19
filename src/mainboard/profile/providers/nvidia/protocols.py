# Structural contracts for the untyped CUPTI, NVTX, and CUDA runtime bindings.

from collections.abc import Callable
from typing import Protocol

from ...protocols import RawActivity


class ActivityKind(Protocol):
    """The `cupti.ActivityKind` enum: members are selected by name with `getattr`."""

    RUNTIME: int
    DRIVER: int


class CallbackDomain(Protocol):
    """The `cupti.CallbackDomain` enum members the tracer routes records and callbacks by."""

    RUNTIME_API: int
    DRIVER_API: int


class ApiCallbackSite(Protocol):
    """The `cupti.ApiCallbackSite` enum: distinguishes the API-enter from the API-exit edge."""

    API_ENTER: int


class CallbackData(Protocol):
    """The CUPTI callback payload the tracer reads: which edge fired and the API name."""

    callback_site: int
    function_name: str


class Cupti(Protocol):
    """The `cupti.cupti` functions and enums the Activity + Callback collectors use.

    `cupti-python` ships no stubs; this pins the asynchronous Activity API surface
    (register, enable/disable, flush) and the synchronous Callback API surface
    (subscribe, enable_domain) that the tracer drives. `subscribe` returns an opaque
    subscriber token, threaded back into `enable_domain`/`unsubscribe` and never inspected.
    """

    ActivityKind: ActivityKind
    CallbackDomain: CallbackDomain
    ApiCallbackSite: ApiCallbackSite

    def activity_disable(self, kind: int) -> None: ...

    def activity_enable(self, kind: int) -> None: ...

    def activity_flush_all(self, flag: int) -> None: ...

    def activity_register_callbacks(
        self,
        on_requested: Callable[[], tuple[int, int]],
        on_completed: Callable[[list[RawActivity]], None],
    ) -> None: ...

    def enable_domain(self, enable: int, subscriber: Subscriber, domain: int) -> None: ...

    def get_callback_name(self, domain: int, cbid: int) -> str: ...

    def get_timestamp(self) -> int: ...
    def subscribe(
        self, callback: Callable[[None, int, int, CallbackData], None], userdata: None
    ) -> Subscriber: ...
    def unsubscribe(self, subscriber: Subscriber) -> None: ...


class Subscriber(Protocol):
    """An opaque CUPTI callback-subscriber token, only threaded back into CUPTI calls."""


class Nvtx(Protocol):
    """The `nvtx` annotation surface the tracer emits."""

    def end_range(self, range_id: tuple[int, int]) -> None: ...

    def mark(self, message: str) -> None: ...

    def pop_range(self) -> None: ...

    def push_range(self, message: str) -> None: ...
    def start_range(self, message: str) -> tuple[int, int]: ...


class CudaRuntime(Protocol):
    """The one `cuda.bindings.runtime` function the tracer calls: the device sync barrier."""

    def cudaDeviceSynchronize(self) -> tuple[int]: ...
