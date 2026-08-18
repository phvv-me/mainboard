"""These Protocols pin an untyped vendor binding's surface; nothing calls them directly.

Importing each module is what "covers" a Protocol's stub method definitions (the `def`
statement itself runs at class-body execution time), so this file exists purely to
exercise the import, matching the convention already used by the top-level
`mainboard.profile.protocols` module.
"""

from mainboard.profile.providers.amd.protocols import Roctx
from mainboard.profile.providers.apple.protocols import IntervalToken, Signposter, SignpostModule
from mainboard.profile.providers.nvidia.protocols import (
    ActivityKind,
    ApiCallbackSite,
    CallbackData,
    CallbackDomain,
    CudaRuntime,
    Cupti,
    Nvtx,
    Subscriber,
)


def test_protocol_modules_import_cleanly() -> None:
    """Every vendor Protocol class is importable, closing the coverage gap in the stubs."""
    assert {
        Roctx,
        IntervalToken,
        Signposter,
        SignpostModule,
        ActivityKind,
        ApiCallbackSite,
        CallbackData,
        CallbackDomain,
        Cupti,
        CudaRuntime,
        Nvtx,
        Subscriber,
    }
