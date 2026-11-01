"""The lock that serializes the mirror-and-pin transactions of one target."""

import hashlib
from typing import TYPE_CHECKING, Self

from filelock import FileLock

from ..shared import state_path
from ..transport.core import Endpoint

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class SyncLock:
    """Serialize mirror-and-pin transactions to one target across local dispatch processes.

    target: the effective rental endpoint or declared SSH alias being mirrored.
    root: local workspace root that owns the generated state directory, discovered upward from
        the current directory when None, so two processes started in different subdirectories
        still queue behind the same lock file.
    """

    def __enter__(self) -> Self:
        """Wait for this target's mirror lock and hold it until context exit."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the kernel lock even when the mirror raises."""
        if self.lock.is_locked:
            self.lock.release()

    def __init__(self, target: str | Endpoint, root: Path | None = None) -> None:
        identity = (
            target if isinstance(target, str) else f"{target.destination}:{target.port or 22}"
        )
        digest = hashlib.blake2s(identity.encode(), digest_size=8).hexdigest()
        self.path = state_path(root) / "locks" / f"sync-{digest}.lock"
        # A mirror nested inside a dispatch's transaction shares its reentrant lock instance.
        self.lock = FileLock(self.path.resolve(), is_singleton=True)
