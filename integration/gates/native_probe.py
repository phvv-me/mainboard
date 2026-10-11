"""Prove OS enforcement before any test runs; start with -S so no Python hook can answer for it."""

import errno
import os
import socket
import sys
from collections.abc import Callable
from pathlib import Path

_REFUSALS = {errno.EPERM, errno.EACCES, errno.EROFS, errno.ENETUNREACH}
# A reserved, non-routable destination: refused at once by policy or the net namespace.
_UNREACHABLE = ("192.0.2.1", 22)


def _refused(what: str, action: Callable[[], object]) -> None:
    try:
        action()
    except OSError as error:
        assert error.errno in _REFUSALS, error
    else:
        raise SystemExit(f"M00-OS: native {what} unexpectedly permitted")


def _connect() -> None:
    with socket.socket() as connection:
        connection.settimeout(1)
        connection.connect(_UNREACHABLE)


def _main(root: Path) -> None:
    # The checkout is writable by the user natively and refused by both sandboxes; the temporary
    # directory beside the root is private and writable under bubblewrap, so it proves nothing.
    outside = Path(__file__).parent / f"{root.name}-forbidden"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    what = f"external write (remove {outside})"
    _refused(what, lambda: os.close(os.open(outside, flags, 0o600)))
    _refused("external connection", _connect)
    (root / "os-isolation-proved").write_text(sys.platform, encoding="utf-8")


if __name__ == "__main__":
    _main(Path(sys.argv[1]))
