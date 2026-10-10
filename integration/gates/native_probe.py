# Run with -S: prove OS enforcement before importing pytest or running any test code.
import errno
import os
import socket
import sys
from pathlib import Path

root = Path(sys.argv[1])
outside = root.parent / f"{root.name}-forbidden"
try:
    descriptor = os.open(outside, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except OSError as error:
    assert error.errno in {errno.EPERM, errno.EACCES, errno.EROFS}, error
else:
    os.close(descriptor)
    outside.unlink()
    raise SystemExit("M00-OS: native external write unexpectedly permitted")

# A reserved, non-routable destination must be refused immediately by policy or net namespace.
with socket.socket() as connection:
    connection.settimeout(1)
    try:
        connection.connect(("192.0.2.1", 22))
    except OSError as error:
        assert error.errno in {errno.EPERM, errno.EACCES, errno.ENETUNREACH}, error
    else:
        raise SystemExit("M00-OS: native external connection unexpectedly permitted")

(root / "os-isolation-proved").write_text(sys.platform, encoding="utf-8")
