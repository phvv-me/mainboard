import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def replacing(path: Path) -> Generator[Path]:
    """A fresh sibling of `path` to write, renamed over `path` when the block ends cleanly.

    Any other exit removes it, so a failed write leaves `path` as it was and no stray file.
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with ExitStack() as cleanup:
        cleanup.callback(temporary.unlink, missing_ok=True)
        yield temporary
        temporary.replace(path)


def write_atomic(
    path: Path, content: str | bytes, *, mode: int | None = None, durable: bool = False
) -> None:
    """Replace `path` with `content` in one rename, text encoded as UTF-8 without newline changes.

    mode: the permission bits whatever the umask, the umask's own when None.
    durable: fsync before the rename, so a crash never leaves an empty file under the new name.
    """
    with replacing(path) as temporary, temporary.open("xb") as stream:
        if mode is not None:
            os.fchmod(stream.fileno(), mode)
        stream.write(content.encode("utf-8") if isinstance(content, str) else content)
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
