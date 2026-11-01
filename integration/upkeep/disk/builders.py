import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def streaming_into(folder: Path) -> Generator[subprocess.Popen[bytes]]:
    """A process in a session of its own writing `folder/transfer.zip`, as a leaked ssh does.

    It is killed on exit.
    """
    folder.mkdir(parents=True)
    with (folder / "transfer.zip").open("wb") as sink:
        stream = subprocess.Popen(("sleep", "60"), stdout=sink, start_new_session=True)
    with stream:
        try:
            yield stream
        finally:
            stream.kill()


def workspace_at(root: Path, *, tmp: Path | None = None) -> Path:
    """A workspace at `root` whose manifest names `tmp` as `[workspace] tmp`, a checkout's."""
    (root / ".git").mkdir(parents=True)
    line = f'tmp = "{tmp}"\n' if tmp else ""
    (root / "mb.toml").write_text(f'[workspace]\nname = "w"\n{line}', encoding="utf-8")
    return root
