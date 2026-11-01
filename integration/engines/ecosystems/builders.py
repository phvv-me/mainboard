import io
import stat
import tarfile
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from mainboard.engines.compile.ecosystems.archive import Receipt, sha256_of

from .constants import CONTENT, URL


def executable(directory: Path, name: str, *, said: str) -> Path:
    """A shell script in `directory` that prints `said` whatever it is asked."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!/bin/sh\necho '{said}'\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def installed(root: Path, name: str, *, version: str, content: bytes = CONTENT) -> Path:
    """What `binary sync` leaves under `root`: the executable in `bin/` and its receipt."""
    script = root / "bin" / name
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(content)
    script.chmod(0o755)
    receipt = Receipt(
        version=version, url=URL, sha256="a" * 64, member=name, executable_sha256=sha256_of(script)
    )
    (root / f"{name}.json").write_text(receipt.model_dump_json(), encoding="utf-8")
    return script


def tarball(path: Path, *entries: tarfile.TarInfo, files: dict[str, bytes] | None = None) -> Path:
    """A tar.gz at `path` holding `files` as regular executables, then `entries` as given."""
    with tarfile.open(path, "w:gz") as packed:
        for name, data in (files or {}).items():
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o755
            packed.addfile(member, io.BytesIO(data))
        for entry in entries:
            packed.addfile(entry)
    return path


def link(name: str, *, kind: bytes = tarfile.SYMTYPE) -> tarfile.TarInfo:
    """A tar entry that is a link (or, with another `kind`, a directory), never a regular file."""
    entry = tarfile.TarInfo(name)
    entry.type, entry.linkname = kind, "elsewhere"
    return entry


def zipped(path: Path, name: str, data: bytes, *, mode: int = stat.S_IFREG | 0o755) -> Path:
    """A ZIP at `path` holding one entry `name` of the given Unix `mode`, and a traversal entry."""
    with ZipFile(path, "w") as packed:
        entry = ZipInfo(name)
        entry.external_attr = mode << 16
        packed.writestr(entry, data)
        packed.writestr("../outside", b"never extracted")
    return path
