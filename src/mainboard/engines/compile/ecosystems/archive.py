"""Third-party executables that neither conda-forge nor PyPI ships at the version we need.

A pinned release archive (`Archive`: version, HTTPS URL, SHA-256, the one member to keep) is
fetched by the `Binary` ecosystem and verified before any byte of it is read as an executable.
`Binaries` finds an executable afterwards, an installed pin first and then the environment's
`bin/` and PATH, and judges it against the version the manifest asks for, so a tool conda-forge
does ship (process-compose, jj) is checked by the same `check` as one fetched here. A pinned tool
is trusted only from its verified install (`pinned`): a same-version twin elsewhere meets no pin.
"""

import hashlib
import re
import shutil
import stat
import tarfile
import zlib
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion
from patos import FrozenModel
from pydantic import ValidationError, model_validator

from ....core.errors import MissionError
from ....core.execution.tool import ToolRunner
from ....core.project import Project
from ....core.section import Section
from ....enums import Source, Verdict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_LIMIT = 256 << 20
"""The most one extracted member may hold, bytes."""

_FAULTS = (BadZipFile, tarfile.TarError, EOFError, OSError, RuntimeError, zlib.error)
_NUMBER = re.compile(r"\d+(?:\.\d+)+")
_PROBE_SECONDS = 10


def sha256_of(path: Path) -> str:
    """The SHA-256 of the file at `path`, as hex."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def accepts(wanted: str, *, version: str) -> bool:
    """Whether `version` meets `wanted`: `*` anything, a bare version exactly, else a range.

    wanted: a manifest version, `0.7.3` for a pin or `>=0.46, <0.47` for a conda range.
    version: what an executable reported; one that is not a version meets nothing but `*`.
    """
    if wanted in ("", "*"):
        return True
    specifier = f"=={wanted}" if wanted[0].isdigit() else wanted
    try:
        return SpecifierSet(specifier).contains(version, prereleases=True)
    except InvalidSpecifier, InvalidVersion:
        return False


class Format(StrEnum):
    """How a release archive is packed, told by the file name its URL ends in."""

    TAR = auto()
    ZIP = auto()

    @classmethod
    def of(cls, url: str) -> Format:
        """The format of the archive at `url`, a ValueError when no reader exists for it."""
        name = urlsplit(url).path.lower()
        if name.endswith(".zip"):
            return cls.ZIP
        if name.endswith((".tar.gz", ".tgz")):
            return cls.TAR
        raise ValueError(f"no reader for the archive {name!r}")

    @staticmethod
    def _irregular() -> MissionError:
        return MissionError(
            f"binary archive member must be a regular executable under {_LIMIT >> 20} MiB"
        )

    @staticmethod
    def _only[Entry](entries: Sequence[Entry]) -> Entry:
        """The one entry that matched the declared member."""
        if len(entries) != 1:
            raise MissionError("binary archive must contain exactly one declared member")
        return entries[0]

    @staticmethod
    def _tarred(archive: Path, member: str) -> bytes:
        with tarfile.open(archive, "r:gz") as packed:
            wanted = PurePosixPath(member)
            entry = Format._only(
                [e for e in packed.getmembers() if PurePosixPath(e.name) == wanted]
            )
            source = packed.extractfile(entry) if entry.isreg() else None
            if source is None or not 0 < entry.size <= _LIMIT:
                raise Format._irregular()
            return source.read()

    @staticmethod
    def _zipped(archive: Path, member: str) -> bytes:
        with ZipFile(archive) as packed:
            entry = Format._only([e for e in packed.infolist() if e.filename == member])
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if (
                entry.is_dir()
                or kind not in (0, stat.S_IFREG)
                or not 0 < entry.file_size <= _LIMIT
            ):
                raise Format._irregular()
            return packed.read(entry)

    def read(self, archive: Path, member: str) -> bytes:
        """The bytes of the one regular file `member` names, never a pathname from `archive`.

        Nothing else in the archive is extracted, so an entry that climbs out of it is inert.
        """
        read = self._zipped if self is Format.ZIP else self._tarred
        try:
            return read(archive, member)
        except _FAULTS as fault:
            raise MissionError(f"cannot read binary {self.value.upper()}: {fault}") from fault


class Archive(FrozenModel):
    """An immutable release coordinate; only its named regular executable is extracted."""

    version: str
    url: str
    sha256: str
    member: str

    @model_validator(mode="after")
    def pinned(self) -> Self:
        location = urlsplit(self.url)
        member = PurePosixPath(self.member)
        Format.of(self.url)
        if (
            not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z_.+-]*", self.version)
            or location.scheme != "https"
            or not location.hostname
            or location.username is not None
            or location.password is not None
            or location.fragment
            or not re.fullmatch(r"[a-f0-9]{64}", self.sha256)
            or member.is_absolute()
            or ".." in member.parts
            or "\\" in self.member
            or not member.name
        ):
            raise ValueError(
                "binary requires a version, HTTPS URL, SHA-256 and relative archive member"
            )
        return self

    @property
    def format(self) -> Format:
        return Format.of(self.url)


class Receipt(Archive):
    """What an install leaves beside the executable: its pin and the digest of what it wrote."""

    executable_sha256: str


class Found(FrozenModel):
    """An executable located, with the version it answers to.

    version: the pinned release for an installed one, what `--version` printed otherwise,
        empty when it printed no version.
    """

    name: str
    path: Path
    version: str
    source: Source


class Binaries:
    """Where a tool's executable is, and whether it is the version asked for."""

    @staticmethod
    def _answered(path: Path) -> str:
        """The first version number `path --version` prints, else `path version`, else empty.

        A Go tool built on cobra (process-compose) knows only the `version` subcommand.
        """
        runner = ToolRunner(timeout=_PROBE_SECONDS, max_bytes=16 << 10)
        for asked in (("--version",), ("version",)):
            said = runner.run(str(path), asked)
            if found := _NUMBER.search(f"{said.stdout}\n{said.stderr}"):
                return found[0]
        return ""

    @staticmethod
    def _judged(found: Found, wanted: str) -> Found:
        """`found` when it meets `wanted`, else a MissionError naming both versions."""
        if not accepts(wanted, version=found.version):
            raise MissionError(
                f"{found.name} {found.version or 'of unknown version'} at {found.path}, "
                f"wanted {wanted}"
            )
        return found

    def __init__(self, installed: Path, *, environment: Path) -> None:
        """installed: the `Binary` ecosystem's directory, holding `bin/` and the receipts.

        environment: the active environment's `bin/`, searched before PATH because a tool run
            outside an activation does not have it on PATH.
        """
        self.installed = installed
        self.environment = environment

    def _pinned(self, name: str) -> Found | None:
        """The installed pin of `name`, None when none was installed, a refusal when altered."""
        executable = self.installed / "bin" / name
        try:
            receipt = Receipt.model_validate_json((self.installed / f"{name}.json").read_bytes())
        except FileNotFoundError, ValidationError:
            return None
        if executable.is_symlink() or not executable.is_file():
            return None
        if sha256_of(executable) != receipt.executable_sha256:
            raise MissionError(f"{executable} differs from the receipt of its install")
        return Found(name=name, path=executable, version=receipt.version, source=Source.INSTALLED)

    def check(self, name: str, *, wanted: str = "*") -> Found:
        """The executable for `name` meeting `wanted`, or a MissionError saying what is wrong."""
        found = self.find(name)
        if found is None:
            raise MissionError(f"{name} is not installed")
        return Binaries._judged(found, wanted)

    def find(self, name: str) -> Found | None:
        """The executable for `name`: its installed pin, else the environment's, else PATH's."""
        if (pinned := self._pinned(name)) is not None:
            return pinned
        for source, directory in (
            (Source.ENVIRONMENT, str(self.environment)),
            (Source.PATH, None),
        ):
            if located := shutil.which(name, path=directory):
                return Found(
                    name=name,
                    path=Path(located),
                    version=self._answered(Path(located)),
                    source=source,
                )
        return None

    def pinned(self, name: str, *, wanted: str = "*") -> Found:
        """The install of `name` from its verified archive, meeting `wanted`.

        A twin in `bin/` or on PATH never counts. A MissionError says what is wrong: no install
        from the archive, an executable altered since, or a version off `wanted`.
        """
        if (found := self._pinned(name)) is None:
            raise MissionError(
                f"{name} is pinned but not installed from its archive; "
                f"run `{Project().name} dep binary`"
            )
        return Binaries._judged(found, wanted)

    def rows(self, wanted: Mapping[str, str], *, fix: str) -> list[Section]:
        """One row per pinned tool in `wanted`: its install at the version it needs, or why not."""
        rows = []
        for name, version in wanted.items():
            try:
                found = self.pinned(name, wanted=version)
            except MissionError as fault:
                rows.append(
                    Section(section=name, verdict=Verdict.FAIL, detail=str(fault), fix=fix)
                )
            else:
                detail = f"{found.version or 'unversioned'} at {found.path} ({found.source})"
                rows.append(Section(section=name, verdict=Verdict.PASS, detail=detail))
        return rows

    def section(self, wanted: Mapping[str, str], *, fix: str) -> Section | None:
        """The one `binaries` row of the doctor over `wanted`, None when nothing is wanted."""
        rows = self.rows(wanted, fix=fix)
        if not rows:
            return None
        broken = [row for row in rows if row.verdict is Verdict.FAIL]
        return Section(
            section="binaries",
            verdict=Verdict.FAIL if broken else Verdict.PASS,
            detail="; ".join(f"{row.section}: {row.detail}" for row in broken or rows),
            fix=fix if broken else "",
        )
