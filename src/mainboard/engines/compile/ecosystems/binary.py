import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

from patos import FrozenModel
from pydantic import ValidationError, model_validator

from ....core.errors import MissionError
from ....core.execution.tool import ToolRunner
from .base import Ecosystem

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from ....manifest.schema.spec import Spec


class Archive(FrozenModel):
    """An immutable ZIP coordinate; only its named regular executable is extracted."""

    version: str
    url: str
    sha256: str
    member: str

    @model_validator(mode="after")
    def pinned(self) -> Self:
        location = urlsplit(self.url)
        member = PurePosixPath(self.member)
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
                "binary requires a version, HTTPS URL, SHA-256 and relative ZIP member"
            )
        return self


class Binary(Ecosystem):
    """Platform-scoped, hash-pinned executable ZIPs, without a system package manager.

    Installation is explicit. Activation and linting only read installed executables, and a
    missing tool remains a failed check. The manifest pins every downloaded byte.
    """

    toolchain: ClassVar[str] = "binary"
    shared: ClassVar[bool] = True

    @property
    def directory(self) -> Path:
        return self.out / "binary"

    @staticmethod
    def _curl_checked() -> None:
        """Refuse a curl older than 8.4, which cannot bound a streaming download."""
        version = ToolRunner(timeout=5).run("curl", ("--version",))
        parsed = re.match(r"curl (\d+)\.(\d+)\.(\d+)", version.stdout)
        if (
            version.returncode
            or version.truncated
            or parsed is None
            or tuple(map(int, parsed.groups())) < (8, 4, 0)
        ):
            raise MissionError("binary setup requires curl 8.4+ for bounded streaming downloads")

    @staticmethod
    def _download(archive: Archive, destination: Path) -> None:
        Binary._curl_checked()
        result = ToolRunner(timeout=125, max_bytes=16 << 10).run(
            "curl",
            (
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--max-redirs",
                "5",
                "--connect-timeout",
                "30",
                "--max-time",
                "120",
                "--max-filesize",
                str(128 << 20),
                "--output",
                str(destination),
                "--url",
                archive.url,
            ),
        )
        if result.returncode or result.truncated:
            raise MissionError(f"binary download failed: {result.stderr or result.returncode}")
        try:
            size, digest = destination.stat().st_size, _digest(destination)
        except OSError as fault:
            raise MissionError(f"cannot read binary download: {fault}") from fault
        if size > 128 << 20:
            raise MissionError("binary download exceeded its size limit")
        if digest != archive.sha256:
            raise MissionError("binary archive SHA-256 does not match its manifest pin")

    @staticmethod
    def _extract(archive: Archive, downloaded: Path, *, destination: Path) -> None:
        try:
            destination.write_bytes(Binary._member(archive, downloaded))
        except (BadZipFile, OSError, RuntimeError) as fault:
            raise MissionError(f"cannot read binary ZIP: {fault}") from fault

    @staticmethod
    def _member(archive: Archive, downloaded: Path) -> bytes:
        """The bytes of the one member `archive` declares, a regular file under 64 MiB.

        Never a pathname from the archive is extracted, including unrelated traversal entries.
        """
        with ZipFile(downloaded) as zipped:
            matches = [entry for entry in zipped.infolist() if entry.filename == archive.member]
            if len(matches) != 1:
                raise MissionError("binary ZIP must contain exactly one declared member")
            entry = matches[0]
            mode = entry.external_attr >> 16
            if (
                entry.is_dir()
                or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or not 0 < entry.file_size <= 64 << 20
            ):
                raise MissionError("binary ZIP member must be a regular executable under 64 MiB")
            return zipped.read(entry)

    @staticmethod
    def _satisfied(executable: Path, *, receipt: Path, archive: Archive) -> bool:
        try:
            record = json.loads(receipt.read_text(encoding="utf-8"))
        except FileNotFoundError, json.JSONDecodeError:
            return False
        return (
            not executable.is_symlink()
            and executable.is_file()
            and executable.stat().st_mode & 0o111 != 0
            and record == {**archive.model_dump(), "executable_sha256": _digest(executable)}
        )

    @staticmethod
    def archive(name: str, spec: Spec) -> Archive:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name.endswith("."):
            raise MissionError(f"invalid binary executable name {name!r}")
        try:
            return Archive.model_validate({"version": spec.version, **(spec.model_extra or {})})
        except ValidationError as fault:
            raise MissionError(f"invalid [binary] dependency {name!r}: {fault}") from fault

    def _installed(self, archive: Archive, *, executable: Path, receipt: Path) -> None:
        """`archive` downloaded, its member checked and moved into place, its receipt last."""
        with TemporaryDirectory(prefix=".install-", dir=self.directory) as temporary:
            staging = Path(temporary)
            downloaded = staging / "download.zip"
            self._download(archive, downloaded)
            binary = staging / "executable"
            self._extract(archive, downloaded, destination=binary)
            record = staging / "receipt.json"
            record.write_text(
                json.dumps({**archive.model_dump(), "executable_sha256": _digest(binary)}) + "\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            binary.replace(executable)
            record.replace(receipt)

    def _pruned(self, directory: Path, archives: Mapping[str, Archive]) -> None:
        """Remove every executable and receipt the manifest no longer pins."""
        for installed in directory.iterdir():
            if installed.name not in archives:
                installed.unlink()
        for receipt in self.directory.glob("*.json"):
            if receipt.stem not in archives:
                receipt.unlink()

    def binary_dirs(self) -> tuple[Path, ...]:
        return (self.directory / "bin",)

    def frozen_inputs(self) -> tuple[Path, ...]:
        for name, spec in self.deps.items():
            self.archive(name, spec)
        return ()

    def sync(self, *, resolve: bool = False) -> None:
        archives = {name: self.archive(name, spec) for name, spec in self.deps.items()}
        directory = self.binary_dirs()[0]
        if not archives and not self.directory.exists():
            return
        if self.directory.is_symlink() or directory.is_symlink():
            raise MissionError("binary install directories must not be symbolic links")
        directory.mkdir(parents=True, exist_ok=True)
        self._pruned(directory, archives)
        for name, archive in archives.items():
            receipt = self.directory / f"{name}.json"
            if not self._satisfied(directory / name, receipt=receipt, archive=archive):
                self._installed(archive, executable=directory / name, receipt=receipt)


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
