import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import ValidationError

from ....core.errors import MissionError
from ....core.execution.tool import ToolRunner
from ....core.project import Project
from .archive import Archive, Binaries, Receipt, sha256_of
from .base import Ecosystem

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ....core.section import Section
    from ....manifest.schema.spec import Spec
    from .stage import SecondStage


class Binary(Ecosystem):
    """Platform-scoped, hash-pinned executables in ZIP or tar.gz releases, with no package manager.

    Installation is explicit. Activation and linting only read installed executables, and a
    missing tool remains a failed check. The manifest pins every downloaded byte.
    """

    toolchain: ClassVar[str] = "binary"
    shared: ClassVar[bool] = True

    @classmethod
    def of(cls, stage: SecondStage, env: str) -> Self:
        """The binary ecosystem `stage` binds for `env`."""
        return next(eco for eco in stage.ecosystems(env) if isinstance(eco, cls))

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
            size, actual = destination.stat().st_size, sha256_of(destination)
        except OSError as fault:
            raise MissionError(f"cannot read binary download: {fault}") from fault
        if size > 128 << 20:
            raise MissionError("binary download exceeded its size limit")
        if actual != archive.sha256:
            raise MissionError("binary archive SHA-256 does not match its manifest pin")

    @staticmethod
    def _extract(archive: Archive, downloaded: Path, *, destination: Path) -> None:
        destination.write_bytes(archive.format.read(downloaded, archive.member))

    @staticmethod
    def _satisfied(executable: Path, *, receipt: Path, archive: Archive) -> bool:
        try:
            record = Receipt.model_validate_json(receipt.read_bytes())
        except FileNotFoundError, ValidationError:
            return False
        return (
            not executable.is_symlink()
            and executable.is_file()
            and executable.stat().st_mode & 0o111 != 0
            and record == Receipt(**archive.model_dump(), executable_sha256=sha256_of(executable))
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
            downloaded = staging / "download"
            self._download(archive, downloaded)
            binary = staging / "executable"
            self._extract(archive, downloaded, destination=binary)
            record = staging / "receipt.json"
            record.write_text(
                Receipt(
                    **archive.model_dump(), executable_sha256=sha256_of(binary)
                ).model_dump_json()
                + "\n",
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

    def declared(self) -> dict[str, str]:
        """Each pinned tool's version for this platform."""
        return {name: spec.version for name, spec in self.deps.items()}

    def frozen_inputs(self) -> tuple[Path, ...]:
        for name, spec in self.deps.items():
            self.archive(name, spec)
        return ()

    def resolver(self) -> Binaries:
        """Where this environment's tools are: its installed pins, `bin/` and then PATH."""
        return Binaries(self.directory, environment=self.pixi.prefix(self.env).path / "bin")

    def section(self) -> Section | None:
        """The doctor's row over the pinned tools, None when this platform pins none."""
        return self.resolver().section(self.declared(), fix=f"{Project().name} dep binary")

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
