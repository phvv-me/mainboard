"""The official DuckDB command line the syntax trees are parsed with where the engine's own DuckDB
cannot load `sitting_duck`.

The community repository builds the extension for DuckDB releases, and the engine runs a
pre-release (`duckdb==2.0.0.dev...`), so a machine without a build of its own parses through the
release the repository serves. The first use downloads that release's command line from DuckDB's
GitHub releases, checks the archive against the digest pinned here (the one the release publishes
for the asset), keeps the binary in the cache and installs the extension for it.
"""

import gzip
import hashlib
import os
import platform
import subprocess  # ruff:ignore[suspicious-subprocess-import]  reason=runs the pinned DuckDB CLI with fixed argv since=2026-10-16
import sys
import urllib.request
from pathlib import Path
from typing import ClassVar

from filelock import FileLock
from patos import FrozenModel
from pydantic import Field

from ...core.errors import MissionError
from ...log import logger


class DuckCli(FrozenModel):
    """A pinned DuckDB release's command line, fetched on first use.

    version: the release tag, whose archive digests `DIGESTS` holds.
    folder: where releases are kept, one folder each.
    """

    version: str = "v1.5.6"
    folder: Path = Field(default_factory=lambda: Path.home() / ".cache/mainboard/duckdb")

    # The sha256 of each release archive (`duckdb_cli-<platform>.gz`), as the release lists it.
    DIGESTS: ClassVar[dict[str, str]] = {
        "osx-arm64": "7f19ac71f0a4bde308a247ac556d93ee95c03adc327ec85eeab9113294f11dd5",
        "osx-amd64": "388ce7e8a96307d040a27709a4c193f455ad1d3bbcd576c118bf1ae8158973e7",
        "linux-arm64": "1af0541e649a3ae34eb20eec2ca564308bf0e8993e5d6dba6be77bfb289a2718",
        "linux-amd64": "cce2687a38681db644c248198b99ecd86170ff47e324c5f33e50c0042f142839",
    }
    RELEASES: ClassVar[str] = "https://github.com/duckdb/duckdb/releases/download"
    TIMEOUT_SECONDS: ClassVar[int] = 120
    # The binaries this process has fetched and installed the extension for.
    PREPARED: ClassVar[set[Path]] = set()

    @property
    def path(self) -> Path:
        """Where the binary lives, whether or not it is fetched yet."""
        return self.folder / self.version / "duckdb"

    def ready(self) -> Path:
        """The binary, fetched with `sitting_duck` installed for it on the process's first call, so
        a clean machine needs nothing done beforehand. Processes take turns on the cache folder."""
        if self.path not in self.PREPARED:
            self.folder.mkdir(parents=True, exist_ok=True)
            with FileLock(self.folder / f"{self.version}.lock"):
                if not self.path.is_file():
                    self._fetch()
                self._install()
            self.PREPARED.add(self.path)
        return self.path

    def _fetch(self) -> None:
        """Download the release archive, check its digest and unpack the binary into place."""
        kind = self._platform()
        url = f"{self.RELEASES}/{self.version}/duckdb_cli-{kind}.gz"
        logger.info("fetching the DuckDB command line from {}", url)
        request = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(request, timeout=self.TIMEOUT_SECONDS) as reply:  # ruff:ignore[suspicious-url-open-usage]  reason=a constant https release root of the pinned tag since=2026-10-16
                archive = reply.read()
        except OSError as error:
            raise MissionError(f"{url} would not answer: {error}") from None
        if (found := hashlib.sha256(archive).hexdigest()) != self.DIGESTS[kind]:
            raise MissionError(f"{url} has sha256 {found}, not the pinned {self.DIGESTS[kind]}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_suffix(f".{os.getpid()}")
        scratch.write_bytes(gzip.decompress(archive))
        scratch.chmod(0o755)
        scratch.replace(self.path)

    def _install(self) -> None:
        """Install `sitting_duck` from the community repository into the binary's extension folder
        (a no-op once there) and load it, so a failure shows now rather than in a parse."""
        try:
            subprocess.run(
                [str(self.path), "-c", "INSTALL sitting_duck FROM community; LOAD sitting_duck;"],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as error:
            raise MissionError(f"sitting_duck would not install: {error.stderr.strip()}") from None

    @classmethod
    def _platform(cls) -> str:
        """The release asset name's platform for this machine, e.g. `osx-arm64`."""
        system = {"darwin": "osx", "linux": "linux"}.get(sys.platform)
        machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(
            platform.machine().lower()
        )
        if f"{system}-{machine}" not in cls.DIGESTS:
            raise MissionError(
                f"no DuckDB command line is pinned for {sys.platform}/{platform.machine()}"
            )
        return f"{system}-{machine}"
