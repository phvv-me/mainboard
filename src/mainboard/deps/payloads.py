"""What each release index answers, validated as it arrives; only the fields read are kept."""

from collections.abc import Mapping
from contextlib import suppress

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version
from patos import FrozenOpenModel
from pydantic import Field


class Crate(FrozenOpenModel):
    """The `crate` half of a crates.io crate document."""

    max_stable_version: str = ""


class CrateDocument(FrozenOpenModel):
    """A crates.io crate document."""

    crate: Crate = Field(default_factory=Crate)


class File(FrozenOpenModel):
    """One file of a PEP 691 listing."""

    filename: str
    yanked: bool | str = False

    @property
    def version(self) -> Version | None:
        """The release this wheel or sdist belongs to, None for a name neither spells."""
        with suppress(InvalidWheelFilename):
            return parse_wheel_filename(self.filename)[1]
        with suppress(InvalidSdistFilename):
            return parse_sdist_filename(self.filename)[1]
        return None


class Latest(FrozenOpenModel):
    """The Go module proxy's `@latest` answer."""

    version: str = Field(default="", alias="Version")


class Listing(FrozenOpenModel):
    """A PEP 691 project page: its files and the versions it names."""

    files: tuple[File, ...] = ()
    versions: tuple[str, ...] = ()

    @staticmethod
    def _offered(version: str, kept: Mapping[Version, bool]) -> bool:
        """Whether `version` still offers a file; one `kept` does not mention is taken to."""
        try:
            return kept.get(Version(version), True)
        except InvalidVersion:
            return True

    def unyanked(self) -> list[str]:
        """The versions that keep a file not yanked.

        A release whose every file was yanked is withdrawn, and a requirement raised to one cannot
        solve (pytest-timeout 2.5.0, 2026-10-23). A version with no file this reads keeps its
        place.
        """
        kept: dict[Version, bool] = {}
        for file in self.files:
            if (version := file.version) is not None:
                kept[version] = kept.get(version, False) or not file.yanked
        return [version for version in self.versions if Listing._offered(version, kept)]


class Package(FrozenOpenModel):
    """An npm package document: its dist-tags and when each version was published."""

    dist_tags: dict[str, str] = Field(default_factory=dict, alias="dist-tags")
    time: dict[str, str] = Field(default_factory=dict)


class Release(FrozenOpenModel):
    """One release record of a conda channel."""

    version: str
