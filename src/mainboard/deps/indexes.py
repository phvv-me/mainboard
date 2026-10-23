import abc
import json
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version
from patos import Registry

from ..core.errors import MissionError
from ..engines.compile.backend import PixiEngine, Process

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..manifest.schema.spec import Json

_TIMEOUT = 20.0

# PEP 691's machine-readable listing, served by every PEP 503 index beside its HTML.
_JSON = "application/json"
_SIMPLE = "application/vnd.pypi.simple.v1+json"

_PYPI = "https://pypi.org/simple"
_NPM = "https://registry.npmjs.org"
# pnpm's default minimumReleaseAge, which its frozen install enforces on every host.
_RELEASE_AGE = timedelta(days=1)
_CRATES = "https://crates.io/api/v1/crates"
_GOPROXY = "https://proxy.golang.org"


class Index(Registry, abc.ABC):
    """Where one ecosystem publishes its releases, and how its own resolver spells a pin.

    Implementations enroll keyed by their manifest table, so `-l rust` reaches crates.io.
    """

    def __init__(self, sources: Sequence[str] = ()) -> None:
        """sources: the declared conda channels or Python index url, empty for the public one."""
        self.sources = tuple(sources)

    @classmethod
    def of(cls, ecosystem: str) -> Index:
        """The index for `ecosystem`, refusing an unknown one with the roster."""
        try:
            return cls.find(ecosystem)()
        except KeyError:
            raise MissionError(
                f"no release index for {ecosystem!r}; pass an explicit version, or one of "
                f"{sorted(cls.names())}"
            ) from None

    @abc.abstractmethod
    def latest(self, name: str) -> str:
        """The newest published release of `name`, as the index itself reports it."""

    def pin(self, version: str) -> str:
        """pixi's semver pin (`1.2.3` -> `>=1.2.3, <2`, `0.1.0` -> `>=0.1.0, <0.2`).

        An unreadable release (a conda date stamp) gets only a floor, which still resolves, and so
        does a lone date (`20260107`), whose next number is tomorrow, not a breaking release.
        """
        try:
            release = list(Version(version).release)
        except InvalidVersion:
            return f">={version}"
        if len(release) == 1 and len(str(release[0])) >= len("YYYYMMDD"):
            return f">={version}"
        carried = next((at for at, part in enumerate(release) if part), len(release) - 1)
        ceiling = [*release[:carried], release[carried] + 1]
        return f">={version}, <{'.'.join(str(part) for part in ceiling)}"


class Conda(Index):
    """The conda channels this workspace declares, read through pixi's own index reader."""

    def latest(self, name: str) -> str:
        channels = [flag for channel in self.sources for flag in ("--channel", channel)]
        command = PixiEngine().command["search", "--json", *channels, name]
        found = cast("dict[str, list[dict[str, str]]]", json.loads(Process.output(command, name)))
        versions = [record["version"] for records in found.values() for record in records]
        return _newest(versions, name=name, where="the declared conda channels")


class Python(Index):
    """A PEP 503 index, listing its releases through the PEP 691 JSON the same URL serves."""

    def latest(self, name: str) -> str:
        index = self.sources[0] if self.sources else _PYPI
        # An extra rides the requirement, never the page: `numba-cuda-mlir[cu13]` is a 404.
        url = f"{index.rstrip('/')}/{name.partition('[')[0]}/"
        listing = cast("dict[str, list[Json]]", _fetched(url, accept=_SIMPLE))
        return _newest(_unyanked(listing), name=name, where=index)


class Nodejs(Index):
    """The npm registry, whose per-package document carries its own dist-tags."""

    def latest(self, name: str) -> str:
        """The `latest` tag once it is older than pnpm's minimum release age, else the newest
        release below it that is: every host installs the lock frozen, and a frozen pnpm refuses a
        release younger than a day (firecrawl-mcp 3.34.1, 22 hours old, 2026-10-23)."""
        document = cast("dict[str, dict[str, str]]", _fetched(f"{_NPM}/{name}", accept=_JSON))
        tagged = document.get("dist-tags", {}).get("latest", "")
        cutoff = datetime.now(UTC) - _RELEASE_AGE
        aged = [
            version
            for version, published in document.get("time", {}).items()
            if version not in {"created", "modified"}
            and datetime.fromisoformat(published) <= cutoff
            and _no_later(version, tagged)
        ]
        return _newest(aged or [tagged], name=name, where=_NPM)

    def pin(self, version: str) -> str:
        """npm's caret range, since npm separates comparators by space, never comma."""
        return f"^{version}"


class Rust(Index):
    """crates.io, whose crate document names the newest release nothing has yanked."""

    def latest(self, name: str) -> str:
        crate = cast("dict[str, dict[str, str]]", _fetched(f"{_CRATES}/{name}", accept=_JSON))
        return _newest(
            [crate.get("crate", {}).get("max_stable_version", "")], name=name, where=_CRATES
        )

    def pin(self, version: str) -> str:
        """The exact version, since a host installs a crate frozen only from one exact release
        (`Rust.frozen_inputs`); a range made every host sync refuse (bookokrat, 2026-10-23)."""
        return version


class Go(Index):
    """The Go module proxy, which answers for `latest` directly and resolves no range at all."""

    def latest(self, name: str) -> str:
        """The proxy's `latest`, without its `v` prefix."""
        found = cast("dict[str, str]", _fetched(f"{_GOPROXY}/{name}/@latest", accept=_JSON))
        return _newest([found.get("Version", "").lstrip("v")], name=name, where=_GOPROXY)

    def pin(self, version: str) -> str:
        """The exact version, since `go install` never resolves a range."""
        return version


def _fetched(url: str, *, accept: str) -> Json:
    """The JSON body `url` answers with under a bounded request for the `accept` media type."""
    request = urllib.request.Request(url, headers={"Accept": accept})
    try:
        reply = urllib.request.urlopen(request, timeout=_TIMEOUT)
    except OSError as refusal:
        raise MissionError(f"{url} would not answer: {refusal}") from None
    with reply:
        return cast("Json", json.load(reply))


def _unyanked(listing: dict[str, list[Json]]) -> list[str]:
    """The versions a PEP 691 listing names that keep a file not yanked: a release whose every
    file was yanked is withdrawn, and a requirement raised to one cannot solve (pytest-timeout
    2.5.0, 2026-10-23). A version with no file this reads keeps its place."""
    kept: dict[Version, bool] = {}
    for file in listing.get("files", []):
        if not isinstance(file, dict) or not isinstance(filename := file.get("filename"), str):
            continue
        try:
            _, version, *_ = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            try:
                _, version = parse_sdist_filename(filename)
            except InvalidSdistFilename:
                continue
        kept[version] = kept.get(version, False) or not file.get("yanked")
    versions = [str(version) for version in listing.get("versions", [])]
    return [version for version in versions if _offered(version, kept)]


def _offered(version: str, kept: dict[Version, bool]) -> bool:
    """Whether `version` still offers a file; one `kept` does not mention is taken to."""
    try:
        return kept.get(Version(version), True)
    except InvalidVersion:
        return True


def _no_later(version: str, ceiling: str) -> bool:
    """Whether `version` reads as a release at or below `ceiling`; an unreadable one is not."""
    try:
        return Version(version) <= Version(ceiling)
    except InvalidVersion:
        return False


def _newest(versions: Sequence[str], *, name: str, where: str) -> str:
    """The newest readable final release among `versions`, as pixi's solver prefers, else the
    newest pre-release when the index lists nothing final; refusing when it listed none."""
    readable = []
    for version in versions:
        try:
            readable.append(Version(version))
        except InvalidVersion:
            continue
    if not readable:
        raise MissionError(f"{where} publishes no readable release of {name!r}")
    final = [version for version in readable if not version.is_prerelease]
    return str(max(final or readable))
