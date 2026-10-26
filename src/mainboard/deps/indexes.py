import abc
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version
from patos import Registry
from pydantic import TypeAdapter

from ..core.errors import MissionError
from ..engines.compile.backend import PixiEngine, Process
from .payloads import CrateDocument, Latest, Listing, Package, Release

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    @abc.abstractmethod
    def latest(self, name: str) -> str:
        """The newest published release of `name`, as the index itself reports it."""

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

    def __init__(self, sources: Sequence[str] = ()) -> None:
        """sources: the declared conda channels or Python index url, empty for the public one."""
        self.sources = tuple(sources)

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
        # `pixi search --json` lists each matching package's releases.
        found = TypeAdapter(dict[str, list[Release]]).validate_json(Process.output(command, name))
        versions = [release.version for releases in found.values() for release in releases]
        return _newest(versions, name=name, where="the declared conda channels")


class Python(Index):
    """A PEP 503 index, listing its releases through the PEP 691 JSON the same URL serves."""

    def latest(self, name: str) -> str:
        index = self.sources[0] if self.sources else _PYPI
        # An extra rides the requirement, never the page: `numba-cuda-mlir[cu13]` is a 404.
        url = f"{index.rstrip('/')}/{name.partition('[')[0]}/"
        listing = Listing.model_validate_json(_fetched(url, accept=_SIMPLE))
        return _newest(listing.unyanked(), name=name, where=index)


class Nodejs(Index):
    """The npm registry, whose per-package document carries its own dist-tags."""

    @staticmethod
    def _no_later(version: str, *, ceiling: str) -> bool:
        """Whether `version` reads as a release at or below `ceiling`; an unreadable one is not."""
        try:
            return Version(version) <= Version(ceiling)
        except InvalidVersion:
            return False

    def latest(self, name: str) -> str:
        """The `latest` tag once older than pnpm's minimum release age, else the newest that is.

        Every host installs the lock frozen, and a frozen pnpm refuses a release younger than a day
        (firecrawl-mcp 3.34.1, 22 hours old, 2026-10-23).
        """
        document = Package.model_validate_json(_fetched(f"{_NPM}/{name}", accept=_JSON))
        tagged = document.dist_tags.get("latest", "")
        cutoff = datetime.now(UTC) - _RELEASE_AGE
        aged = [
            version
            for version, published in document.time.items()
            if version not in {"created", "modified"}
            and datetime.fromisoformat(published) <= cutoff
            and Nodejs._no_later(version, ceiling=tagged)
        ]
        return _newest(aged or [tagged], name=name, where=_NPM)

    def pin(self, version: str) -> str:
        """npm's caret range, since npm separates comparators by space, never comma."""
        return f"^{version}"


class _Exact(Index, abc.ABC):
    """An index whose installer takes one exact version and resolves no range.

    A host installs a crate frozen only from an exact version (`Rust.frozen_inputs`; a range
    made every host sync refuse, bookokrat 2026-10-23), and `go install` never resolves a range.
    """

    def pin(self, version: str) -> str:
        return version


class Rust(_Exact):
    """crates.io, whose crate document names the newest release nothing has yanked."""

    def latest(self, name: str) -> str:
        document = CrateDocument.model_validate_json(_fetched(f"{_CRATES}/{name}", accept=_JSON))
        return _newest([document.crate.max_stable_version], name=name, where=_CRATES)


class Go(_Exact):
    """The Go module proxy, which answers for `latest` directly and resolves no range at all."""

    def latest(self, name: str) -> str:
        """The proxy's `latest`, without its `v` prefix."""
        found = Latest.model_validate_json(_fetched(f"{_GOPROXY}/{name}/@latest", accept=_JSON))
        return _newest([found.version.lstrip("v")], name=name, where=_GOPROXY)


def _fetched(url: str, *, accept: str) -> bytes:
    """The body `url` answers with under a bounded request for the `accept` media type."""
    request = urllib.request.Request(url, headers={"Accept": accept})
    try:
        reply = urllib.request.urlopen(request, timeout=_TIMEOUT)
    except OSError as refusal:
        raise MissionError(f"{url} would not answer: {refusal}") from None
    with reply:
        return reply.read()


def _newest(versions: Sequence[str], *, name: str, where: str) -> str:
    """The newest readable final release among `versions`, as pixi's solver prefers.

    When the index lists nothing final it is the newest pre-release, and with nothing listed it
    refuses.
    """
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
