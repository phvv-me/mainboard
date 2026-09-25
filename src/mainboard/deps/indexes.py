import abc
import json
import urllib.request
from typing import TYPE_CHECKING, cast

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

        An unreadable release (a conda date stamp) gets only a floor, which still resolves.
        """
        try:
            release = list(Version(version).release)
        except InvalidVersion:
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
        versions = [str(record["version"]) for records in found.values() for record in records]
        return _newest(versions, name=name, where="the declared conda channels")


class Python(Index):
    """A PEP 503 index, listing its releases through the PEP 691 JSON the same URL serves."""

    def latest(self, name: str) -> str:
        index = self.sources[0] if self.sources else _PYPI
        # An extra rides the requirement, never the page: `numba-cuda-mlir[cu13]` is a 404.
        url = f"{index.rstrip('/')}/{name.partition('[')[0]}/"
        listing = cast("dict[str, list[str]]", _fetched(url, accept=_SIMPLE))
        return _newest(listing.get("versions", []), name=name, where=index)


class Nodejs(Index):
    """The npm registry, whose per-package document carries its own dist-tags."""

    def latest(self, name: str) -> str:
        tags = cast("dict[str, dict[str, str]]", _fetched(f"{_NPM}/{name}", accept=_JSON))
        return _newest([tags.get("dist-tags", {}).get("latest", "")], name=name, where=_NPM)

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


class Go(Index):
    """The Go module proxy, which answers for `latest` directly and resolves no range at all."""

    def latest(self, name: str) -> str:
        """The proxy's `latest`, without its `v` prefix."""
        found = cast("dict[str, str]", _fetched(f"{_GOPROXY}/{name}/@latest", accept=_JSON))
        return _newest([str(found.get("Version", "")).lstrip("v")], name=name, where=_GOPROXY)

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


def _newest(versions: Sequence[str], *, name: str, where: str) -> str:
    """The newest readable release among `versions`, refusing when the index listed none."""
    readable = []
    for version in versions:
        try:
            readable.append(Version(version))
        except InvalidVersion:
            continue
    if not readable:
        raise MissionError(f"{where} publishes no readable release of {name!r}")
    return str(max(readable))
