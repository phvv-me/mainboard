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

# Long enough for an index that is merely slow, short enough that a typo in a package name
# never leaves a terminal waiting on a registry that will answer with nothing anyway.
_TIMEOUT = 20.0

# The machine-readable listing PEP 691 defines, which every PEP 503 index serves beside its
# HTML. Asking for it is the difference between reading an index and scraping a web page.
_JSON = "application/json"
_SIMPLE = "application/vnd.pypi.simple.v1+json"

_PYPI = "https://pypi.org/simple"
_NPM = "https://registry.npmjs.org"
_CRATES = "https://crates.io/api/v1/crates"
_GOPROXY = "https://proxy.golang.org"


class Index(Registry, abc.ABC):
    """Where one ecosystem publishes its releases, and how its manager spells a pin.

    Both halves of the same question and so one class: `add` with no version and `upgrade` both
    want the newest release of a name and the requirement that names it, and the requirement
    has to be written in the syntax the ecosystem's own resolver reads, which is why a npm
    caret and a conda range cannot share one formatter. Implementations enroll under this root
    keyed by the manifest table they answer for, so `-l rust` reaches the crates registry
    without anything here listing the ecosystems that exist.
    """

    def __init__(self, sources: Sequence[str] = ()) -> None:
        """Hold the registries version queries will read.

        sources: the registries the manifest declares for this ecosystem, the conda channels
        or a Python index url, empty for one published to a single public registry.
        """
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
        """The requirement naming `version` and the releases compatible with it.

        pixi's own semver pinning, where `1.2.3` becomes `>=1.2.3, <2` and `0.1.0` becomes
        `>=0.1.0, <0.2`, which is already the shape this manifest writes for conda and Python.
        A release the version grammar cannot read, a conda date stamp for one, gets a floor and
        no ceiling rather than a refusal, since a wider pin still resolves and a refusal does
        not.
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
        """The newest release across every declared channel and platform pixi finds it on."""
        channels = [flag for channel in self.sources for flag in ("--channel", channel)]
        command = PixiEngine().command["search", "--json", *channels, name]
        found = cast("dict[str, list[dict[str, str]]]", json.loads(Process.output(command, name)))
        versions = [str(record["version"]) for records in found.values() for record in records]
        return _newest(versions, name=name, where="the declared conda channels")


class Python(Index):
    """A PEP 503 index, listing its releases through the PEP 691 JSON the same URL serves."""

    def latest(self, name: str) -> str:
        """The newest release the index lists, PyPI when the manifest declares no other."""
        index = self.sources[0] if self.sources else _PYPI
        url = f"{index.rstrip('/')}/{name}/"
        listing = cast("dict[str, list[str]]", _fetched(url, accept=_SIMPLE))
        return _newest(listing.get("versions", []), name=name, where=index)


class Nodejs(Index):
    """The npm registry, whose per-package document carries its own dist-tags."""

    def latest(self, name: str) -> str:
        """The release npm's own `latest` tag points at."""
        tags = cast("dict[str, dict[str, str]]", _fetched(f"{_NPM}/{name}", accept=_JSON))
        return _newest([tags.get("dist-tags", {}).get("latest", "")], name=name, where=_NPM)

    def pin(self, version: str) -> str:
        """npm's caret range, since its grammar separates comparators by space, never comma."""
        return f"^{version}"


class Rust(Index):
    """crates.io, whose crate document names the newest release nothing has yanked."""

    def latest(self, name: str) -> str:
        """The newest stable release crates.io reports for the crate."""
        crate = cast("dict[str, dict[str, str]]", _fetched(f"{_CRATES}/{name}", accept=_JSON))
        return _newest(
            [crate.get("crate", {}).get("max_stable_version", "")], name=name, where=_CRATES
        )


class Go(Index):
    """The Go module proxy, which answers for `latest` directly and resolves no range at all."""

    def latest(self, name: str) -> str:
        """The version the module proxy resolves `latest` to, without its `v` prefix."""
        found = cast("dict[str, str]", _fetched(f"{_GOPROXY}/{name}/@latest", accept=_JSON))
        return _newest([str(found.get("Version", "")).lstrip("v")], name=name, where=_GOPROXY)

    def pin(self, version: str) -> str:
        """The exact version, since `go install` resolves one version and never a range."""
        return version


def _fetched(url: str, *, accept: str) -> Json:
    """The JSON body `url` answers with, under a bounded request.

    url: the registry endpoint to read.
    accept: the media type the registry serves its machine-readable listing as.
    """
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
