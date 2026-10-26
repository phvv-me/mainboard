import os
import re
from collections import deque
from collections.abc import Iterable
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel

from ..core.section import Section, Verdict
from ..runtime.activation import Runtime
from .elf import Elf
from .inventory import Family, Inventory, Provided, family_of
from .loader import Found, Loader, system_cache
from .remedy import Raise, Relink, Remedy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# A library bigger than this carries kernels, not the names of its siblings, so its strings are
# not read: cuDNN's dispatcher is 129 KB and its graph library 128 MB, its engines 200 MB and up.
_STRINGS_CEILING = 192 * 2**20
_CUDA_MAJOR = re.compile(r"-cu(\d+)$")
_NUMBER = re.compile(r"\d+")
_SUFFIX = re.compile(r"\.so((?:\.\d+)+)$")

type Release = tuple[int, ...]


class Kind(StrEnum):
    """What is wrong with a family: a library from outside, one from nowhere, or two releases."""

    OUTSIDE = auto()
    MISSING = auto()
    SPLIT = auto()


class Finding(FrozenModel):
    """One clash an environment has with its host or with itself.

    detail: the sentence saying which library, who asked for it and where it came from.
    remedy: the repair, None when no change to this environment can end it.
    """

    kind: Kind
    family: str
    detail: str
    verdict: Verdict
    remedy: Remedy | None = None


class Resolution(FrozenModel):
    """One soname a library of the environment asks for, and the file the loader answers.

    linked: the requester links against it, so nothing answering is a load failure; otherwise
        the requester only carries its name and may never try it.
    """

    soname: str
    requester: Path
    found: Found | None
    linked: bool


def _named(name: str) -> Release:
    """The numbers a library file name ends in: `libcudnn.so.9.27.0` is (9, 27, 0)."""
    suffix = _SUFFIX.search(name)
    return tuple(int(number) for number in _NUMBER.findall(suffix.group(1))) if suffix else ()


def _release(provided: Provided) -> Release:
    """The major and minor of the package that installed a library.

    The package's, not the file's: a soname numbers its own interface (`libnvToolsExt.so.1.0.0`
    ships in nvtx 13.2), and two packages of one family are compared by what they were released as.
    """
    found = tuple(int(number) for number in _NUMBER.findall(provided.provider.version))
    return (found or _named(provided.path.resolve().name))[:2]


def _is_agree(first: Release, *, second: Release) -> bool:
    """Whether two releases are one on every number both state."""
    return all(a == b for a, b in zip(first, second, strict=False))


class LibraryAudit:
    """Which CUDA libraries an environment's processes load, and from where.

    Replays the dynamic loader over every library the environment's CUDA packages installed and
    over torch's, for the sonames they link and the ones they open by name, and reports the
    ones found outside the environment (the host's apt cuDNN taking over the sub-library the
    wheel lacks), the ones found nowhere, and the families the environment holds in two
    releases. Nothing is imported or run: the files are read, so it answers on a login node.

    prefix: the installed environment.
    env: its name, empty for the default, which a repair that edits the manifest passes on.
    library_path: `LD_LIBRARY_PATH` as the environment's activation sets it.
    cache: the host's `ld.so.cache`, a soname map.
    trusted: the host's default directories, this machine's own when None.
    """

    @classmethod
    def entered(cls, prefix: Path, *, env: str = "") -> LibraryAudit:
        """The audit of `prefix` as a job enters it on this machine.

        It reads the path the activation builds over this process's own, and this host's cache.
        """
        entered = Runtime(prefix).entered(os.environ)
        path = entered.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        return cls(
            prefix, env=env, library_path=[Path(d) for d in path if d], cache=system_cache()
        )

    def __init__(
        self,
        prefix: Path,
        *,
        env: str = "",
        library_path: Sequence[Path],
        cache: Mapping[str, Path],
        trusted: Sequence[Path] | None = None,
    ) -> None:
        self.prefix = prefix
        self.env = env
        self.inventory = Inventory(prefix)
        self.loader = Loader(library_path, cache, trusted)

    def _answers(self, path: Path) -> list[Resolution]:
        """The resolution of each family soname the library at `path` asks for."""
        elf = Elf.read(path)
        return [
            Resolution(
                soname=soname, requester=path, found=self.loader.find(soname, elf), linked=linked
            )
            for soname, linked in self._asked(elf)
        ]

    def _asked(self, elf: Elf) -> list[tuple[str, bool]]:
        """The family sonames `elf` links against, then those it may open by name.

        A sibling carried as a bare `libcudnn_graph.so` is opened with the version appended, as
        the soname (`.so.9`) or in full (`.so.9.27.0`), so both are asked for. The library's own
        name is no sibling.
        """
        needed = [soname for soname in elf.needed if family_of(soname)]
        asked = [(soname, True) for soname in needed]
        family = family_of(elf.path.name)
        if family is None or elf.path.stat().st_size > _STRINGS_CEILING:
            return asked
        own, version = elf.path.name.split(".so")[0], self._version(elf.path)
        for name in elf.mentions(family.stem):
            if name.split(".so")[0] == own or name in needed:
                continue
            candidates = [name] if not name.endswith(".so") else [f"{name}.{v}" for v in version]
            asked += [(candidate, False) for candidate in candidates]
        return asked

    def _majors(self) -> list[Finding]:
        """A finding when wheels for different CUDA majors (`-cu12`, `-cu13`) share the prefix."""
        wheels = {
            provided.provider.name: match.group(1)
            for provided in self.inventory.libraries
            if (match := _CUDA_MAJOR.search(provided.provider.name))
        }
        if len(set(wheels.values())) < 2:
            return []
        detail = "wheels for several CUDA majors installed together: " + ", ".join(sorted(wheels))
        return [Finding(kind=Kind.SPLIT, family="CUDA", detail=detail, verdict=Verdict.FAIL)]

    def _remedy(self, soname: str, requester: Path, shipped: Provided | None) -> Remedy | None:
        """The link when the environment has the file under another name, else the package.

        Raising is preferred whenever the file is not in the environment at all, since a newer
        release that ships the family ends the clash in the manifest where it can be reviewed.
        """
        if shipped is not None:
            if shipped.path.name == soname:
                return None
            return Relink(directory=shipped.path.parent, name=soname, target=shipped.path.name)
        family = family_of(soname)
        owner = self.inventory.owner(requester) if family_of(requester.name) == family else None
        owner = owner or next(
            (p.provider for p in self.inventory.libraries if p.family == family), None
        )
        if owner is None:
            return None
        return Raise(package=owner.name, ecosystem=owner.ecosystem, env=self.env)

    def _shipped(self, soname: str) -> Provided | None:
        """The environment's own file for `soname`, spelled differently, else None.

        Two spellings are one file when one name begins with the other (`.so.9` and
        `.so.9.27.0`) and the numbers the wanted name states are the release the file's package
        installed. A soname of another version is another ABI and no substitute.
        """
        stem = soname.split(".so")[0]
        wanted = _named(soname)[:2]
        return next(
            (
                provided
                for provided in self.inventory.libraries
                if provided.path.name.split(".so")[0] == stem
                and (
                    soname.startswith(provided.path.name) or provided.path.name.startswith(soname)
                )
                and _is_agree(wanted, second=_release(provided))
            ),
            None,
        )

    def _split(self, family: Family, releases: Mapping[Release, Provided | None]) -> Finding:
        """A family in two releases: broken when the host is one, a warning when all are ours.

        Report only. A mix inside the environment is the owner's to settle, since a pip wheel
        beside a conda build is declared on purpose; the host's copy is repaired through the
        library the environment lacks, which the unresolved finding raises or relinks.
        """
        host = None in releases.values()
        spoken = [
            f"{p.provider.name} {p.provider.version}" if p else "the host"
            for p in releases.values()
        ]
        return Finding(
            kind=Kind.SPLIT,
            family=family.name,
            detail=("releases mixed in one process: " if host else "releases installed together: ")
            + ", ".join(spoken),
            verdict=Verdict.FAIL if host else Verdict.WARN,
        )

    def _splits(self, resolutions: Iterable[Resolution]) -> list[Finding]:
        """One finding per family held in two releases, the host's copies counted.

        Each family maps the releases in play to who supplies them: a package of the environment
        (the one to raise, lowest first) or the host. A host file is only a release of the
        family when its name's first number is a major the environment's packages use too
        (cuDNN 9.27 against 9.20), since a soname that numbers its own interface says nothing
        about the toolkit.
        """
        supplied: dict[Family, dict[Release, Provided | None]] = {}
        for provided in self.inventory.libraries:
            supplied.setdefault(provided.family, {}).setdefault(_release(provided), provided)
        for resolution in resolutions:
            found = resolution.found
            family = family_of(resolution.soname)
            if found is None or family is None or self.inside(found.path):
                continue
            host = _named(found.path.resolve().name)[:2]
            if any(host[:1] == ours[:1] for ours in supplied.get(family, {})):
                supplied[family].setdefault(host, None)
        return [
            self._split(family, releases)
            for family, releases in supplied.items()
            if not all(_is_agree(first, second=other) for first in releases for other in releases)
        ]

    def _unresolved(self, resolutions: Iterable[Resolution]) -> list[Finding]:
        """One finding per soname the host answers, and per linked one nothing does.

        A name a library only carries and nothing answers is a candidate it may never try, and
        says nothing; one the host answers is the library it would load.
        """
        asked: dict[str, list[Resolution]] = {}
        for resolution in resolutions:
            found = resolution.found
            if found is None and not resolution.linked:
                continue
            if found is None or not self.inside(found.path):
                asked.setdefault(resolution.soname, []).append(resolution)
        return [self._unresolved_finding(soname, group) for soname, group in asked.items()]

    def _unresolved_finding(self, soname: str, group: Sequence[Resolution]) -> Finding:
        found = group[0].found
        family = family_of(soname)
        assert family is not None  # only family sonames are ever asked for
        shipped = self._shipped(soname)
        who = ", ".join(sorted({resolution.requester.name for resolution in group}))
        where = f"loads from {found.path} ({found.via})" if found else "is found nowhere"
        has = (
            f"the environment ships {shipped.path}" if shipped else "the environment ships no copy"
        )
        return Finding(
            kind=Kind.OUTSIDE if found else Kind.MISSING,
            family=family.name,
            detail=f"{soname}, asked by {who}, {where}; {has}",
            verdict=Verdict.WARN if found else Verdict.FAIL,
            remedy=self._remedy(soname, group[0].requester, shipped),
        )

    def _version(self, path: Path) -> list[str]:
        """The version suffixes a sibling is opened with, from `path`'s name or its package.

        The major (`9`) and the full release (`9.27.0`), empty when neither is known.
        """
        numbers = _named(path.resolve().name)
        owner = self.inventory.owner(path)
        if len(numbers) < 3 and owner is not None:
            numbers = tuple(int(number) for number in _NUMBER.findall(owner.version))
        return (
            list(dict.fromkeys([str(numbers[0]), ".".join(map(str, numbers[:3]))]))
            if numbers
            else []
        )

    def findings(self) -> list[Finding]:
        """Every clash, library by library first and family by family after."""
        resolved = self.resolutions()
        return [*self._unresolved(resolved), *self._splits(resolved), *self._majors()]

    def inside(self, path: Path) -> bool:
        """Whether `path` is a file of the environment, wherever its link leads."""
        return path.is_relative_to(self.prefix) or path.resolve().is_relative_to(
            self.prefix.resolve()
        )

    def remedies[R: Remedy](self, kind: type[R]) -> list[R]:
        """The distinct repairs of one kind the findings call for."""
        found = (finding.remedy for finding in self.findings())
        return list(dict.fromkeys(remedy for remedy in found if isinstance(remedy, kind)))

    def resolutions(self) -> list[Resolution]:
        """Every family soname the environment's libraries ask for, and who answers it.

        A library found along the way is read in turn, whoever owns it, so a sibling the host
        supplies is followed to what it asks for next.
        """
        queue = deque(
            [*(provided.path for provided in self.inventory.libraries), *self.inventory.torch]
        )
        visited: set[Path] = set()
        resolutions: list[Resolution] = []
        while queue:
            path = queue.popleft()
            if (real := path.resolve()) in visited:
                continue
            visited.add(real)
            answers = self._answers(path)
            resolutions.extend(answers)
            queue.extend(answer.found.path for answer in answers if answer.found is not None)
        return resolutions

    def section(self) -> Section:
        """The audit as one row, each finding a sentence and each distinct repair a command."""
        found = self.findings()
        name = self.env or "default"
        if not found:
            families = len({provided.family for provided in self.inventory.libraries})
            return Section(
                section="libraries",
                verdict=Verdict.PASS,
                detail=f"{name}: {families} CUDA families, all loading from the environment",
            )
        failing = any(finding.verdict is Verdict.FAIL for finding in found)
        return Section(
            section="libraries",
            verdict=Verdict.FAIL if failing else Verdict.WARN,
            detail=f"{name}: " + "; ".join(f"{f.family}: {f.detail}" for f in found),
            fix="; ".join(dict.fromkeys(f.remedy.command for f in found if f.remedy)),
        )
