"""Release evidence of a project's direct Python dependencies, for mcmr's DependencyFact.

Each declared dependency is looked up on PyPI (its simple index and its resolved and latest
compatible release) and on GitHub (whether its source repository is archived). Every answer is
cached under `~/.cache/mainboard/releases` for a day, and an expired answer still serves when the
network is down. A project that cannot reach the network and holds no cached answer gets no
evidence at all: unknown states would read as missing evidence in the rules, which would blame the
project for the machine's connectivity.
"""

import os
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from functools import cached_property
from hashlib import sha1
from importlib import metadata
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version
from patos import FrozenModel
from pydantic import AwareDatetime, BaseModel, Field, PrivateAttr, ValidationError

from ...log import logger

if TYPE_CHECKING:
    from ..facts import Facts

type Cell = str | int | bool | None

CACHE = Path.home() / ".cache/mainboard/releases"
JSON = "application/json"
SIMPLE = "application/vnd.pypi.simple.v1+json"
LINK_ORDER = {"source": 0, "repository": 1, "code": 2, "homepage": 3}
PROJECT_STATES = {"active", "archived", "deprecated", "quarantined"}


class Unreachable(OSError):
    """The network did not answer and no cached answer exists."""


class Simple(BaseModel):
    """What the PyPI simple index says of a project."""

    versions: list[str] = []
    status: dict[str, str] = Field(default_factory=dict, validation_alias="project-status")


class Artifact(BaseModel):
    upload_time_iso_8601: AwareDatetime
    yanked: bool = False


class Info(BaseModel):
    project_urls: dict[str, str] | None = None


class Release(BaseModel):
    """What PyPI says of one exact release; a release with no artifact says nothing."""

    info: Info
    urls: list[Artifact] = Field(min_length=1)

    @property
    def day(self) -> int:
        """The ordinal day of the earliest upload."""
        return min(artifact.upload_time_iso_8601 for artifact in self.urls).date().toordinal()

    @property
    def state(self) -> str:
        return "yanked" if all(artifact.yanked for artifact in self.urls) else "available"


class Repository(BaseModel):
    archived: bool


class Declaration(FrozenModel):
    name: str
    requirement: str
    is_development: bool = False


class Record(FrozenModel):
    """One dependency's row of `release_records`, its fields in the table's column order."""

    name: str
    is_development: bool
    resolved_release_day: int | None
    latest_compatible_release_day: int | None
    latest_compatible_version: str | None
    project_state: str
    repository_state: str
    resolved_release_state: str


class Failure(FrozenModel):
    """One fact the evidence could not establish, as the rules' evidence lists it."""

    signal: str
    source: str
    reason: str

    @property
    def detail(self) -> str:
        return f"{self.signal.rsplit(':', 1)[-1]} is unknown because {self.reason}"


class Manifests(FrozenModel):
    """The direct dependencies a project declares and the version each resolves to locally."""

    root: Path

    def declarations(self) -> list[Declaration]:
        """Unique direct requirements by name, a runtime one winning over a development one."""
        found: dict[str, Declaration] = {}
        for declaration in [*self._pyproject(), *self._workspace()]:
            if declaration.name not in found or found[declaration.name].is_development:
                found[declaration.name] = declaration
        return [found[name] for name in sorted(found)]

    def resolve(self, declaration: Declaration) -> Version:
        """The greatest locked registry version the requirement allows, else the installed one.

        Raises LookupError with the reason when neither exists.
        """
        specifier = Requirement(declaration.requirement).specifier
        locked = self.locked.get(declaration.name, [])
        if allowed := [v for v in locked if specifier.contains(v, True)]:
            return max(allowed)
        try:
            installed = Version(metadata.version(declaration.name))
        except metadata.PackageNotFoundError:
            raise LookupError("dependency is not installed") from None
        except InvalidVersion:
            raise LookupError("installed version is not valid PEP 440") from None
        if not specifier.contains(installed, True):
            raise LookupError("installed version does not satisfy declaration")
        return installed

    @cached_property
    def locked(self) -> dict[str, list[Version]]:
        """The registry versions `uv.lock` holds, by canonical name."""
        found: dict[str, list[Version]] = {}
        for package in self._read("uv.lock").get("package", []):
            if "registry" in package.get("source", {}):
                found.setdefault(canonicalize_name(package["name"]), []).append(
                    Version(package["version"])
                )
        return found

    def _read(self, name: str) -> dict:
        try:
            return tomllib.loads((self.root / name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _pyproject(self) -> list[Declaration]:
        project = self._read("pyproject.toml")
        runtime = project.get("project", {})
        return [
            *self._declared(runtime.get("dependencies", []), development=False),
            *[
                d
                for group in runtime.get("optional-dependencies", {}).values()
                for d in self._declared(group, development=True)
            ],
            *self._declared(project.get("build-system", {}).get("requires", []), development=True),
        ]

    def _workspace(self) -> list[Declaration]:
        """The Python tables of `mainboard.toml`, minus entries installing this project itself."""
        manifest = self._read("mainboard.toml")
        tables = (
            (manifest.get("python", {}).get("deps", {}), False),
            (manifest.get("dev", {}).get("python", {}).get("deps", {}), True),
        )
        return [
            d
            for table, development in tables
            for name, value in table.items()
            if not self._is_self(value)
            for d in self._declared([f"{name}{self._constraint(value)}"], development=development)
        ]

    def _is_self(self, value: str | dict) -> bool:
        path = value.get("path") if isinstance(value, dict) else None
        return isinstance(path, str) and (self.root / path).resolve() == self.root.resolve()

    @staticmethod
    def _constraint(value: str | dict) -> str:
        constraint = value if isinstance(value, str) else value.get("version", "")
        return "" if constraint == "*" else constraint

    @staticmethod
    def _declared(requirements: list[str], *, development: bool) -> list[Declaration]:
        found = []
        for text in requirements:
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                logger.warning("skipped an unreadable dependency", requirement=text)
                continue
            found.append(
                Declaration(
                    name=canonicalize_name(requirement.name),
                    requirement=str(requirement),
                    is_development=development,
                )
            )
        return found


class Web(FrozenModel):
    """JSON documents over HTTP, cached on disk."""

    cache: Path = CACHE
    ttl_seconds: int = 24 * 3600
    timeout_seconds: int = 15
    _offline: Event = PrivateAttr(default_factory=Event)

    def get(self, url: str, accept: str = JSON) -> str:
        """The document at `url`: a fresh cached copy, else the network's, else an expired copy.

        A 404 is an answer too and is cached. Raises HTTPError when the server answers with a
        failure, Unreachable when it does not answer and nothing is cached.
        """
        entry = self.cache / sha1(f"{accept} {url}".encode(), usedforsecurity=False).hexdigest()
        age = self._age(entry)
        if age is not None and age < self.ttl_seconds:
            return self._read(entry, url)
        try:
            text = self._download(url, accept)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                self._store(entry, "")
            raise
        except OSError as error:
            self._offline.set()
            if age is None:
                raise Unreachable(f"{url}: {error}") from error
            return self._read(entry, url)
        self._store(entry, text)
        return text

    def _store(self, entry: Path, text: str) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        scratch = entry.with_suffix(f".{os.getpid()}.{id(text)}")
        scratch.write_text(text, encoding="utf-8")
        scratch.replace(entry)

    @staticmethod
    def _read(entry: Path, url: str) -> str:
        """A cached document; an empty entry is a cached 404."""
        if text := entry.read_text(encoding="utf-8"):
            return text
        raise urllib.error.HTTPError(url, 404, "not found (cached)", Message(), None)

    @staticmethod
    def _age(entry: Path) -> float | None:
        try:
            return time.time() - entry.stat().st_mtime
        except FileNotFoundError:
            return None

    def _download(self, url: str, accept: str) -> str:
        if self._offline.is_set():
            raise OSError("the network did not answer earlier in this run")
        headers = {"Accept": accept, "User-Agent": "mainboard-release-evidence"}
        if (token := os.environ.get("GITHUB_TOKEN")) and url.startswith("https://api.github.com/"):
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode()


class Collector(FrozenModel):
    """The evidence of one dependency: PyPI's index and releases, then its GitHub repository."""

    web: Web
    manifests: Manifests

    def collect(self, declaration: Declaration) -> tuple[Record, list[Failure]]:
        """The dependency's record and the facts it could not establish."""
        failures: list[Failure] = []
        name = declaration.name
        try:
            version = self.manifests.resolve(declaration)
        except LookupError as error:
            version = None
            failures.append(
                Failure(
                    signal=f"dependency:{name}:resolved-version",
                    source="installed-metadata",
                    reason=str(error),
                )
            )
        simple = self._fetch(
            Simple, f"https://pypi.org/simple/{name}/", f"{name}:index", failures, SIMPLE
        )
        latest = self._latest(declaration, simple, failures)
        resolved = self._release(name, version, "resolved-release", failures)
        newest = resolved
        if resolved is None or latest != version:
            newest = self._release(name, latest, "latest-compatible-release", failures)
        urls = (resolved.info.project_urls if resolved else None) or {}
        record = Record(
            name=name,
            is_development=declaration.is_development,
            resolved_release_day=resolved.day if resolved else None,
            latest_compatible_release_day=newest.day if newest else None,
            latest_compatible_version=str(latest) if latest else None,
            project_state=self._project_state(simple),
            repository_state=self._repository_state(name, urls, failures),
            resolved_release_state=resolved.state if resolved else "unknown",
        )
        return record, failures

    def _fetch[M: BaseModel](
        self, model: type[M], url: str, signal: str, failures: list[Failure], accept: str = JSON
    ) -> M | None:
        try:
            return model.model_validate_json(self.web.get(url, accept))
        except (urllib.error.HTTPError, ValidationError) as error:
            failures.append(
                Failure(signal=f"dependency:{signal}", source=url, reason=type(error).__name__)
            )
            return None

    def _latest(
        self, declaration: Declaration, simple: Simple | None, failures: list[Failure]
    ) -> Version | None:
        if simple is None:
            return None
        versions = []
        for text in simple.versions:
            try:
                versions.append(Version(text))
            except InvalidVersion:
                continue
        allowed = list(Requirement(declaration.requirement).specifier.filter(versions))
        if not allowed:
            failures.append(
                Failure(
                    signal=f"dependency:{declaration.name}:latest-compatible-version",
                    source=f"https://pypi.org/simple/{declaration.name}/",
                    reason="no compatible PyPI release",
                )
            )
        return max(allowed, default=None)

    def _release(
        self, name: str, version: Version | None, field: str, failures: list[Failure]
    ) -> Release | None:
        if version is None:
            failures.append(
                Failure(
                    signal=f"dependency:{name}:{field}",
                    source="installed-metadata",
                    reason="version is unresolved",
                )
            )
            return None
        return self._fetch(
            Release, f"https://pypi.org/pypi/{name}/{version}/json", f"{name}:{field}", failures
        )

    @staticmethod
    def _project_state(simple: Simple | None) -> str:
        state = simple.status.get("status", "unknown") if simple else "unknown"
        return state if state in PROJECT_STATES else "unknown"

    def _repository_state(self, name: str, urls: dict[str, str], failures: list[Failure]) -> str:
        if not (repository := self._repository(urls)):
            failures.append(
                Failure(
                    signal=f"dependency:{name}:repository-state",
                    source="pypi-release",
                    reason="GitHub repository is not published",
                )
            )
            return "unknown"
        response = self._fetch(
            Repository,
            f"https://api.github.com/repos/{repository}",
            f"{name}:repository-state",
            failures,
        )
        if response is None:
            return "unknown"
        return "archived" if response.archived else "active"

    @staticmethod
    def _repository(urls: dict[str, str]) -> str:
        """The GitHub owner and repository the most direct project link names."""
        for _, url in sorted(urls.items(), key=lambda item: LINK_ORDER.get(item[0].casefold(), 4)):
            parsed = urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if parsed.hostname == "github.com" and len(parts) >= 2:
                return f"{parts[0]}/{parts[1].removesuffix('.git')}"
        return ""


class ReleaseEvidence(FrozenModel):
    """Loads `release_runs`, `release_records` and `release_failures` for the DependencyFact
    relations: nothing at all when the network was needed and unreachable."""

    root: Path
    web: Web = Field(default_factory=Web)
    workers: int = 8

    def register(self, facts: Facts) -> None:
        """Create the three tables in `facts`, once."""
        if facts.db.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = 'release_runs'"
        ).fetchone():
            return
        facts.db.execute("CREATE TABLE release_runs (collected BOOLEAN)")
        facts.db.execute(
            "CREATE TABLE release_records (ordinal UBIGINT, name VARCHAR, is_development BOOLEAN,"
            " resolved_release_day BIGINT, latest_compatible_release_day BIGINT,"
            " latest_compatible_version VARCHAR, project_state VARCHAR, repository_state VARCHAR,"
            " resolved_release_state VARCHAR)"
        )
        facts.db.execute(
            "CREATE TABLE release_failures (ordinal UBIGINT, signal VARCHAR, source VARCHAR,"
            " detail VARCHAR)"
        )
        manifests = Manifests(root=self.root)
        collector = Collector(web=self.web, manifests=manifests)
        try:
            with ThreadPoolExecutor(self.workers) as pool:
                collected = list(pool.map(collector.collect, manifests.declarations()))
        except Unreachable as error:
            logger.warning("no release evidence, DependencyFact stays empty", reason=str(error))
            return
        facts.db.execute("INSERT INTO release_runs VALUES (true)")
        failures = sorted(
            {f.signal: f for _, found in collected for f in found}.values(), key=lambda f: f.signal
        )
        self._insert(facts, "release_records", [r.model_dump().values() for r, _ in collected])
        self._insert(facts, "release_failures", [(f.signal, f.source, f.detail) for f in failures])

    @staticmethod
    def _insert(facts: Facts, table: str, rows: Sequence[Iterable[Cell]]) -> None:
        """Append `rows`, each led by its ordinal."""
        if rows:
            marks = ", ".join("?" * (len(list(rows[0])) + 1))
            facts.db.executemany(
                f"INSERT INTO {table} VALUES ({marks})", [[i, *row] for i, row in enumerate(rows)]
            )
