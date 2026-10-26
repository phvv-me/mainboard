"""Release evidence of a project's direct Python dependencies, for mcmr's DependencyFact.

Each declared dependency is looked up on PyPI (its simple index and its resolved and latest
compatible release) and on GitHub (whether its source repository is archived). Every answer is
cached under `~/.cache/mainboard/releases` for a day, and an expired answer still serves when the
network is down. A project that cannot reach the network and holds no cached answer gets no
evidence at all: unknown states would read as missing evidence in the rules, which would blame the
project for the machine's connectivity.
"""

import urllib.error
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
from patos import FrozenModel
from pydantic import BaseModel, Field, ValidationError

from ....log import logger
from .exceptions import Unreachable
from .manifests import Manifests
from .models import Declaration, Failure, Record, Release, Repository, Simple
from .web import Web

if TYPE_CHECKING:
    from ...engine.facts import Facts

type Cell = str | int | bool | None

_SIMPLE = "application/vnd.pypi.simple.v1+json"
_LINK_ORDER = {"source": 0, "repository": 1, "code": 2, "homepage": 3}
_PROJECT_STATES = {"active", "archived", "deprecated", "quarantined"}
_TABLES = """
    CREATE TABLE release_runs (collected BOOLEAN);
    CREATE TABLE release_records (ordinal UBIGINT, name VARCHAR, is_development BOOLEAN,
        resolved_release_day BIGINT, latest_compatible_release_day BIGINT,
        latest_compatible_version VARCHAR, project_state VARCHAR, repository_state VARCHAR,
        resolved_release_state VARCHAR);
    CREATE TABLE release_failures (ordinal UBIGINT, signal VARCHAR, source VARCHAR,
        detail VARCHAR);
"""


class Collector(FrozenModel):
    """The evidence of one dependency: PyPI's index and releases, then its GitHub repository."""

    web: Web
    manifests: Manifests

    @staticmethod
    def _project_state(simple: Simple | None) -> str:
        state = simple.status.get("status", "unknown") if simple else "unknown"
        return state if state in _PROJECT_STATES else "unknown"

    @staticmethod
    def _repository(urls: Mapping[str, str]) -> str:
        """The GitHub owner and repository the most direct project link names."""
        for _, url in sorted(
            urls.items(), key=lambda item: _LINK_ORDER.get(item[0].casefold(), 4)
        ):
            parsed = urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if parsed.hostname == "github.com" and len(parts) >= 2:
                return f"{parts[0]}/{parts[1].removesuffix('.git')}"
        return ""

    @staticmethod
    def _version(text: str) -> Version | None:
        """`text` as a PEP 440 version, None when it is not one."""
        with suppress(InvalidVersion):
            return Version(text)
        return None

    def _fetch[M: BaseModel](
        self,
        model: type[M],
        url: str,
        *,
        signal: str,
        failures: list[Failure],
        accept: str = "",
    ) -> M | None:
        try:
            return model.model_validate_json(self.web.get(url, accept=accept))
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
        versions = [found for text in simple.versions if (found := self._version(text))]
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
            Release,
            f"https://pypi.org/pypi/{name}/{version}/json",
            signal=f"{name}:{field}",
            failures=failures,
        )

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
            signal=f"{name}:repository-state",
            failures=failures,
        )
        if response is None:
            return "unknown"
        return "archived" if response.archived else "active"

    def _resolved(self, declaration: Declaration, failures: list[Failure]) -> Version | None:
        """The version `declaration` resolves to here, None with the reason in `failures`."""
        try:
            return self.manifests.resolve(declaration)
        except LookupError as error:
            failures.append(
                Failure(
                    signal=f"dependency:{declaration.name}:resolved-version",
                    source="installed-metadata",
                    reason=str(error),
                )
            )
            return None

    def collect(self, declaration: Declaration) -> tuple[Record, list[Failure]]:
        """The dependency's record and the facts it could not establish."""
        failures: list[Failure] = []
        name = declaration.name
        version = self._resolved(declaration, failures)
        simple = self._fetch(
            Simple,
            f"https://pypi.org/simple/{name}/",
            signal=f"{name}:index",
            failures=failures,
            accept=_SIMPLE,
        )
        latest = self._latest(declaration, simple, failures)
        resolved = self._release(name, version, "resolved-release", failures)
        newest = (
            resolved
            if resolved is not None and latest == version
            else self._release(name, latest, "latest-compatible-release", failures)
        )
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


class ReleaseEvidence(FrozenModel):
    """Loads `release_runs`, `release_records` and `release_failures` for the DependencyFact
    relations: nothing at all when the network was needed and unreachable."""

    root: Path
    web: Web = Field(default_factory=Web)
    workers: int = 8

    @staticmethod
    def _insert(facts: Facts, table: str, rows: Sequence[Iterable[Cell]]) -> None:
        """Append `rows`, each led by its ordinal."""
        if rows:
            marks = ", ".join("?" * (len(list(rows[0])) + 1))
            facts.db.executemany(
                f"INSERT INTO {table} VALUES ({marks})", [[i, *row] for i, row in enumerate(rows)]
            )

    def _collected(self) -> list[tuple[Record, list[Failure]]]:
        """Every declared dependency's record and the facts it could not establish."""
        manifests = Manifests(root=self.root)
        collector = Collector(web=self.web, manifests=manifests)
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(collector.collect, manifests.declarations()))

    def register(self, facts: Facts) -> None:
        """Create the three tables in `facts`, once."""
        if facts.db.execute(
            "SELECT 1 FROM duckdb_tables() WHERE table_name = 'release_runs'"
        ).fetchone():
            return
        facts.db.execute(_TABLES)
        try:
            collected = self._collected()
        except Unreachable as error:
            logger.warning("no release evidence, DependencyFact stays empty", reason=str(error))
            return
        facts.db.execute("INSERT INTO release_runs VALUES (true)")
        failures = sorted(
            {f.signal: f for _, found in collected for f in found}.values(), key=lambda f: f.signal
        )
        self._insert(facts, "release_records", [r.model_dump().values() for r, _ in collected])
        self._insert(facts, "release_failures", [(f.signal, f.source, f.detail) for f in failures])
