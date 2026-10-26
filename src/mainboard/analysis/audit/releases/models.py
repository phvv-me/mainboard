"""What PyPI and GitHub say of a dependency, and the rows its evidence becomes."""

from patos import FrozenModel
from pydantic import AwareDatetime, Field


class Simple(FrozenModel):
    """What the PyPI simple index says of a project."""

    versions: list[str] = []
    status: dict[str, str] = Field(default_factory=dict, validation_alias="project-status")


class Artifact(FrozenModel):
    upload_time_iso_8601: AwareDatetime
    yanked: bool = False


class Info(FrozenModel):
    project_urls: dict[str, str] | None = None


class Release(FrozenModel):
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


class Repository(FrozenModel):
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
