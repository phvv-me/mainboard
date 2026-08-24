from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mainboard.engines.compile import Ecosystem

if TYPE_CHECKING:
    from mainboard.manifest.schema.spec import Json


class Bind(Protocol):
    """Binds one ecosystem implementation to a table body, in the fixture workspace."""

    def __call__[E: Ecosystem](self, kind: type[E], body: Mapping[str, Json]) -> E: ...


class Record(Protocol):
    """Writes one `dist-info` into a site-packages tree the way an installer leaves it."""

    def __call__(
        self,
        site_packages: Path,
        name: str,
        *,
        installer: str = ...,
        roots: str = ...,
        url: str = ...,
        editable: bool = ...,
        files: list[str] | None = ...,
    ) -> Path: ...
