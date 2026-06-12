from __future__ import annotations

from ..enums import ToolCategory
from .base import Field, FrozenModel


class DetectedTool(FrozenModel):
    """A build tool discovered on the host PATH.

    name: display name of the tool, e.g. `gcc` or `cmake`.
    category: the toolchain group the tool belongs to.
    path: absolute path to the resolved binary, `None` when not on PATH.
    version: parsed version string, `None` when absent or unparseable.
    available: whether the binary was found on PATH.
    """

    name: str
    category: ToolCategory = Field(default=ToolCategory.BUILD_SYSTEM)
    path: str | None = None
    version: str | None = None
    available: bool = False
