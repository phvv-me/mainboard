from pydantic import model_validator

from ...core.base import Declared
from .spec import Json


class Template(Declared):
    """One project template this workspace keeps, under the name `new` renders it by.

    path: a workspace-relative directory or anything the renderer resolves, a git URL included.
    into: the workspace-relative directory projects land under, the root when empty.
    answers: the questions this workspace always answers the same way.
    """

    path: str
    into: str = ""
    answers: dict[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def from_bare_string(cls, value: Json) -> Json:
        """Accept `research = "templates/research"` as shorthand for `{ path = ... }`."""
        if isinstance(value, str):
            return {"path": value}
        return value
