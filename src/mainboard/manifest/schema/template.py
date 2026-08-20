from pydantic import model_validator

from ...core.base import Declared
from .spec import Json


class Template(Declared):
    """One project template this workspace keeps, under the name `new` renders it by.

    path: where the template lives, a workspace-relative directory or any location the renderer
        itself resolves, a git URL included.
    into: the directory a rendered project lands under, workspace-relative, the workspace root
        itself when empty.
    answers: the questions this workspace always answers the same way, which is half of what
        turns a questionnaire into one argument.
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
