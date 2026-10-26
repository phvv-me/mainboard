from collections.abc import Set
from typing import Self

from patos import FlexModel
from pydantic import ConfigDict, model_validator

type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]

_SOURCES = {"path", "git", "url"}
_SOURCE_FIELDS = _SOURCES | {"branch", "tag", "rev", "subdirectory", "index"}


class Spec(FlexModel):
    """One dependency requirement; unknown keys (`path`, `git`, pins) reach the solver untyped."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = "*"

    @model_validator(mode="before")
    @classmethod
    def from_bare_string(cls, value: Json) -> Json:
        """Accept `torch = ">=2.9"` as shorthand for `{ version = ">=2.9" }`."""
        if isinstance(value, str):
            return {"version": value}
        return value

    @property
    def is_editable(self) -> bool:
        return bool((self.model_extra or {}).get("editable"))

    @property
    def is_path(self) -> bool:
        return "path" in (self.model_extra or {})

    @staticmethod
    def _superseded(declared: Set[str], version: str) -> set[str]:
        """The inherited fields a spec declaring `declared` keys at `version` drops."""
        if _SOURCES & declared:
            return {"version", *_SOURCE_FIELDS}
        if version != "*":
            return {*_SOURCE_FIELDS, "editable"}
        return set()

    def merged(self, over: Self) -> Self:
        """This spec layered over `over`, later keys winning key-by-key.

        A source and a registry version are alternatives, so declaring either on top drops the
        other's inherited coordinates first.
        """
        base = {"version": over.version, **(over.model_extra or {})}
        extras = self.model_extra or {}
        top = {"version": self.version, **extras}
        for field in self._superseded(extras.keys(), self.version):
            base.pop(field, None)
        if top["version"] == "*":
            top.pop("version")
        return type(self).model_validate({**base, **top})
