from typing import Self

from patos import FlexModel
from pydantic import ConfigDict, model_validator

type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]

_SOURCES = {"path", "git", "url"}
_SOURCE_FIELDS = _SOURCES | {"branch", "tag", "rev", "subdirectory", "index"}


class Spec(FlexModel):
    """One dependency requirement: a version string or a table with extras.

    Unknown keys (`path`, `editable`, `git`, `index`, channel pins) ride through
    to the solver untyped, so the manifest never lags a solver feature.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = "*"

    @property
    def is_editable(self) -> bool:
        """Whether this requirement is an editable local install."""
        return bool((self.model_extra or {}).get("editable"))

    @property
    def is_path(self) -> bool:
        """Whether this requirement points at a local path dependency."""
        return "path" in (self.model_extra or {})

    @model_validator(mode="before")
    @classmethod
    def from_bare_string(cls, value: Json) -> Json:
        """Accept `torch = ">=2.9"` as shorthand for `{ version = ">=2.9" }`."""
        if isinstance(value, str):
            return {"version": value}
        return value

    def merged(self, over: Self) -> Self:
        """This spec layered over `over`, later keys winning key-by-key.

        A location source and a registry version are alternative requirements, not two keys that
        narrow each other. Declaring either on the upper layer therefore removes the other's
        inherited source coordinates before ordinary extras are merged.

        over: the lower-precedence spec being overlaid.
        """
        base = {"version": over.version, **(over.model_extra or {})}
        extras = self.model_extra or {}
        top = {"version": self.version, **extras}
        if _SOURCES & extras.keys():
            base.pop("version", None)
            for field in _SOURCE_FIELDS:
                base.pop(field, None)
        elif self.version != "*":
            for field in _SOURCE_FIELDS | {"editable"}:
                base.pop(field, None)
        if top["version"] == "*":
            top.pop("version")
        return type(self).model_validate({**base, **top})
