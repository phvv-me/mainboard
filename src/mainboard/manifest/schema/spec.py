from typing import Self

from patos import FlexModel
from pydantic import ConfigDict, model_validator

type Json = str | int | float | bool | None | list["Json"] | dict[str, "Json"]


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

        over: the lower-precedence spec being overlaid.
        """
        base = {"version": over.version, **(over.model_extra or {})}
        top = {"version": self.version, **(self.model_extra or {})}
        if top["version"] == "*":
            top.pop("version")
        return type(self).model_validate({**base, **top})
